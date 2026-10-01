"""Compare the filtered and unfiltered 25k-step models.

For each model:
  * loss, perplexity and next-token accuracy on Paloma C4 (the training-time validation set)
    and on held-out raw Pile text (a deduplicated-Pile shard neither model trained on);
  * one generation per prompt for 50 prompts (same prompt, seed and sampling for both models),
    each scored for fluency by the original GPT-2 (perplexity of the generated continuation).

Writes results/compare_25k_<time>.csv (one row per prompt, both models' outputs) and
results/compare_25k_summary_<time>.json.

    python inference_check/compare_models.py <work_dir> <paloma_val.bin>
"""
import csv, json, os, subprocess, sys, time

MIN_FREE_GPU_GB = 3.0  # leave headroom for other users' jobs on a shared GPU


def _free_gpu_gb() -> float:
    """Free memory on GPU 0 via nvidia-smi (asking CUDA would itself reserve GPU memory)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0]) / 1024 if out.returncode == 0 else 0.0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0.0


USE_GPU = _free_gpu_gb() >= MIN_FREE_GPU_GB
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU mode: never touch the shared GPU
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cs336-basics"))
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, GPT2LMHeadModel
from cs336_basics.model import BasicsTransformerLM

MODELS = {"filtered": "ViratGarg/cs336-a4-filtered-25k", "unfiltered": "ViratGarg/cs336-a4-unfiltered-25k"}
HELDOUT_PILE_REPO = "EleutherAI/the_pile_deduplicated"
HELDOUT_SHARD = 1649  # the matched datasets use shards 0-269; benchmarks used 300-315
CONTEXT = 512
TEMPERATURE, TOP_K, MAX_NEW_TOKENS, SEED = 0.7, 50, 100, 1234

PROMPTS = [
    # Narrative
    "Once upon a time, there was a little girl who",
    "Linda was on a walk in the park when",
    "The old man opened the letter and",
    "It was raining heavily when the train finally arrived at the station, and",
    "My grandmother always told me that",
    "The detective looked at the broken window and said,",
    "On the first day of school, Maya",
    "When the power went out across the city,",
    # News and current affairs
    "The stock market fell sharply today after",
    "The city council voted on Tuesday to",
    "Officials said the wildfire had",
    "The new smartphone, released this week,",
    "Doctors are warning that the flu season",
    "The football team won the championship after",
    # Science and nature
    "Scientists have discovered a new species of",
    "The human heart pumps blood by",
    "Photosynthesis is the process by which plants",
    "Black holes are regions of space where",
    "Climate change affects ocean temperatures because",
    "The theory of evolution explains how",
    "Vaccines work by training the immune system to",
    "The Moon causes tides on Earth because",
    # History and society
    "The history of the Roman Empire",
    "During the Industrial Revolution,",
    "The French Revolution began in 1789 when",
    "Mahatma Gandhi is remembered for",
    "The printing press changed Europe by",
    "World War II ended in 1945 after",
    # Technology and AI
    "In machine learning, a neural network is",
    "The internet was originally designed to",
    "A computer's CPU is responsible for",
    "Large language models are trained on",
    "Cloud computing allows companies to",
    "The main difference between RAM and storage is",
    # Instructions and advice
    "The best way to cook pasta is",
    "To change a flat tire, you should first",
    "Here are three tips for staying productive while working from home:",
    "If you want to learn a new language, start by",
    "To make a good cup of coffee,",
    "Before going on a long hike, it is important to",
    # Opinion, dialogue and everyday text
    "I think the most important quality in a friend is",
    "Dear Sir or Madam, I am writing to complain about",
    "\"Where have you been?\" she asked.",
    "The restaurant was crowded, but the food",
    "Our company is looking for a software engineer who",
    "Thank you for your order. Your package will",
    # Code and structured text
    "def fibonacci(n):",
    "The following Python function returns the largest element of a list:",
    "Ingredients: 2 cups of flour,",
    "Q: What is the capital of France?\nA:",
]
assert len(PROMPTS) == 50

work = sys.argv[1]
paloma_path = sys.argv[2]
os.makedirs(work, exist_ok=True)
dev = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
eval_batches, eval_bs = (200, 16) if dev == "cuda" else (40, 8)
print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
tok = AutoTokenizer.from_pretrained("gpt2")
EOS = tok.eos_token_id


def heldout_pile_tokens() -> np.ndarray:
    """GPT-2 tokens of a raw deduplicated-Pile shard outside the training shards (with EOS)."""
    path = os.path.join(work, "heldout_pile.bin")
    if os.path.exists(path):
        return np.memmap(path, dtype=np.uint16, mode="r")
    name = next(f for f in HfApi().list_repo_files(HELDOUT_PILE_REPO, repo_type="dataset")
                if f.startswith(f"data/train-{HELDOUT_SHARD:05d}-of-") and f.endswith(".parquet"))
    local = hf_hub_download(HELDOUT_PILE_REPO, name, repo_type="dataset", local_dir=os.path.join(work, "pile"))
    texts = pq.read_table(local, columns=["text"]).column("text").to_pylist()
    texts = [texts[i] for i in np.random.default_rng(0).permutation(len(texts))[:4000] if texts[i]]
    ids = [t + [EOS] for t in tok(texts, add_special_tokens=False)["input_ids"]]
    arr = np.fromiter((x for d in ids for x in d), dtype=np.uint16)
    arr.tofile(path)
    os.remove(local)
    print(f"held-out Pile: {len(texts)} docs, {len(arr):,} tokens from {name}", flush=True)
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_model(repo: str) -> BasicsTransformerLM:
    d = os.path.join(work, repo.split("/")[1])
    for f in ("model_config.json", "model.pt", "checkpoint_info.json"):
        hf_hub_download(repo, f, local_dir=d)
    with open(os.path.join(d, "model_config.json")) as f:
        model = BasicsTransformerLM(**json.load(f))
    state = torch.load(os.path.join(d, "model.pt"), map_location="cpu")
    model.load_state_dict({k.split("_orig_mod.", 1)[-1]: v for k, v in state.items()})
    return model.to(dev).eval()


@torch.no_grad()
def evaluate(model, data: np.ndarray) -> dict:
    """Mean next-token loss / accuracy over the same random 512-token windows for every model."""
    g = torch.Generator().manual_seed(0)
    nll = correct = count = 0.0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    for _ in range(eval_batches):
        idx = torch.randint(len(data) - CONTEXT - 1, (eval_bs,), generator=g)
        batch = torch.stack([torch.from_numpy(data[i:i + CONTEXT + 1].astype(np.int64)) for i in idx]).to(dev)
        with amp:
            logits = model(batch[:, :-1]).float()
        target = batch[:, 1:]
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == target).sum().item()
        count += target.numel()
    loss = nll / count
    return {"loss": loss, "perplexity": float(np.exp(loss)), "next_token_accuracy": correct / count, "tokens": int(count)}


eval_sets = {"paloma_c4": np.memmap(paloma_path, dtype=np.uint16, mode="r"), "heldout_pile": heldout_pile_tokens()}
summary = {"device": dev, "eval_tokens_per_set": eval_batches * eval_bs * CONTEXT,
           "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
           "models": {}}
generations = {}
for arm, repo in MODELS.items():
    t0 = time.time()
    model = load_model(repo)
    info = json.load(open(os.path.join(work, repo.split("/")[1], "checkpoint_info.json")))
    res = {"repo": repo, "train_steps": info.get("step"), "final_val_loss_logged": info.get("eval_loss")}
    for name, data in eval_sets.items():
        res[name] = evaluate(model, data)
        print(f"{arm:10s} {name:12s} loss {res[name]['loss']:.4f}  ppl {res[name]['perplexity']:.2f}  "
              f"acc {100 * res[name]['next_token_accuracy']:.2f}%", flush=True)
    outs = []
    for i, prompt in enumerate(PROMPTS):
        torch.manual_seed(SEED + i)  # same seed per prompt for both models
        ids = torch.tensor(tok.encode(prompt), device=dev)
        with torch.no_grad():
            out = model.generate(ids, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, eos_token_id=EOS)
        outs.append(tok.decode(out[0].tolist()))
    generations[arm] = outs
    summary["models"][arm] = res
    print(f"{arm}: 50 generations done ({time.time() - t0:.0f}s total)", flush=True)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

# Neutral fluency score: perplexity of each generated continuation under the original GPT-2 (124M).
judge = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()


@torch.no_grad()
def gpt2_ppl(prompt: str, continuation: str) -> float | None:
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(continuation)[:MAX_NEW_TOKENS]
    if not c_ids:
        return None
    x = torch.tensor([p_ids + c_ids], device=dev)
    logits = judge(x).logits[0, len(p_ids) - 1:-1].float()
    return float(torch.exp(F.cross_entropy(logits, torch.tensor(c_ids, device=dev))))


rows = []
for i, prompt in enumerate(PROMPTS):
    row = {"prompt_id": i + 1, "prompt": prompt}
    for arm in MODELS:
        row[f"{arm}_generation"] = generations[arm][i]
        row[f"{arm}_gpt2_ppl"] = gpt2_ppl(prompt, generations[arm][i])
    rows.append(row)
for arm in MODELS:
    vals = [r[f"{arm}_gpt2_ppl"] for r in rows if r[f"{arm}_gpt2_ppl"] is not None]
    summary["models"][arm]["generation_gpt2_ppl"] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)), "n": len(vals)}
summary["prompts_where_filtered_more_fluent"] = sum(
    1 for r in rows if r["filtered_gpt2_ppl"] is not None and r["unfiltered_gpt2_ppl"] is not None
    and r["filtered_gpt2_ppl"] < r["unfiltered_gpt2_ppl"])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(out_dir, f"compare_25k_{stamp}.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summary["csv"] = os.path.basename(csv_path)
with open(os.path.join(out_dir, f"compare_25k_summary_{stamp}.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"Saved {len(rows)} rows to {csv_path}")
