"""Load the earlier model (ViratGarg/gpt-model), check its Paloma C4 loss, and sample text.

Uses the GPU only when it has enough free memory (shared machines), otherwise the CPU.
Writes results/prev_model_samples_<time>.csv (prompt, sample, generation) and a summary JSON.

    python inference_check/infer_prev.py <model_dir>
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cs336-basics"))
import numpy as np, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from cs336_basics.model import BasicsTransformerLM

try:  # the model repo is public; a token only avoids anonymous rate limits
    from cs336_data import config
    HF_TOKEN = config.HUGGINGFACE_KEY
except Exception:
    HF_TOKEN = None

stage = sys.argv[1]
if USE_GPU and torch.cuda.is_available():
    dev, eval_batches, eval_bs = "cuda", 200, 16
else:
    dev, eval_batches, eval_bs = "cpu", 40, 8
    torch.set_num_threads(min(8, os.cpu_count() or 1))
node = os.uname().nodename
print(f"node {node}, device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)

for f in ("model_config.json", "model.pt", "paloma_c4_100_val_tokenised.bin"):
    t = time.time(); hf_hub_download("ViratGarg/gpt-model", f, local_dir=stage, token=HF_TOKEN)
    print(f"downloaded {f} in {time.time()-t:.0f}s", flush=True)

# Same as BasicsTransformerLM.from_pretrained, but weights are loaded onto the CPU first
# so CPU mode never allocates memory on a shared GPU.
with open(os.path.join(stage, "model_config.json")) as f:
    model = BasicsTransformerLM(**json.load(f))
state = torch.load(os.path.join(stage, "model.pt"), map_location="cpu")
state = {k.split("_orig_mod.", 1)[-1]: v for k, v in state.items()}
model.load_state_dict(state)
model = model.to(dev).eval()
print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), flush=True)

# Validation loss on Paloma C4 (same estimator as train.py: random 512-token windows).
val = np.memmap(f"{stage}/paloma_c4_100_val_tokenised.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(0)
losses = []
amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
t = time.time()
with torch.no_grad(), amp:
    for _ in range(eval_batches):
        idx = torch.randint(len(val) - 513, (eval_bs,), generator=g)
        batch = torch.stack([torch.from_numpy(val[i:i + 513].astype(np.int64)) for i in idx]).to(dev)
        logits = model(batch[:, :-1])
        losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)).item())
loss = float(np.mean(losses))
eval_tokens = eval_batches * eval_bs * 512
print(f"Paloma C4 val loss ({eval_tokens:,} tokens): {loss:.3f} | perplexity {np.exp(loss):.1f} ({time.time()-t:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained("gpt2", token=HF_TOKEN)
prompts = [
    "Linda was on a walk in the park",
    "The history of the Roman Empire",
    "In machine learning, a neural network is",
    "Once upon a time, there was a little girl who",
    "The best way to cook pasta is",
    "Scientists have discovered a new species of",
    "The stock market fell sharply today after",
    "def fibonacci(n):",
]
SAMPLES_PER_PROMPT, TEMPERATURE, TOP_K, MAX_NEW_TOKENS = 3, 0.7, 50, 80
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(out_dir, f"prev_model_samples_{stamp}.csv")

torch.manual_seed(1234)
rows = []
for p in prompts:
    ids = torch.tensor(tok.encode(p), device=dev)
    for k in range(SAMPLES_PER_PROMPT):
        with torch.no_grad():
            out = model.generate(ids, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, eos_token_id=tok.eos_token_id)
        generation = tok.decode(out[0].tolist())
        rows.append({"prompt": p, "sample": k + 1, "generation": generation})
        if k == 0:
            print(f"\n### {p!r} (T={TEMPERATURE})\n{p}{generation}", flush=True)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prompt", "sample", "generation"])
    w.writeheader()
    w.writerows(rows)
summary = {
    "model": "ViratGarg/gpt-model", "node": node, "device": dev,
    "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
    "paloma_c4_val_loss": loss, "perplexity": float(np.exp(loss)), "eval_tokens": eval_tokens,
    "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "max_new_tokens": MAX_NEW_TOKENS,
                 "samples_per_prompt": SAMPLES_PER_PROMPT, "seed": 1234},
    "samples_csv": csv_path,
}
with open(os.path.join(out_dir, f"prev_model_summary_{stamp}.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved {len(rows)} prompt/generation rows to {csv_path}")
