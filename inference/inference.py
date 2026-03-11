"""
Inference script for the trained GPT model hosted on HuggingFace.

Downloads model.pt and model_config.json from ViratGarg/gpt-model,
loads the BasicsTransformerLM architecture, and generates text.

Usage:
    python inference/inference.py
    python inference/inference.py --prompt "Once upon a time"
    python inference/inference.py --prompt "The meaning of life is" --max-new-tokens 512 --temperature 0.8
    python inference/inference.py --device cpu  # for machines without GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

# Add cs336-basics to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cs336-basics"))
from cs336_basics.model import BasicsTransformerLM


HF_REPO = "ViratGarg/gpt-model"


def download_model(cache_dir: str = "inference/model_cache") -> str:
    """Download model.pt and model_config.json from HuggingFace.
    
    Returns the local directory containing both files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    files_to_download = [
        ("model_config.json", "Config"),
        ("model.pt", "Weights (~649MB)"),
    ]

    for filename, desc in tqdm(files_to_download, desc="📥 Downloading files", unit="file"):
        tqdm.write(f"  ⏳ Downloading {filename}...")
        t0 = time.time()
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=cache_dir,
        )
        elapsed = time.time() - t0
        tqdm.write(f"  ✅ {desc}: {filename} ({elapsed:.1f}s)")

    return cache_dir


def load_model(model_dir: str, device: str = "cuda") -> BasicsTransformerLM:
    """Load the model from local directory."""
    config_path = os.path.join(model_dir, "model_config.json")
    weights_path = os.path.join(model_dir, "model.pt")

    steps = ["Load config", "Init model", "Load weights", "Clean keys", "Move to device"]
    pbar = tqdm(steps, desc="🧠 Loading model", unit="step")

    # Step 1: Load config
    pbar.set_postfix_str("reading config")
    with open(config_path) as f:
        config = json.load(f)
    print(f"\n📐 Model config: {json.dumps(config, indent=2)}")
    pbar.update(1)

    # Step 2: Init model
    pbar.set_postfix_str("initializing architecture")
    model = BasicsTransformerLM(**config)
    pbar.update(1)

    # Step 3: Load weights from disk
    pbar.set_postfix_str("loading weights from disk")
    t0 = time.time()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    tqdm.write(f"  ⏱ Weights loaded in {time.time() - t0:.1f}s")
    pbar.update(1)

    # Step 4: Clean keys (remove _orig_mod. prefix from compiled checkpoints)
    pbar.set_postfix_str("cleaning state_dict keys")
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        idx = k.find(unwanted_prefix)
        if idx != -1:
            state_dict[k[idx + len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    pbar.update(1)

    # Step 5: Move to device
    pbar.set_postfix_str(f"moving to {device}")
    model.eval()
    model.to(device)
    pbar.update(1)

    pbar.set_postfix_str("done ✅")
    pbar.close()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model loaded: {num_params / 1e6:.1f}M parameters on {device}")

    return model


def generate(
    model: BasicsTransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt with a tqdm progress bar."""
    import torch.nn.functional as F

    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor(prompt_ids, device=device).unsqueeze(0)  # (1, seq_len)
    eos_token_id = tokenizer.eos_token_id
    context_length = model.context_length

    generated_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        pbar = tqdm(
            total=max_new_tokens,
            desc="🔤 Generating",
            unit="tok",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tokens [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for _ in range(max_new_tokens):
            # Crop to context length
            x_cond = x[:, -context_length:] if x.size(1) > context_length else x
            logits = model(x_cond)
            next_logits = logits[:, -1] / temperature

            if top_k:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = topk_vals[:, -1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                pbar.update(1)
                generated_tokens += 1
                break

            x = torch.cat((x, next_id), dim=-1)
            generated_tokens += 1
            pbar.update(1)

        elapsed = time.time() - t0
        pbar.close()
        tqdm.write(f"  ⏱ Generated {generated_tokens} tokens in {elapsed:.2f}s ({generated_tokens / elapsed:.1f} tok/s)")

    new_tokens = x[0, len(prompt_ids):].tolist()
    return tokenizer.decode(prompt_ids + new_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on trained GPT model")
    parser.add_argument(
        "--prompt", type=str, default="Hi, how are you all doing today?",
        help="Text prompt to start generation from",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature — lower = more deterministic (default: 0.7)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k sampling — only sample from top k tokens (default: 50)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples to generate (default: 3)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to local model directory (skips download if provided)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode — keep prompting for input",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("GPT MODEL INFERENCE")
    print("=" * 80)
    print(f"  Device:         {device}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Top-k:          {args.top_k}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 80)
    print()

    # Download or use local model
    if args.model_dir:
        model_dir = args.model_dir
        print(f"📂 Using local model: {model_dir}")
    else:
        print(f"📥 Downloading model from {HF_REPO}...")
        model_dir = download_model()
    print()

    # Load model and tokenizer
    model = load_model(model_dir, device=device)
    print("⏳ Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✅ Tokenizer ready")
    print()

    if args.interactive:
        # Interactive mode
        print("💬 Interactive mode — type your prompt and press Enter (Ctrl+C to quit)")
        print("-" * 80)
        while True:
            try:
                prompt = input("\n🟢 Prompt: ").strip()
                if not prompt:
                    continue
                output = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                )
                print(f"\n🤖 Generated:\n{output}")
                print("-" * 80)
            except KeyboardInterrupt:
                print("\n\n👋 Bye!")
                break
    else:
        # Batch mode — generate num_samples completions
        for i in range(args.num_samples):
            print(f"{'=' * 80}")
            print(f"  Sample {i + 1}/{args.num_samples}")
            print(f"{'=' * 80}")
            print(f"  Prompt: {args.prompt}")
            print(f"{'-' * 80}")
            output = generate(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            print(f"  Generated:\n{output}")
            print()


if __name__ == "__main__":
    main()
