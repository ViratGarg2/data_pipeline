"""
Tokenise datasets from HuggingFace and save as .bin for training.

Memory-efficient: streams from HF, tokenizes in batches via multiprocessing,
and appends to disk incrementally — never holds the full dataset in RAM.

Supports two modes:
  1. Named config (e.g. allenai/paloma with config c4_100_domains):
       python hf_tokenisation.py --repo allenai/paloma --config c4_100_domains --split val
  2. Data-files / subdir (e.g. your own parquet repo):
       python hf_tokenisation.py --repo ViratGarg/pile_data --subdir processed --split train
"""

import argparse
import multiprocessing
import os
import struct
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
from huggingface_hub import login

# Suppress "Token indices sequence length is longer than..." warnings.
# This is expected — the training script samples fixed-length windows from the .bin file.
hf_logging.set_verbosity_error()

from cs336_data.config import HUGGINGFACE_KEY

# ============================================================
# Worker initializer for multiprocessing
# ============================================================
# Each worker process gets its own tokenizer copy via the initializer,
# avoiding the need for a global that must survive fork().
_worker_tokenizer = None
_worker_eos_id = None


def _pool_initializer(tokenizer_name: str):
    """Called once per worker process to load the tokenizer."""
    global _worker_tokenizer, _worker_eos_id
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _worker_eos_id = _worker_tokenizer.eos_token_id


def _tokenize_one(text: str) -> list:
    """Tokenize a single text string and append EOS (runs in worker)."""
    if not text or not isinstance(text, str):
        return []
    return _worker_tokenizer.encode(text, add_special_tokens=False) + [_worker_eos_id]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenise HuggingFace parquet data and save as .bin"
    )
    parser.add_argument(
        "--repo", type=str, default="allenai/paloma",
        help="HuggingFace dataset repo (default: allenai/paloma)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Named dataset config, e.g. c4_100_domains for allenai/paloma. "
             "When set, --subdir is ignored and load_dataset uses this as the config name.",
    )
    parser.add_argument(
        "--subdir", type=str, default=None,
        help="Folder inside the HF repo containing data files (for repos without named configs)",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Dataset split to load, e.g. val, test, train (default: val)",
    )
    parser.add_argument(
        "--output", type=str, default=str(Path(__file__).resolve().parent / "val_train.bin"),
        help="Output .bin file path",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="HuggingFace tokenizer name (default: gpt2)",
    )
    parser.add_argument(
        "--start-file", type=int, default=0,
        help="Starting file index, e.g. 40 to start from train-00040-... (default: 0)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of parquet files to process from start-file (default: all)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum number of rows to process (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="Number of texts to accumulate before sending to workers (default: 5000)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Chunk size for multiprocessing imap (default: 500)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of multiprocessing workers (default: cpu_count)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_workers = args.workers or multiprocessing.cpu_count()

    # ----------------------------------------------------------
    # Authenticate with HuggingFace
    # ----------------------------------------------------------
    print("🔑 Logging into HuggingFace...")
    if HUGGINGFACE_KEY:  # otherwise use the cached `huggingface-cli login` token
        login(token=HUGGINGFACE_KEY)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("=" * 60)
    print("TOKENISATION CONFIG")
    print("=" * 60)
    print(f"  Repo:       {args.repo}")
    if args.config:
        print(f"  Config:     {args.config}")
    if args.subdir:
        print(f"  Subdir:     {args.subdir}")
    print(f"  Split:      {args.split}")
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  Start file: {args.start_file}")
    print(f"  Max files:  {args.max_files or 'all'}")
    print(f"  Max rows:   {args.max_rows or 'all'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Workers:    {num_workers}")
    print(f"  Output:     {args.output}")
    print("=" * 60)
    print()

    # ----------------------------------------------------------
    # Determine loading mode: named config vs data_files
    # ----------------------------------------------------------
    use_config_mode = args.config is not None
    data_files = None

    if use_config_mode:
        # Named config mode (e.g. allenai/paloma + c4_100_domains)
        # No file slicing needed — HF handles the config internally.
        print(f"📂 Using named config: {args.config} (split={args.split})")
    else:
        # Data-files mode (e.g. ViratGarg/pile_data with subdir=processed)
        subdir = args.subdir or ""
        if args.max_files is not None or args.start_file > 0:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_files = api.list_repo_files(args.repo, repo_type="dataset")
            # Match any data file type: .parquet, .jsonl.gz, .json.gz, etc.
            data_file_list = sorted(
                f for f in repo_files
                if (not subdir or f.startswith(subdir + "/"))
                and any(f.endswith(ext) for ext in (".parquet", ".jsonl.gz", ".json.gz", ".jsonl", ".json"))
            )
            # Slice from start_file index
            data_file_list = data_file_list[args.start_file:]
            # Then limit to max_files if set
            if args.max_files is not None:
                data_file_list = data_file_list[: args.max_files]
            if not data_file_list:
                print(f"❌ No data files found in {args.repo}/{subdir}/")
                return
            print(f"📂 Selected {len(data_file_list)} file(s):")
            for f in data_file_list:
                print(f"     {f}")
            print()
            data_files = data_file_list
        else:
            if subdir:
                data_files = f"{subdir}/*"
            # else: no data_files, load entire repo

    # ----------------------------------------------------------
    # Stream dataset from HuggingFace
    # ----------------------------------------------------------
    print("📥 Streaming from HuggingFace...")
    if use_config_mode:
        dataset = load_dataset(
            args.repo,
            args.config,
            split=args.split,
            streaming=True,
        )
    elif data_files:
        dataset = load_dataset(
            args.repo,
            data_files=data_files,
            split=args.split,
            streaming=True,
        )
    else:
        dataset = load_dataset(
            args.repo,
            split=args.split,
            streaming=True,
        )

    # ----------------------------------------------------------
    # Stream → batch → tokenize → append to disk
    # ----------------------------------------------------------
    # We never hold more than `batch_size` texts + their tokens in RAM.
    # Each batch is tokenized via multiprocessing and flushed to the
    # output file before the next batch is collected.
    # ----------------------------------------------------------
    pool = multiprocessing.Pool(
        num_workers,
        initializer=_pool_initializer,
        initargs=(args.tokenizer,),
    )

    total_tokens = 0
    total_texts = 0
    total_bytes = 0
    num_skipped = 0
    batch = []
    hit_limit = False

    # Open output file — append if resuming from a non-zero start, else overwrite
    file_mode = "ab" if args.start_file > 0 else "wb"
    if file_mode == "ab" and os.path.exists(args.output):
        existing_size = os.path.getsize(args.output)
        print(f"📎 Appending to existing file ({existing_size / (1024**2):.1f} MB)")
    elif file_mode == "ab":
        print("⚠️  --start-file > 0 but output file does not exist; creating new file")

    with open(args.output, file_mode) as f_out:
        for example in tqdm(dataset, desc="Streaming rows"):
            text = example.get("text", None)
            if text and isinstance(text, str) and text.strip():
                text = text.strip()
                batch.append(text)
            else:
                num_skipped += 1
                continue

            # Check max-rows limit
            if args.max_rows and (total_texts + len(batch)) >= args.max_rows:
                # Trim batch to exact limit
                remaining = args.max_rows - total_texts
                batch = batch[:remaining]
                hit_limit = True

            # Flush batch when full or when we've hit the limit
            if len(batch) >= args.batch_size or hit_limit:
                tokens_written = _flush_batch(batch, pool, args.chunk_size, f_out)
                total_bytes += sum(len(t.encode("utf-8")) for t in batch)
                total_texts += len(batch)
                total_tokens += tokens_written
                batch = []

                if hit_limit:
                    print(f"\n  Reached max-rows limit ({args.max_rows})")
                    break

        # Flush remaining texts in the last partial batch
        if batch:
            tokens_written = _flush_batch(batch, pool, args.chunk_size, f_out)
            total_bytes += sum(len(t.encode("utf-8")) for t in batch)
            total_texts += len(batch)
            total_tokens += tokens_written

    pool.close()
    pool.join()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    file_size_mb = os.path.getsize(args.output) / (1024 ** 2)

    print()
    print("=" * 60)
    print("TOKENISATION COMPLETE")
    print("=" * 60)
    print(f"  Texts processed: {total_texts:,}")
    print(f"  Skipped (empty): {num_skipped:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Total bytes:     {total_bytes:,}")
    if total_bytes > 0:
        print(f"  LT/LB ratio:    {total_tokens / total_bytes:.4f}")
    print(f"  Dtype:           uint16")
    print(f"  Output:          {args.output} ({file_size_mb:.1f} MB)")
    print("=" * 60)


def _flush_batch(batch, pool, chunk_size, f_out):
    """Tokenize a batch via the pool and write uint16 tokens to f_out.

    Returns the number of tokens written.
    """
    tokens_written = 0
    for token_ids in pool.imap(
        _tokenize_one, batch, chunksize=chunk_size
    ):
        if token_ids:
            arr = np.array(token_ids, dtype=np.uint16)
            arr.tofile(f_out)
            tokens_written += len(token_ids)
    return tokens_written


if __name__ == "__main__":
    main()