"""
Build token-matched filtered / unfiltered training data from the same Pile shards.

For every source shard i:
  filtered   = process_html_pipeline output of shard i
  unfiltered = a seeded random sample of raw documents from the same shard i,
               drawn until it has exactly as many GPT-2 tokens as the filtered
               version (the last sampled document is truncated to hit the count).

So both datasets come from identical source files and contain the same number
of tokens shard by shard. Each side is written as:
  text/train-XXXXX.parquet     documents
  tokens/train-XXXXX.bin       uint16 GPT-2 token ids, <|endoftext|> after each doc
  manifests/train-XXXXX.json   counts, seed, source file
and uploaded to its own HF dataset repo. To train, concatenate the token files:
  cat tokens/train-*.bin > train.bin

Usage:
  python cs336_data/prepare_matched_data.py --shards 210-238 \\
      --filtered-dir /ssd_scratch/.../pipeline_out \\
      --stage-dir /ssd_scratch/.../stage \\
      --filtered-repo ViratGarg/pile_filtered_13B --unfiltered-repo ViratGarg/pile_unfiltered_13B
"""

import argparse
import functools
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, logging as hf_logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_data import config

# Long documents are expected; the trainer samples fixed-length windows.
hf_logging.set_verbosity_error()

RAW_DIR = "/data3/dataset/the_pile_deduplicated/data"
CHUNK_DOCS = 1000

# Pass the token explicitly so a stale cached HF login is never used.
tokenizer = AutoTokenizer.from_pretrained("gpt2", token=config.HUGGINGFACE_KEY)
EOS = tokenizer.eos_token_id


def tokenize(texts: list[str]) -> list[list[int]]:
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [x + [EOS] for x in ids]


def shard_file(directory: str, shard: int) -> str:
    matches = sorted(glob.glob(os.path.join(directory, f"train-{shard:05d}-of-*.parquet")))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file for shard {shard} in {directory}, found {matches}")
    return matches[0]


@functools.cache
def _hf_files(repo_id: str) -> tuple[str, ...]:
    from huggingface_hub import HfApi

    return tuple(HfApi(token=config.HUGGINGFACE_KEY).list_repo_files(repo_id, repo_type="dataset"))


def fetch_filtered_from_hf(repo_id: str, prefix: str, shard: int, dest_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    name = f"train-{shard:05d}-of-"
    files = [f for f in _hf_files(repo_id) if f.startswith(f"{prefix}/{name}") and f.endswith(".parquet")]
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one file for shard {shard} in {repo_id}/{prefix}, found {files}")
    path = hf_hub_download(
        repo_id, files[0], repo_type="dataset", token=config.HUGGINGFACE_KEY, local_dir=dest_dir,
    )
    return path


def write_filtered(src_parquet: str, out_text: str, out_tokens: str) -> dict:
    """Tokenize the filtered shard; returns doc and token counts."""
    shutil.copyfile(src_parquet, out_text)
    source = pq.ParquetFile(src_parquet)
    n_docs = n_tokens = 0
    with open(out_tokens, "wb") as f:
        for batch in source.iter_batches(batch_size=CHUNK_DOCS, columns=["text"]):
            texts = [t for t in batch.column(0).to_pylist() if t]
            for ids in tokenize(texts):
                np.asarray(ids, dtype=np.uint16).tofile(f)
                n_tokens += len(ids)
            n_docs += len(texts)
    # Output columns show which pipeline stages ran (e.g. nsfw_label => content classification).
    return {"docs": n_docs, "tokens": n_tokens, "pipeline_columns": source.schema_arrow.names}


def write_unfiltered(raw_parquet: str, target_tokens: int, seed: int, out_text: str, out_tokens: str) -> dict:
    """Sample raw docs in seeded random order until exactly target_tokens are written."""
    texts = pq.read_table(raw_parquet, columns=["text"]).column("text").to_pylist()
    order = np.random.default_rng(seed).permutation(len(texts))

    rows, kept_texts = [], []
    n_tokens = 0
    truncated_last = False
    with open(out_tokens, "wb") as f:
        for start in range(0, len(order), CHUNK_DOCS):
            idx = [int(i) for i in order[start:start + CHUNK_DOCS] if texts[i]]
            for i, ids in zip(idx, tokenize([texts[i] for i in idx])):
                remaining = target_tokens - n_tokens
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    truncated_last = True
                np.asarray(ids, dtype=np.uint16).tofile(f)
                n_tokens += len(ids)
                rows.append(i)
                kept_texts.append(texts[i])
                if n_tokens == target_tokens:
                    break
            if n_tokens == target_tokens:
                break

    if n_tokens < target_tokens:
        raise RuntimeError(f"{raw_parquet} has only {n_tokens} tokens, needed {target_tokens}")

    pq.write_table(
        pa.table({"text": kept_texts, "source_row": pa.array(rows, type=pa.int32())}),
        out_text, compression="zstd",
    )
    return {
        "docs": len(rows),
        "tokens": n_tokens,
        "raw_docs_in_shard": len(texts),
        "sampled_pct_of_raw_docs": 100.0 * len(rows) / len(texts),
        "last_doc_truncated": truncated_last,
    }


def upload(repo_id: str, folder: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=config.HUGGINGFACE_KEY)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    # upload_large_folder batches commits and retries, which avoids the hub's
    # per-repo commit rate limit when many array tasks upload at once. Few workers:
    # with many parallel streams per task, some uploads stalled for hours.
    api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="dataset", num_workers=4)


def parse_shards(spec: str) -> list[int]:
    """'0-71,80-114,115' -> [0, ..., 71, 80, ..., 115]"""
    shards = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        shards.extend(range(int(lo), int(hi or lo) + 1))
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True, help="Shard ranges, e.g. 210-238 or 0-71,80-114")
    parser.add_argument("--filtered-dir", help="Local directory with pipeline outputs for these shards")
    parser.add_argument("--filtered-hf-repo", help="Instead of --filtered-dir, download pipeline outputs from this HF dataset")
    parser.add_argument("--filtered-hf-prefix", default="processed")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--stage-dir", required=True, help="Scratch directory for outputs before upload")
    parser.add_argument("--filtered-repo", required=True)
    parser.add_argument("--unfiltered-repo", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; shard i samples with seed + i")
    parser.add_argument("--no-upload", action="store_true", help="Keep outputs in --stage-dir, do not upload")
    parser.add_argument(
        "--upload-sides", default="filtered,unfiltered",
        help="Which sides to upload, e.g. 'unfiltered' when the filtered side is already on HF",
    )
    args = parser.parse_args()

    if not (args.filtered_dir or args.filtered_hf_repo):
        parser.error("one of --filtered-dir or --filtered-hf-repo is required")

    sides = {"filtered": args.filtered_repo, "unfiltered": args.unfiltered_repo}
    for side in sides:
        for sub in ("text", "tokens", "manifests"):
            os.makedirs(os.path.join(args.stage_dir, side, sub), exist_ok=True)

    for shard in parse_shards(args.shards):
        t0 = time.time()
        name = f"train-{shard:05d}"
        if args.filtered_hf_repo:
            src = fetch_filtered_from_hf(
                args.filtered_hf_repo, args.filtered_hf_prefix, shard, os.path.join(args.stage_dir, "download"),
            )
        else:
            src = shard_file(args.filtered_dir, shard)
        raw = shard_file(args.raw_dir, shard)

        filtered = write_filtered(
            src,
            os.path.join(args.stage_dir, "filtered", "text", f"{name}.parquet"),
            os.path.join(args.stage_dir, "filtered", "tokens", f"{name}.bin"),
        )
        unfiltered = write_unfiltered(
            raw, filtered["tokens"], args.seed + shard,
            os.path.join(args.stage_dir, "unfiltered", "text", f"{name}.parquet"),
            os.path.join(args.stage_dir, "unfiltered", "tokens", f"{name}.bin"),
        )
        manifest = {
            "shard": shard,
            "source_file": os.path.basename(raw),
            "seed": args.seed + shard,
            "tokenizer": "gpt2",
            "token_dtype": "uint16",
            "filtered": filtered,
            "unfiltered": unfiltered,
            "seconds": time.time() - t0,
        }
        for side in sides:
            with open(os.path.join(args.stage_dir, side, "manifests", f"{name}.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"{name}: {filtered['docs']:,} filtered docs / {unfiltered['docs']:,} raw docs, "
              f"{filtered['tokens']:,} tokens each ({manifest['seconds']:.0f}s)", flush=True)

    shutil.rmtree(os.path.join(args.stage_dir, "download"), ignore_errors=True)
    if not args.no_upload:
        for side in args.upload_sides.split(","):
            repo = sides[side]
            print(f"Uploading {side} -> {repo}", flush=True)
            upload(repo, os.path.join(args.stage_dir, side))


if __name__ == "__main__":
    main()
