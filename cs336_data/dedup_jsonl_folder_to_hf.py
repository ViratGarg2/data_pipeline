"""Deduplicate .jsonl.zst files in a folder and upload deduplicated outputs to HF.

Workflow:
1. Discover all ``*.jsonl.zst`` files from an input folder.
2. Run MinHash+LSH deduplication using local ``lsh`` tabular dedup API.
3. Rebuild each deduplicated file one-by-one into a temporary file.
4. Upload each deduplicated file to a Hugging Face dataset repo via HTTP.
5. Delete the temporary deduplicated file after successful upload.
"""

import argparse
import io
import json
import os
import posixpath
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# Ensure local sibling `lsh/` package is importable when script is run directly.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load_env_file(path):
    """Parse simple KEY=VALUE entries from a .env file."""
    values = {}
    if not path.exists():
        return values

    with open(str(path), "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            values[key] = value
    return values


def _resolve_hf_token(env_values):
    """Resolve HF token preferring .env values, then process env."""
    token_keys = ["HUGGINGFACE_KEY", "HUGGINGFACE_TOKEN", "HF_TOKEN"]

    for key in token_keys:
        value = env_values.get(key)
        if value:
            return value

    for key in token_keys:
        value = os.getenv(key, "")
        if value:
            return value

    raise ValueError(
        "Hugging Face token not found. Set HUGGINGFACE_KEY (or HUGGINGFACE_TOKEN/HF_TOKEN) in .env."
    )


def _resolve_repo_id(cli_repo_id, env_values):
    """Resolve dataset repo id from CLI, .env, then process env."""
    if cli_repo_id:
        return cli_repo_id

    repo_keys = ["HUGGINGFACE_DATASET_REPO", "HF_DATASET_REPO"]

    for key in repo_keys:
        value = env_values.get(key)
        if value:
            return value

    for key in repo_keys:
        value = os.getenv(key, "")
        if value:
            return value

    raise ValueError(
        "HF dataset repo id not found. Pass --hf-repo-id or set HUGGINGFACE_DATASET_REPO in .env."
    )


def _open_text_file(path, mode):
    """Open text files with transparent .zst handling."""
    path_str = str(path)
    if path_str.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required for .zst files. Install with: pip install zstandard"
            ) from exc

        if mode == "rt":
            raw_f = open(path_str, "rb")
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(raw_f)
            return io.TextIOWrapper(reader, encoding="utf-8")
        if mode == "wt":
            raw_f = open(path_str, "wb")
            cctx = zstd.ZstdCompressor(level=3)
            writer = cctx.stream_writer(raw_f)
            return io.TextIOWrapper(writer, encoding="utf-8")
        raise ValueError("Unsupported mode for text open: {}".format(mode))

    try:
        from xopen import xopen
    except ImportError:
        if mode == "rt":
            return open(path_str, "r", encoding="utf-8")
        if mode == "wt":
            return open(path_str, "w", encoding="utf-8")
        raise ValueError("Unsupported mode for text open: {}".format(mode))

    return xopen(path_str, mode)


def _ensure_dataset_repo_http(api, repo_id, hf_token):
    """Create dataset repo if needed; tolerate old/new HF API signatures."""
    attempts = [
        {"repo_id": repo_id, "repo_type": "dataset", "exist_ok": True, "token": hf_token},
        {"repo_id": repo_id, "repo_type": "dataset", "exist_ok": True, "use_auth_token": hf_token},
        {"repo_id": repo_id, "repo_type": "dataset", "exist_ok": True},
    ]

    # Legacy signature fallback: name + organization
    if "/" in repo_id:
        org, name = repo_id.split("/", 1)
        attempts.extend([
            {"name": name, "organization": org, "repo_type": "dataset", "exist_ok": True, "token": hf_token},
            {"name": name, "organization": org, "repo_type": "dataset", "exist_ok": True},
        ])

    last_err = None
    for kwargs in attempts:
        try:
            api.create_repo(**kwargs)
            return
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "409" in msg:
                return
            last_err = exc
            continue

    raise RuntimeError("Could not create/verify dataset repo '{}': {}".format(repo_id, last_err))


def _upload_file_with_progress_http(api, local_path, repo_id, path_in_repo, commit_message, hf_token):
    """Upload one file to HF Hub over HTTP using local file path."""
    attempts = [
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "path_in_repo": path_in_repo,
            "commit_message": commit_message,
            "token": hf_token,
        },
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "path_in_repo": path_in_repo,
            "commit_message": commit_message,
            "use_auth_token": hf_token,
        },
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "path_in_repo": path_in_repo,
            "commit_message": commit_message,
        },
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "path_in_repo": path_in_repo,
            "token": hf_token,
        },
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "path_in_repo": path_in_repo,
        },
        {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "token": hf_token,
        },
        {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
        },
    ]

    last_err = None
    for kwargs in attempts:
        call_kwargs = dict(kwargs)
        # Pass path string (not open buffer) to avoid Xet buffer fallback warnings.
        call_kwargs["path_or_fileobj"] = str(local_path)
        try:
            return api.upload_file(**call_kwargs)
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:
            msg = str(exc).lower()
            # Old SDK endpoint removed by server; advise upgrade.
            if "410" in msg or "gone" in msg or "/upload/main/" in msg:
                raise RuntimeError(
                    "Your huggingface_hub version is using a deprecated upload endpoint. "
                    "Upgrade with: pip install -U huggingface_hub"
                )
            raise

    raise RuntimeError("No compatible upload_file signature worked: {}".format(last_err))


def _print_timing_table(timing, rewrite_upload_seconds=None):
    """Print a compact timing table."""
    rows = [
        ("Step 1 - Scan + Signature", timing.get("step1_scan_and_signature_seconds", 0.0)),
        ("Step 2 - LSH Bucketing", timing.get("step2_lsh_bucketing_seconds", 0.0)),
        ("Step 3 - Jaccard Verify", timing.get("step3_jaccard_verification_seconds", 0.0)),
        ("Step 4 - Component Reduce", timing.get("step4_component_reduce_seconds", 0.0)),
    ]
    if rewrite_upload_seconds is not None:
        rows.append(("Step 5 - Rewrite + Upload", rewrite_upload_seconds))
    total = sum(sec for _, sec in rows)

    print("\n" + "-" * 72)
    print("TIME PROFILE")
    print("-" * 72)
    print("{:<34} {:>10} {:>10}".format("Step", "Seconds", "Share"))
    for name, sec in rows:
        share = 0.0 if total == 0 else (100.0 * sec / total)
        print("{:<34} {:>10.2f} {:>9.2f}%".format(name, sec, share))
    print("-" * 72)
    print("{:<34} {:>10.2f}".format("Total Profiled", total))

    # Granular step-1 breakdown (scan + signature internals).
    norm_s = timing.get("step1_normalization_seconds")
    shingle_s = timing.get("step1_shingling_seconds")
    shingle_to_int_s = timing.get("step1_shingle_to_int_seconds")
    sig_core_s = timing.get("step1_signature_core_seconds")
    sig_s = timing.get("step1_signature_seconds")
    cache_s = timing.get("step1_cache_write_seconds")
    other_s = timing.get("step1_other_overhead_seconds")
    docs_processed = timing.get("step1_docs_processed", 0)
    total_shingles = timing.get("step1_total_shingles")
    band_partitions = timing.get("step2_band_partitions")
    skipped_large_buckets = timing.get("step2_skipped_large_buckets", 0)
    skipped_large_bucket_docs = timing.get("step2_skipped_large_bucket_docs", 0)

    if norm_s is not None and shingle_s is not None and sig_s is not None:
        print("\n" + "-" * 72)
        print("STEP 1 BREAKDOWN (SCAN + SIGNATURE)")
        print("-" * 72)
        print("{:<34} {:>10}".format("Sub-step", "Seconds"))
        print("{:<34} {:>10.2f}".format("Normalization", float(norm_s)))
        print("{:<34} {:>10.2f}".format("Shingling", float(shingle_s)))
        if shingle_to_int_s is not None and sig_core_s is not None:
            print("{:<34} {:>10.2f}".format("Shingle -> int conversion", float(shingle_to_int_s)))
            print("{:<34} {:>10.2f}".format("Signature compute (core)", float(sig_core_s)))
            print("{:<34} {:>10.2f}".format("Signature compute (total)", float(sig_s)))
        else:
            print("{:<34} {:>10.2f}".format("Signature compute", float(sig_s)))
        if cache_s is not None:
            print("{:<34} {:>10.2f}".format("Cache write", float(cache_s)))
        if other_s is not None:
            print("{:<34} {:>10.2f}".format("Other overhead", float(other_s)))
        print("-" * 72)
        if docs_processed and docs_processed > 0:
            print("{:<34} {:>10}".format("Docs processed", "{:,}".format(docs_processed)))
            print(
                "{:<34} {:>10.4f}".format(
                    "Avg ms/doc (step1)",
                    1000.0 * float(timing.get("step1_scan_and_signature_seconds", 0.0)) / float(docs_processed),
                )
            )
            if total_shingles is not None:
                print("{:<34} {:>10}".format("Total shingles", "{:,}".format(int(total_shingles))))
                print(
                    "{:<34} {:>10.2f}".format(
                        "Avg shingles/doc",
                        float(total_shingles) / float(docs_processed),
                    )
                )
    if skipped_large_buckets:
        print("\n" + "-" * 72)
        print("LSH GUARDRAIL STATS")
        print("-" * 72)
        if band_partitions is not None:
            print("{:<34} {:>10}".format("Band partitions used", "{:,}".format(int(band_partitions))))
        print("{:<34} {:>10}".format("Skipped large buckets", "{:,}".format(int(skipped_large_buckets))))
        print("{:<34} {:>10}".format("Docs in skipped buckets", "{:,}".format(int(skipped_large_bucket_docs))))


def _rewrite_deduplicated_file(
    input_path,
    output_path,
    keep_rows,
    max_rows_per_file=None,
    show_progress=False,
    progress_desc=None,
):
    """Write deduplicated rows for one .jsonl.zst file.

    Returns:
        (valid_rows_seen, rows_kept, invalid_rows_skipped)
    """
    valid_row_idx = 0
    rows_kept = 0
    invalid_rows = 0

    with _open_text_file(input_path, "rt") as src, _open_text_file(output_path, "wt") as dst:
        line_bar = None
        try:
            line_iter = src
            if show_progress:
                line_bar = tqdm(
                    src,
                    desc=progress_desc or "Rewriting {}".format(Path(input_path).name),
                    unit="lines",
                    total=max_rows_per_file if max_rows_per_file is not None else None,
                    leave=False,
                )
                line_iter = line_bar

            for raw_line in line_iter:
                if max_rows_per_file is not None and valid_row_idx >= max_rows_per_file:
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    json.loads(stripped)
                except ValueError:
                    invalid_rows += 1
                    continue

                if valid_row_idx in keep_rows:
                    dst.write(stripped)
                    dst.write("\n")
                    rows_kept += 1

                valid_row_idx += 1
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Failed to decode {} as UTF-8 text. For .jsonl.zst files, "
                "ensure zstandard is installed and the source file is valid.".format(input_path)
            ) from exc
        finally:
            if line_bar is not None:
                line_bar.close()

    return valid_row_idx, rows_kept, invalid_rows


def dedup_and_upload(
    input_folder,
    hf_repo_id,
    hf_token,
    hf_path_prefix,
    text_column,
    num_hashes,
    num_bands,
    ngrams,
    jaccard_threshold,
    batch_size,
    cache_dir,
    workers=1,
    max_bucket_size=5000,
    band_partitions=128,
    max_files=None,
    max_rows=None,
    max_words=10000,
    count_only=False,
):
    """Run folder-level dedup and upload deduplicated files to HF one by one."""
    try:
        from lsh import minhash_lsh_deduplication_tabular_files
    except ImportError as exc:
        raise ImportError(
            "Could not import local 'lsh' package. Keep this script and the 'lsh/' folder in the same directory."
        ) from exc

    input_files = sorted(input_folder.glob("*.jsonl.zst"))
    if max_files is not None:
        input_files = input_files[:max_files]
    if not input_files:
        raise FileNotFoundError("No .jsonl.zst files found in: {}".format(input_folder))
    if not count_only and (not hf_repo_id or not hf_token):
        raise ValueError("hf_repo_id and hf_token are required unless --count-only is enabled")

    print("=" * 72)
    print("JSONL.ZST FOLDER DEDUP + HF UPLOAD")
    print("=" * 72)
    print("Input folder: {}".format(input_folder))
    print("Files found: {:,}".format(len(input_files)))
    print("Text column: {}".format(text_column))
    if count_only:
        print("Mode: count-only (no rewrite, no upload)")
    else:
        print("HF repo: {}".format(hf_repo_id))
        print("HF path prefix: {}".format(hf_path_prefix or "(root)"))
    print("Max files: {}".format(max_files if max_files is not None else "ALL"))
    print("Max rows per file: {}".format(max_rows if max_rows is not None else "ALL"))
    print("Max words per row for dedup: {}".format(max_words if max_words is not None else "ALL"))
    print("Workers (scan+signature): {}".format(workers))
    print("Max bucket size (LSH): {}".format(max_bucket_size if max_bucket_size is not None else "UNLIMITED"))
    print("Band partitions (LSH): {}".format(band_partitions))
    print("Dedup params:")
    print(
        "  num_hashes={}, num_bands={}, ngrams={}, threshold={}".format(
            num_hashes, num_bands, ngrams, jaccard_threshold
        )
    )
    print("=" * 72)

    print("Running deduplication engine...")
    overall_start = time.time()
    dedup_result = minhash_lsh_deduplication_tabular_files(
        input_files=[str(p) for p in input_files],
        text_column=text_column,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_rows_per_file=max_rows,
        max_words_per_row=max_words,
        workers=workers,
        max_bucket_size=max_bucket_size,
        band_partitions=band_partitions,
        verbose=True,
    )

    total_docs = dedup_result["total"]
    kept_docs = dedup_result["kept"]
    removed_docs = dedup_result["removed"]
    candidate_pairs = dedup_result["candidate_pairs_checked"]
    verified_pairs = dedup_result["verified_pairs"]
    keep_rows_per_file = dedup_result["keep_rows_per_file"]
    total_shingles = dedup_result.get("total_shingles", 0)

    print("\n" + "-" * 72)
    print("DEDUPLICATION STATISTICS")
    print("-" * 72)
    print("Total valid docs scanned: {:,}".format(total_docs))
    print("Docs kept after dedup:   {:,}".format(kept_docs))
    print("Docs removed:            {:,}".format(removed_docs))
    print("Total shingles generated: {:,}".format(total_shingles))
    print("Candidate pair checks:   {:,}".format(candidate_pairs))
    print("Verified dup pairs:      {:,}".format(verified_pairs))
    dedup_timing = dedup_result.get("timing", {})

    if count_only:
        _print_timing_table(dedup_timing, rewrite_upload_seconds=0.0)
        print("Count-only completed. Duplicate docs: {:,}".format(removed_docs))
        print("Wall time: {:.2f} seconds".format(time.time() - overall_start))
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        if "dataclasses" in str(exc):
            raise ImportError(
                "Python 3.6 detected and 'dataclasses' backport is missing. Run: pip install dataclasses"
            ) from exc
        raise ImportError(
            "huggingface_hub is required. Install with: pip install -U huggingface_hub"
        ) from exc

    print("Upload mode: huggingface_hub HTTP")
    api = HfApi()
    _ensure_dataset_repo_http(api, hf_repo_id, hf_token)

    uploaded_files = 0
    upload_failures = 0
    uploaded_bytes = 0
    total_valid_rows_seen = 0
    total_rows_written = 0
    total_invalid_rows = 0
    rewrite_upload_start = time.time()

    print("Step 4/4: Rewriting deduplicated files and uploading...")
    file_iter = tqdm(
        list(enumerate(input_files)),
        desc="Uploading deduped files",
        unit="file",
    )

    hf_path_prefix = hf_path_prefix.strip("/")

    for file_idx, input_path in file_iter:
        keep_rows = keep_rows_per_file.get(file_idx, set())

        with tempfile.NamedTemporaryFile(prefix="dedup_", suffix=".jsonl.zst", delete=False) as tmp_f:
            tmp_path = Path(tmp_f.name)

        try:
            valid_rows_seen, rows_kept, invalid_rows = _rewrite_deduplicated_file(
                input_path=input_path,
                output_path=tmp_path,
                keep_rows=keep_rows,
                max_rows_per_file=max_rows,
                show_progress=True,
                progress_desc="Rewrite {}".format(input_path.name),
            )

            total_valid_rows_seen += valid_rows_seen
            total_rows_written += rows_kept
            total_invalid_rows += invalid_rows

            remote_name = input_path.name
            if hf_path_prefix:
                remote_path = posixpath.join(hf_path_prefix, remote_name)
            else:
                remote_path = remote_name

            commit_message = "Upload deduplicated file: {}".format(remote_name)
            upload_url = _upload_file_with_progress_http(
                api=api,
                local_path=tmp_path,
                repo_id=hf_repo_id,
                path_in_repo=remote_path,
                commit_message=commit_message,
                hf_token=hf_token,
            )

            file_bytes = tmp_path.stat().st_size
            uploaded_files += 1
            uploaded_bytes += file_bytes

            print(
                "Uploaded {}: kept {:,}/{:,} valid rows (invalid skipped: {:,}) -> {}".format(
                    remote_name,
                    rows_kept,
                    valid_rows_seen,
                    invalid_rows,
                    upload_url,
                )
            )
        except Exception as exc:
            upload_failures += 1
            print("ERROR uploading {}: {}".format(input_path.name, exc))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    print("\n" + "-" * 72)
    print("UPLOAD SUMMARY")
    print("-" * 72)
    print("Files attempted:          {:,}".format(len(input_files)))
    print("Files uploaded:           {:,}".format(uploaded_files))
    print("Upload failures:          {:,}".format(upload_failures))
    print("Total uploaded size (GB): {:.3f}".format(uploaded_bytes / float(1024 ** 3)))
    print("Valid rows seen:          {:,}".format(total_valid_rows_seen))
    print("Rows written/uploaded:    {:,}".format(total_rows_written))
    print("Invalid JSONL rows:       {:,}".format(total_invalid_rows))
    print("=" * 72)
    _print_timing_table(dedup_timing, rewrite_upload_seconds=time.time() - rewrite_upload_start)
    print("Wall time: {:.2f} seconds".format(time.time() - overall_start))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate .jsonl.zst files in a folder and upload deduplicated outputs to HF.",
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=None,
        help="Folder containing .jsonl.zst files (positional).",
    )
    parser.add_argument(
        "--input_folder",
        dest="input_folder_opt",
        type=str,
        default=None,
        help="Folder containing .jsonl.zst files (optional flag form).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file containing HUGGINGFACE_KEY and optionally HUGGINGFACE_DATASET_REPO",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HF dataset repo id (username/repo). Overrides .env/env value.",
    )
    parser.add_argument(
        "--hf-path-prefix",
        type=str,
        default="",
        help="Optional folder prefix in repo, e.g. processed/train",
    )
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--num-hashes", type=int, default=256)
    parser.add_argument("--num-bands", type=int, default=32)
    parser.add_argument("--ngrams", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for Step 1 (scan+signature). Use >1 to parallelize CPU-heavy signature compute.",
    )
    parser.add_argument(
        "--max-bucket-size",
        type=int,
        default=5000,
        help="Skip LSH buckets larger than this size during candidate generation (memory/speed guard).",
    )
    parser.add_argument(
        "--band-partitions",
        type=int,
        default=128,
        help="Number of on-disk partitions per LSH band for low-memory bucketing on large runs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many .jsonl.zst files from the folder (sorted order).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many valid JSONL rows per file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=10000,
        help="Use only first N words per row for dedup computation (default: 10000).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache dir for dedup internals (default: temporary directory)",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only run dedup and print duplicate counts + timing profile. Skip rewrite and HF upload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_folder_arg = args.input_folder_opt or args.input_folder
    if not input_folder_arg:
        raise ValueError("Provide input folder via positional argument or --input_folder")

    input_folder = Path(input_folder_arg).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        raise NotADirectoryError("Invalid input folder: {}".format(input_folder))

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = SCRIPT_DIR / env_path

    env_values = {}
    env_values.update(_load_env_file(env_path))

    hf_token = None
    hf_repo_id = None
    if not args.count_only:
        hf_token = _resolve_hf_token(env_values)
        hf_repo_id = _resolve_repo_id(args.hf_repo_id, env_values)

    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be a positive integer")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")
    if args.max_words is not None and args.max_words <= 0:
        raise ValueError("--max-words must be a positive integer")
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if args.max_bucket_size is not None and args.max_bucket_size <= 1:
        raise ValueError("--max-bucket-size must be > 1 or omitted")
    if args.band_partitions <= 0:
        raise ValueError("--band-partitions must be a positive integer")

    dedup_and_upload(
        input_folder=input_folder,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        hf_path_prefix=args.hf_path_prefix,
        text_column=args.text_column,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        ngrams=args.ngrams,
        jaccard_threshold=args.jaccard_threshold,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        workers=args.workers,
        max_bucket_size=args.max_bucket_size,
        band_partitions=args.band_partitions,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_words=args.max_words,
        count_only=args.count_only,
    )


if __name__ == "__main__":
    main()
