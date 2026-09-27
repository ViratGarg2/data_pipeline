"""
HTML to Text Processing Pipeline with Language Identification and PII Masking

This script processes parquet files containing text data, detects HTML content,
extracts plain text from HTML using the extract_text_from_html_bytes function,
identifies the language using FastText, masks PII (emails, phones, IPs),
and outputs processed data to new parquet files.

Usage:
    python -m cs336_data.process_html_pipeline
    OR
    python process_html_pipeline.py --max-files 10
"""

import gc
import glob
import io
import os
import posixpath
import re
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure the project root is on sys.path so cs336_data imports work when this
# file is run as a script (e.g. from an IDE) rather than via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language, get_language_name
from cs336_data.pii_masking import mask_all_pii
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.quality_filter import gopher_quality_filter
from cs336_data.quality_classifier_fasttext import get_all_predictions
from cs336_data.lsh import minhash_lsh_deduplication_tabular_files
from cs336_data.jsonl_io import read_jsonl_compressed, write_jsonl_compressed, get_output_path
from cs336_data.pipeline_metrics import build_metrics, new_latency_store, write_metrics
from cs336_data import config


# ---------------- MEMORY MONITORING UTILITIES ----------------
def get_memory_usage() -> dict:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory stats in GB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return {
        "process_rss_gb": mem_info.rss / (1024 ** 3),
        "process_vms_gb": mem_info.vms / (1024 ** 3),
        "system_total_gb": virtual_mem.total / (1024 ** 3),
        "system_available_gb": virtual_mem.available / (1024 ** 3),
        "system_used_gb": virtual_mem.used / (1024 ** 3),
        "system_percent": virtual_mem.percent,
    }


def get_disk_usage(path: str = "/") -> dict:
    """Get disk usage statistics.
    
    Args:
        path: Path to check disk usage for
        
    Returns:
        Dictionary with disk stats in GB
    """
    disk = shutil.disk_usage(path)
    return {
        "total_gb": disk.total / (1024 ** 3),
        "used_gb": disk.used / (1024 ** 3),
        "free_gb": disk.free / (1024 ** 3),
        "percent_used": (disk.used / disk.total) * 100,
    }


def print_memory_status(prefix: str = ""):
    """Print current memory and disk status."""
    mem = get_memory_usage()
    print(f"{prefix}Memory: Process RSS={mem['process_rss_gb']:.2f}GB, "
          f"System={mem['system_used_gb']:.2f}/{mem['system_total_gb']:.2f}GB ({mem['system_percent']:.1f}%)")


def force_memory_cleanup():
    """Force garbage collection and release memory."""
    gc.collect()
    # Force Python to release memory back to the OS (if possible)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass  # Not available on macOS or if libc not found


def upload_file_to_hf_dataset(
    local_path: str,
    repo_id: str,
    token: str,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a file to a Hugging Face dataset repo (LFS handled by HF Hub)."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for uploads. Install with: pip install huggingface_hub"
        ) from exc

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found for upload: {local_path}")

    remote_path = path_in_repo or os.path.basename(local_path)
    file_size = os.path.getsize(local_path)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    progress_bar = tqdm(
        total=file_size,
        desc=f"  Uploading {os.path.basename(local_path)}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    )

    class _TqdmBufferedReader(io.BufferedReader):
        """Buffered reader that reports read progress to tqdm."""

        def read(self, size=-1):
            chunk = super().read(size)
            if chunk:
                progress_bar.update(len(chunk))
            return chunk

        def readinto(self, b):
            n = super().readinto(b)
            if n and n > 0:
                progress_bar.update(n)
            return n

    try:
        with open(local_path, "rb") as raw_f:
            with _TqdmBufferedReader(raw_f) as wrapped_f:
                upload_url = api.upload_file(
                    path_or_fileobj=wrapped_f,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or f"Add {remote_path}",
                )
    finally:
        progress_bar.close()
    return upload_url


# ---------------- CONFIG ----------------
INPUT_GLOB = config.INPUT_GLOB
INPUT_GLOB_JSON = getattr(config, "INPUT_GLOB_JSON", None)
OUTPUT_DIR = config.OUTPUT_DIR
TEXT_COLUMN = "text"
BATCH_SIZE = 1000
COMPRESSION = "zstd"
MAX_FILES = None  # Set to an integer to limit number of files processed, None for all
START_FILE_INDEX = 0  # Index of the first file to process (0-based)
# ----------------------------------------


def is_html_content(text: str) -> tuple[bool, str]:
    """
    Detect if the given text contains HTML content.
    
    Args:
        text: Text string to check for HTML content
        
    Returns:
        Tuple of (is_html: bool, html_type: str)
        html_type can be: 'full_html', 'partial_html', 'xml', 'plain_text'
    """
    if not text or not isinstance(text, str):
        return False, "empty"
    
    # Check for common HTML indicators
    text_lower = text[:2000].lower()  # Check first 2000 chars for performance
    
    # Full HTML document
    if "<!doctype html" in text_lower or "<html" in text_lower:
        return True, "full_html"
    
    # XML document
    if text_lower.strip().startswith("<?xml"):
        return True, "xml"
    
    # Partial HTML (contains HTML tags but not full document)
    html_tag_pattern = re.compile(r'<(div|span|p|a|img|table|tr|td|ul|ol|li|h[1-6]|br|hr|script|style|head|body|meta|link|form|input|button|nav|header|footer|section|article)\b', re.IGNORECASE)
    if html_tag_pattern.search(text[:5000]):
        return True, "partial_html"
    
    # Check for generic tags
    generic_tag_pattern = re.compile(r'<[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>.*?</[a-zA-Z][a-zA-Z0-9]*>', re.DOTALL)
    if generic_tag_pattern.search(text[:5000]):
        return True, "generic_tags"
    
    return False, "plain_text"


def process_single_file(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    batch_size: int = 1000,
    compression: str = "zstd",
    enable_langid: bool = True,
    enable_pii_masking: bool = True,
    enable_content_classification: bool = True,
    enable_quality_filter: bool = True,
    enable_quality_classifier: bool = True,
    max_rows: int | None = None,
    input_format: str = "parquet",
    output_format: str = "parquet",
) -> dict:
    """
    Process a single parquet file: detect HTML, extract text, identify language, mask PII, classify content, and apply quality filter.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        text_column: Column name containing text/HTML content
        batch_size: Number of rows to process at once
        compression: Output compression codec
        enable_langid: Whether to perform language identification
        enable_pii_masking: Whether to mask PII (emails, phones, IPs)
        enable_content_classification: Whether to classify NSFW and toxic content
        enable_quality_filter: Whether to apply Gopher quality filtering
        enable_quality_classifier: Whether to use NVIDIA Fasttext quality classifier
        max_rows: Maximum number of rows to process per file (None for all)
        
    Returns:
        Dictionary with processing statistics for this file
    """
    stats = {
        "total_rows": 0,
        "html_detected": 0,
        "html_extracted": 0,
        "extraction_failed": 0,
        "html_types": defaultdict(int),
        "languages": defaultdict(int),
        "language_scores": defaultdict(list),
        "pii_masked": {
            "emails": 0,
            "phones": 0,
            "ips": 0,
            "total": 0,
        },
        "content_classification": {
            "nsfw": 0,
            "non_nsfw": 0,
            "toxic": 0,
            "non_toxic": 0,
        },
        "filtered": {
            "non_english": 0,
            "nsfw": 0,
            "toxic": 0,
            "low_quality_Fasttext": 0,
            "total_filtered": 0,
            "kept": 0,
        },
        "quality_filter": {
            "passed": 0,
            "failed": 0,
            "failed_reasons": {
                "word_count": 0,
                "mean_word_length": 0,
                "hash_ratio": 0,
                "ellipsis_ratio": 0,
                "bullet_line_ratio": 0,
                "ellipsis_line_ratio": 0,
                "alphabetic_word_ratio": 0,
                "stop_word_count": 0,
                "empty_or_invalid": 0,
            },
        },
        "quality_classifier": {
            "positive": 0,
            "negative": 0,
        },
        # Per-step timing accumulators (seconds)
        "timings": {
            "html_detection": 0.0,
            "html_extraction": 0.0,
            "pii_masking": 0.0,
            "langid": 0.0,
            "quality_filter": 0.0,
            "quality_classifier": 0.0,
            "content_classification": 0.0,
            "io_read": 0.0,
            "io_write": 0.0,
            "total_file": 0.0,
        },
        # Raw measurements for pipeline_metrics (per-doc latencies in seconds)
        "input_path": input_path,
        "input_bytes": os.path.getsize(input_path),
        "output_bytes": 0,
        "chars_in": 0,
        "chars_out": 0,
        "stage_docs_in": defaultdict(int),
        "latencies": new_latency_store(),
    }

    def _record(stage: str, elapsed: float):
        stats["timings"][stage] += elapsed
        stats["stage_docs_in"][stage] += 1
        stats["latencies"][stage].append(elapsed)

    score_columns = {"language_score", "nsfw_score", "toxic_score", "quality_score"}
    label_columns = {"detected_language", "nsfw_label", "toxic_label", "quality_label"}

    def _empty_doc_data() -> dict:
        return {
            "text": None,
            "detected_language": None,
            "language_score": 0.0,
            "nsfw_label": None,
            "nsfw_score": 0.0,
            "toxic_label": None,
            "toxic_score": 0.0,
            "quality_label": None,
            "quality_score": 0.0,
        }

    def _process_text(text: str) -> dict | None:
        """Process a single text entry and return doc data if kept, else None."""
        doc_data = _empty_doc_data()

        # Handle None texts
        if text is None:
            stats["html_types"]["none"] += 1
            stats["filtered"]["total_filtered"] += 1
            return None

        # Step 1: HTML detection + extraction
        t0 = time.perf_counter()
        is_html, html_type = is_html_content(text)
        stats["html_types"][html_type] += 1
        _record("html_detection", time.perf_counter() - t0)

        if is_html:
            stats["html_detected"] += 1

            t0 = time.perf_counter()
            if isinstance(text, str):
                html_bytes = text.encode("utf-8")
            else:
                html_bytes = text

            extracted = extract_text_from_html_bytes(html_bytes)

            if extracted is not None:
                current_text = extracted
                stats["html_extracted"] += 1
            else:
                current_text = text
                stats["extraction_failed"] += 1
            _record("html_extraction", time.perf_counter() - t0)
        else:
            current_text = text

        # Step 2: PII masking
        if enable_pii_masking and current_text:
            t0 = time.perf_counter()
            current_text, pii_stats = mask_all_pii(current_text)
            _record("pii_masking", time.perf_counter() - t0)
            stats["pii_masked"]["emails"] += pii_stats["emails"]
            stats["pii_masked"]["phones"] += pii_stats["phones"]
            stats["pii_masked"]["ips"] += pii_stats["ips"]
            stats["pii_masked"]["total"] += pii_stats["total"]

        doc_data["text"] = current_text

        # Step 3: Language filter
        if enable_langid and current_text:
            t0 = time.perf_counter()
            lang_code, lang_score = identify_language(current_text)
            _record("langid", time.perf_counter() - t0)
            doc_data["detected_language"] = lang_code
            doc_data["language_score"] = lang_score
            stats["languages"][lang_code] += 1

            if lang_code != "en" or lang_score < 0.5:
                stats["filtered"]["non_english"] += 1
                stats["filtered"]["total_filtered"] += 1
                return None

        # Step 4: Gopher quality filter
        if enable_quality_filter and current_text:
            t0 = time.perf_counter()
            quality_passed, quality_details = gopher_quality_filter(current_text)
            _record("quality_filter", time.perf_counter() - t0)

            if quality_passed:
                stats["quality_filter"]["passed"] += 1
            else:
                stats["quality_filter"]["failed"] += 1
                for check in quality_details.get("failed_checks", []):
                    if check in stats["quality_filter"]["failed_reasons"]:
                        stats["quality_filter"]["failed_reasons"][check] += 1
                stats["filtered"]["total_filtered"] += 1
                return None

        # Step 5: Fasttext quality classifier
        if enable_quality_classifier and current_text:
            t0 = time.perf_counter()
            predictions = get_all_predictions(current_text)
            _record("quality_classifier", time.perf_counter() - t0)

            doc_data["quality_label"] = predictions["label"]
            doc_data["quality_score"] = predictions["confidence"]
            stats["quality_classifier"][predictions["label"]] += 1

            if doc_data["quality_label"] != "positive":
                stats["filtered"]["low_quality_Fasttext"] += 1
                stats["filtered"]["total_filtered"] += 1
                return None

        # Step 6: Content classification (NSFW + toxic)
        if enable_content_classification and current_text:
            t0 = time.perf_counter()
            nsfw_label, nsfw_score = classify_nsfw(current_text)
            toxic_label, toxic_score = classify_toxic_speech(current_text)
            _record("content_classification", time.perf_counter() - t0)

            doc_data["nsfw_label"] = nsfw_label
            doc_data["nsfw_score"] = nsfw_score
            doc_data["toxic_label"] = toxic_label
            doc_data["toxic_score"] = toxic_score

            # Update stats
            if nsfw_label == "nsfw":
                stats["content_classification"]["nsfw"] += 1
            else:
                stats["content_classification"]["non_nsfw"] += 1
            if toxic_label == "toxic":
                stats["content_classification"]["toxic"] += 1
            else:
                stats["content_classification"]["non_toxic"] += 1

            # Short-circuit: if NSFW or toxic, filter out
            if nsfw_label == "nsfw":
                stats["filtered"]["nsfw"] += 1
                stats["filtered"]["total_filtered"] += 1
                return None

            if toxic_label == "toxic":
                stats["filtered"]["toxic"] += 1
                stats["filtered"]["total_filtered"] += 1
                return None

        stats["filtered"]["kept"] += 1
        return doc_data

    def _process_text_timed(text: str) -> dict | None:
        """_process_text plus end-to-end latency and chars in/out for metrics."""
        t_doc = time.perf_counter()
        doc_data = _process_text(text)
        stats["latencies"]["document"].append(time.perf_counter() - t_doc)
        if text is not None:
            stats["chars_in"] += len(text)
        if doc_data is not None and doc_data["text"]:
            stats["chars_out"] += len(doc_data["text"])
        return doc_data

    def _build_output_columns(source_columns: list[str]) -> list[str]:
        output_columns = list(source_columns)
        if enable_langid:
            output_columns.extend(["detected_language", "language_score"])
        if enable_quality_classifier:
            output_columns.extend(["quality_label", "quality_score"])
        if enable_content_classification:
            output_columns.extend(["nsfw_label", "nsfw_score", "toxic_label", "toxic_score"])
        return output_columns

    def _empty_output_data(columns: list[str]) -> dict:
        return {col: [] for col in columns}

    def _append_output_row(
        output_data: dict,
        source_columns: list[str],
        row_values: dict,
        row_idx: int,
        doc_data: dict,
    ):
        for col in source_columns:
            if col == text_column:
                output_data[col].append(doc_data["text"])
            else:
                output_data[col].append(row_values[col][row_idx])
        if enable_langid:
            output_data["detected_language"].append(doc_data["detected_language"])
            output_data["language_score"].append(doc_data["language_score"])
        if enable_quality_classifier:
            output_data["quality_label"].append(doc_data["quality_label"])
            output_data["quality_score"].append(doc_data["quality_score"])
        if enable_content_classification:
            output_data["nsfw_label"].append(doc_data["nsfw_label"])
            output_data["nsfw_score"].append(doc_data["nsfw_score"])
            output_data["toxic_label"].append(doc_data["toxic_label"])
            output_data["toxic_score"].append(doc_data["toxic_score"])

    def _to_arrow_arrays(output_data: dict, output_columns: list[str], source_types: dict | None = None) -> list:
        arrays = []
        for key in output_columns:
            values = output_data[key]
            if key in score_columns:
                arrays.append(pa.array(values, type=pa.float64()))
            elif key in label_columns:
                arrays.append(pa.array(values, type=pa.string()))
            elif source_types and key in source_types:
                arrays.append(pa.array(values, type=source_types[key]))
            else:
                if hasattr(values, "tolist"):
                    values = values.tolist()
                arrays.append(pa.array(values))
        return arrays

    # Check memory before starting
    print_memory_status("  [BEFORE] ")

    file_start_time = time.perf_counter()

    if input_format == "parquet":
        parquet_file = pq.ParquetFile(input_path)
        source_columns = parquet_file.schema.names

        if text_column not in source_columns:
            print(f"  Warning: Column '{text_column}' not found in {input_path}")
            return stats

        output_columns = _build_output_columns(source_columns)
        source_types = {field.name: field.type for field in parquet_file.schema_arrow}

        rows_available = parquet_file.metadata.num_rows if parquet_file.metadata else None
        rows_to_process = rows_available
        if max_rows is not None:
            if rows_available is None:
                rows_to_process = max_rows
            else:
                rows_to_process = min(max_rows, rows_available)
                if max_rows < rows_available:
                    print(f"  Limiting to {max_rows} rows (out of {rows_available})")

        if rows_to_process is not None:
            stats["total_rows"] = rows_to_process

        parquet_writer = None
        output_data = _empty_output_data(output_columns)
        rows_processed = 0
        flush_every = max(batch_size, 1)

        def _flush_output(force: bool = False):
            nonlocal parquet_writer
            if output_format != "parquet":
                return
            buffered_rows = len(output_data[text_column]) if text_column in output_data else 0
            if buffered_rows == 0:
                return
            if not force and buffered_rows < flush_every:
                return

            arrays = _to_arrow_arrays(output_data, output_columns, source_types=source_types)
            output_table = pa.table(dict(zip(output_columns, arrays)))
            if output_format == "parquet":
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, output_table.schema, compression=compression)
                parquet_writer.write_table(output_table)
            del output_table, arrays

            for col in output_columns:
                output_data[col].clear()

        progress = tqdm(
            total=rows_to_process,
            desc="  Processing rows",
            unit="rows",
            leave=False,
        )

        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=source_columns)):
            if rows_to_process is not None and rows_processed >= rows_to_process:
                break

            t_io = time.perf_counter()
            batch_dict = {col: batch.column(i).to_pylist() for i, col in enumerate(source_columns)}
            stats["timings"]["io_read"] += time.perf_counter() - t_io
            batch_rows = len(batch_dict[text_column])

            if rows_to_process is not None:
                remaining = rows_to_process - rows_processed
                if batch_rows > remaining:
                    for col in source_columns:
                        batch_dict[col] = batch_dict[col][:remaining]
                    batch_rows = remaining

            for row_idx in range(batch_rows):
                text = batch_dict[text_column][row_idx]
                doc_data = _process_text_timed(text)
                if doc_data is None:
                    continue

                _append_output_row(output_data, source_columns, batch_dict, row_idx, doc_data)

            rows_processed += batch_rows
            progress.update(batch_rows)
            if output_format == "parquet":
                t_io = time.perf_counter()
                _flush_output(force=False)
                stats["timings"]["io_write"] += time.perf_counter() - t_io

            del batch, batch_dict
            if (batch_idx + 1) % 10 == 0:
                gc.collect()

        progress.close()

        if rows_to_process is None:
            stats["total_rows"] = rows_processed

        if output_format == "parquet":
            t_io = time.perf_counter()
            _flush_output(force=True)
            if parquet_writer is not None:
                parquet_writer.close()
            else:
                # Preserve prior behavior: still write an empty parquet output file.
                empty_arrays = _to_arrow_arrays(output_data, output_columns, source_types=source_types)
                output_table = pa.table(dict(zip(output_columns, empty_arrays)))
                pq.write_table(output_table, output_path, compression=compression)
                del output_table, empty_arrays
            stats["timings"]["io_write"] += time.perf_counter() - t_io
        else:
            actual_output_path = write_jsonl_compressed(output_path, output_data, compression=compression)
            if actual_output_path:
                output_path = actual_output_path

        del output_data
    else:
        # Keep JSONL input path behavior as-is (already row-limited during read).
        df_dict = read_jsonl_compressed(input_path, text_column, max_rows=max_rows)
        if text_column not in df_dict:
            print(f"  Warning: Column '{text_column}' not found in {input_path}")
            return stats

        source_columns = list(df_dict.keys())
        output_columns = _build_output_columns(source_columns)
        output_data = _empty_output_data(output_columns)
        texts = df_dict[text_column]
        stats["total_rows"] = len(texts)

        for idx, text in enumerate(tqdm(texts, desc="  Processing rows", unit="rows", leave=False)):
            doc_data = _process_text_timed(text)
            if doc_data is None:
                continue

            _append_output_row(output_data, source_columns, df_dict, idx, doc_data)

        arrays = _to_arrow_arrays(output_data, output_columns)
        if output_format == "parquet":
            output_table = pa.table(dict(zip(output_columns, arrays)))
            pq.write_table(output_table, output_path, compression=compression)
            del output_table
        else:
            actual_output_path = write_jsonl_compressed(output_path, output_data, compression=compression)
            if actual_output_path:
                output_path = actual_output_path

        del arrays, output_data, df_dict, texts

    print(f"  Filtered: {stats['filtered']['total_filtered']} rows removed, {stats['filtered']['kept']} rows kept")
    print(f"    - Non-English: {stats['filtered']['non_english']}")
    print(f"    - Gopher quality failed: {stats['quality_filter']['failed']}")
    print(f"    - Fasttext low quality: {stats['filtered']['low_quality_Fasttext']}")
    print(f"    - NSFW: {stats['filtered']['nsfw']}")
    print(f"    - Toxic: {stats['filtered']['toxic']}")

    # Record total file time
    stats["timings"]["total_file"] = time.perf_counter() - file_start_time
    if output_path and os.path.exists(output_path):
        stats["output_bytes"] = os.path.getsize(output_path)

    # Print per-step timing breakdown
    t = stats["timings"]
    total_t = t["total_file"] or 1e-9
    print(f"\n  ⏱  TIMING PROFILE (total {total_t:.2f}s):")
    print(f"    {'Step':<28} {'Time (s)':>10} {'% of total':>12}")
    print(f"    {'-'*52}")
    for step_name, step_time in [
        ("I/O Read (parquet decode)", t["io_read"]),
        ("HTML Detection",           t["html_detection"]),
        ("HTML Extraction",          t["html_extraction"]),
        ("PII Masking",              t["pii_masking"]),
        ("Language ID (FastText)",   t["langid"]),
        ("Gopher Quality Filter",   t["quality_filter"]),
        ("Fasttext Quality Clf",    t["quality_classifier"]),
        ("Content Classification",  t["content_classification"]),
        ("I/O Write (parquet enc)",  t["io_write"]),
    ]:
        pct = (step_time / total_t) * 100 if total_t > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {step_name:<28} {step_time:>10.2f} {pct:>10.1f}%  {bar}")
    accounted = sum([t["io_read"], t["html_detection"], t["html_extraction"],
                     t["pii_masking"], t["langid"], t["quality_filter"],
                     t["quality_classifier"], t["content_classification"], t["io_write"]])
    overhead = total_t - accounted
    pct_oh = (overhead / total_t) * 100 if total_t > 0 else 0
    print(f"    {'Overhead / other':<28} {overhead:>10.2f} {pct_oh:>10.1f}%")
    print()

    gc.collect()
    print_memory_status("  [AFTER] ")

    return stats


def _merge_stats(total_stats: dict, file_stats: dict, enable_flags: dict) -> None:
    """Merge per-file stats into total_stats in-place."""
    total_stats["files_processed"] += 1
    total_stats["total_rows"] += file_stats["total_rows"]
    total_stats["html_detected"] += file_stats["html_detected"]
    total_stats["html_extracted"] += file_stats["html_extracted"]
    total_stats["extraction_failed"] += file_stats["extraction_failed"]

    for html_type, count in file_stats["html_types"].items():
        total_stats["html_types"][html_type] += count

    if enable_flags.get("langid"):
        for lang, count in file_stats["languages"].items():
            total_stats["languages"][lang] += count

    if enable_flags.get("pii"):
        for key in ("emails", "phones", "ips", "total"):
            total_stats["pii_masked"][key] += file_stats["pii_masked"][key]

    if enable_flags.get("content_classification"):
        for key in ("nsfw", "non_nsfw", "toxic", "non_toxic"):
            total_stats["content_classification"][key] += file_stats["content_classification"][key]

    for key in ("non_english", "nsfw", "toxic", "total_filtered", "kept"):
        total_stats["filtered"][key] += file_stats["filtered"].get(key, 0)
    total_stats["filtered"]["low_quality_Fasttext"] += file_stats["filtered"].get("low_quality_Fasttext", 0)

    if enable_flags.get("quality_filter"):
        total_stats["quality_filter"]["passed"] += file_stats["quality_filter"]["passed"]
        total_stats["quality_filter"]["failed"] += file_stats["quality_filter"]["failed"]
        for reason, count in file_stats["quality_filter"]["failed_reasons"].items():
            total_stats["quality_filter"]["failed_reasons"][reason] += count

    if enable_flags.get("quality_classifier"):
        for ql in ("positive", "negative"):
            total_stats["quality_classifier"][ql] += file_stats["quality_classifier"].get(ql, 0)

    # Aggregate per-step timings (sum of CPU-seconds across workers)
    if "timings" in file_stats:
        if "timings" not in total_stats:
            total_stats["timings"] = {}
        for step, secs in file_stats["timings"].items():
            total_stats["timings"][step] = total_stats["timings"].get(step, 0.0) + secs

    # Raw measurements for pipeline_metrics
    for key in ("input_bytes", "output_bytes", "chars_in", "chars_out"):
        total_stats[key] += file_stats[key]
    for stage, count in file_stats["stage_docs_in"].items():
        total_stats["stage_docs_in"][stage] += count
    for stage, samples in file_stats["latencies"].items():
        total_stats["latencies"][stage].extend(samples)
    file_seconds = file_stats["timings"]["total_file"]
    total_stats["per_file"].append({
        "file": os.path.basename(file_stats["input_path"]),
        "rows": file_stats["total_rows"],
        "kept": file_stats["filtered"]["kept"],
        "seconds": file_seconds,
        "rows_per_second": file_stats["total_rows"] / file_seconds if file_seconds > 0 else 0.0,
    })


def _process_file_worker(kwargs: dict) -> dict:
    """
    Worker function for multiprocessing.  Receives a dict of keyword arguments,
    calls process_single_file, and returns a result dict with the stats and
    output path.

    Each worker process will lazily import / load ML models on first use
    (FastText, toxicity classifiers, etc.) — this is fine because the
    underlying libraries cache models after the first load inside each process.
    """
    input_path = kwargs["input_path"]
    output_path = kwargs["output_path"]
    worker_id = kwargs.pop("worker_id", "?")

    try:
        file_stats = process_single_file(**{
            k: v for k, v in kwargs.items() if k != "worker_id"
        })
        return {
            "success": True,
            "input_path": input_path,
            "output_path": output_path,
            "stats": file_stats,
            "worker_id": worker_id,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "input_path": input_path,
            "output_path": output_path,
            "stats": None,
            "worker_id": worker_id,
            "error": traceback.format_exc(),
        }


def run_pipeline(
    input_glob: str = INPUT_GLOB,
    output_dir: str = OUTPUT_DIR,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    compression: str = COMPRESSION,
    max_files: int | None = MAX_FILES,
    start_index: int = START_FILE_INDEX,
    enable_langid: bool = True,
    enable_pii_masking: bool = True,
    enable_content_classification: bool = True,
    enable_quality_filter: bool = True,
    enable_quality_classifier: bool = True,
    enable_deduplication: bool = True,
    max_rows: int | None = 1000,  # DEBUG: Limit rows per file (set to None for all rows)
    input_format: str = "parquet",  # "parquet" or "jsonl"
    output_format: str = "parquet",  # "parquet" or "jsonl"
    push_to_hf: bool = False,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    delete_local_after_hf_upload: bool = True,
    num_workers: int = 1,
    metrics_json: str | None = None,
):
    """
    Run the full HTML extraction, language identification, PII masking, content classification, and quality filtering pipeline.
    
    Args:
        input_glob: Glob pattern for input files
        output_dir: Directory to write processed files
        text_column: Column name containing text/HTML content
        batch_size: Batch size for processing
        compression: Output compression codec
        max_files: Maximum number of files to process (None for all)
        start_index: Index of the first file to process (0-based)
        enable_langid: Whether to perform language identification
        enable_pii_masking: Whether to mask PII (emails, phones, IPs)
        enable_content_classification: Whether to classify NSFW and toxic content
        enable_quality_filter: Whether to apply Gopher quality filtering
        max_rows: Maximum number of rows to process per file (None for all, default: 1000 for debugging)
        input_format: Input file format ("parquet" or "jsonl")
        output_format: Output file format ("parquet" or "jsonl")
        push_to_hf: Upload each processed file to HF dataset repo
        hf_repo_id: HF dataset repo in format "username/repo"
        hf_path_prefix: Optional subdirectory within the dataset repo
        delete_local_after_hf_upload: Delete local file after successful upload
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all input files
    all_input_files = sorted(glob.glob(input_glob))
    
    if not all_input_files:
        print(f"No files found matching pattern: {input_glob}")
        return
    
    total_available = len(all_input_files)
    
    # Apply start_index and max_files constraints
    if start_index >= total_available:
        print(f"Start index {start_index} is beyond available files ({total_available})")
        return
    
    # Slice the files based on start_index and max_files
    if max_files is not None:
        end_index = min(start_index + max_files, total_available)
        input_files = all_input_files[start_index:end_index]
    else:
        input_files = all_input_files[start_index:]

    # Read dedup parameters from central config (override CLI args)
    dedup_num_hashes = getattr(config, "DEDUP_NUM_HASHES", 100)
    dedup_num_bands = getattr(config, "DEDUP_NUM_BANDS", 10)
    dedup_ngrams = getattr(config, "DEDUP_NGRAMS", 5)
    dedup_jaccard_threshold = getattr(config, "DEDUP_JACCARD_THRESHOLD", 0.8)
    # Allow global enable flag in config to disable dedup regardless of CLI
    if not getattr(config, "DEDUPLICATION_ENABLED", True):
        enable_deduplication = False

    hf_repo_id = hf_repo_id or getattr(config, "HUGGINGFACE_DATASET_REPO", "")
    hf_token = getattr(config, "HUGGINGFACE_KEY", "")
    hf_path_prefix = hf_path_prefix or getattr(config, "HUGGINGFACE_DATASET_PATH_PREFIX", "")
    hf_path_prefix = hf_path_prefix.strip("/")

    if push_to_hf and not hf_repo_id:
        raise ValueError("HF upload enabled but no repo configured. Set --hf-repo-id or config.HUGGINGFACE_DATASET_REPO")
    if push_to_hf and not hf_token:
        raise ValueError("HF upload enabled but no token: export HF_TOKEN=... (see cs336_data/config.py)")
    if push_to_hf and enable_deduplication and delete_local_after_hf_upload:
        raise ValueError(
            "delete_local_after_hf_upload cannot be used with deduplication. "
            "Disable dedup or keep local files until dedup completes."
        )
    
    print("=" * 60)
    print("HTML to Text Extraction Pipeline with Language ID & PII Masking")
    print("=" * 60)
    print(f"Input pattern: {input_glob}")
    print(f"Input format: {input_format.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {output_format.upper()}")
    print(f"Total files available: {total_available}")
    print(f"Start index: {start_index}")
    print(f"Max files to process: {max_files if max_files else 'ALL'}")
    print(f"Max rows per file: {max_rows if max_rows else 'ALL'}")
    print(f"Files to process this run: {len(input_files)}")
    print(f"File range: [{start_index} - {start_index + len(input_files) - 1}]")
    print(f"Language identification: {'ENABLED' if enable_langid else 'DISABLED'}")
    print(f"PII masking: {'ENABLED' if enable_pii_masking else 'DISABLED'}")
    print(f"Content classification: {'ENABLED' if enable_content_classification else 'DISABLED'}")
    print(f"Quality filter (Gopher): {'ENABLED' if enable_quality_filter else 'DISABLED'}")
    print(f"Quality classifier (Fasttext): {'ENABLED' if enable_quality_classifier else 'DISABLED'}")
    print(f"Deduplication (MinHash+LSH): {'ENABLED' if enable_deduplication else 'DISABLED'}")
    print(f"Parallel workers: {num_workers}")
    print(f"Hugging Face upload: {'ENABLED' if push_to_hf else 'DISABLED'}")
    if push_to_hf:
        print(f"  HF dataset repo: {hf_repo_id}")
        print(f"  HF repo path prefix: {hf_path_prefix or '(root)'}")
        print(f"  Delete local file after upload: {'YES' if delete_local_after_hf_upload else 'NO'}")
    if enable_deduplication:
        print(f"  Hashes={dedup_num_hashes}, Bands={dedup_num_bands}, "
              f"N-grams={dedup_ngrams}, Threshold={dedup_jaccard_threshold}")
    print("=" * 60)
    
    # Aggregate statistics
    total_stats = {
        "files_processed": 0,
        "total_rows": 0,
        "html_detected": 0,
        "html_extracted": 0,
        "extraction_failed": 0,
        "html_types": defaultdict(int),
        "languages": defaultdict(int),
        "pii_masked": {
            "emails": 0,
            "phones": 0,
            "ips": 0,
            "total": 0,
        },
        "content_classification": {
            "nsfw": 0,
            "non_nsfw": 0,
            "toxic": 0,
            "non_toxic": 0,
        },
        "filtered": {
            "non_english": 0,
            "nsfw": 0,
            "toxic": 0,
            "low_quality_Fasttext": 0,
            "total_filtered": 0,
            "kept": 0,
        },
        "quality_filter": {
            "passed": 0,
            "failed": 0,
            "failed_reasons": {
                "word_count": 0,
                "mean_word_length": 0,
                "hash_ratio": 0,
                "ellipsis_ratio": 0,
                "bullet_line_ratio": 0,
                "ellipsis_line_ratio": 0,
                "alphabetic_word_ratio": 0,
                "stop_word_count": 0,
                "empty_or_invalid": 0,
            },
        },
        "quality_classifier": {
            "positive": 0,
            "negative": 0,
        },
        "huggingface": {
            "uploaded_files": 0,
            "upload_failed": 0,
            "uploaded_bytes": 0,
            "deleted_local_files": 0,
        },
        "deduplication": {
            "total_docs_before": 0,
            "total_docs_after": 0,
            "duplicates_removed": 0,
            "candidate_pairs": 0,
            "verified_pairs": 0,
            "files_rewritten": 0,
        },
        "failed_files": [],
        "input_bytes": 0,
        "output_bytes": 0,
        "chars_in": 0,
        "chars_out": 0,
        "stage_docs_in": defaultdict(int),
        "latencies": new_latency_store(),
        "per_file": [],
    }

    start_time = time.time()
    dedup_elapsed = 0.0
    
    # Track which output files were actually written in THIS run
    output_files_this_run: list[str] = []

    # Build the list of per-file job kwargs
    file_jobs: list[dict] = []
    for i, input_path in enumerate(input_files):
        output_path = get_output_path(input_path, output_dir, input_format, output_format)
        file_jobs.append({
            "input_path": input_path,
            "output_path": output_path,
            "text_column": text_column,
            "batch_size": batch_size,
            "compression": compression,
            "enable_langid": enable_langid,
            "enable_pii_masking": enable_pii_masking,
            "enable_content_classification": enable_content_classification,
            "enable_quality_filter": enable_quality_filter,
            "enable_quality_classifier": enable_quality_classifier,
            "max_rows": max_rows,
            "input_format": input_format,
            "output_format": output_format,
            "worker_id": i,
        })

    enable_flags = {
        "langid": enable_langid,
        "pii": enable_pii_masking,
        "content_classification": enable_content_classification,
        "quality_filter": enable_quality_filter,
        "quality_classifier": enable_quality_classifier,
    }

    def _print_file_result(result: dict, file_idx: int, total_files: int):
        """Print per-file stats from a completed worker result."""
        filename = os.path.basename(result["input_path"])
        if not result["success"]:
            total_stats["failed_files"].append(result["input_path"])
            print(f"\n[{file_idx+1}/{total_files}] ERROR processing {filename}:")
            print(result["error"])
            return

        fs = result["stats"]
        print(f"\n[{file_idx+1}/{total_files}] Done: {filename}")
        print(f"  Filtered: {fs['filtered']['total_filtered']} rows removed, {fs['filtered']['kept']} rows kept")
        print(f"  Rows: {fs['total_rows']:,}  |  HTML: {fs['html_detected']:,}  |  Extracted: {fs['html_extracted']:,}")

        if enable_pii_masking and fs["pii_masked"]["total"] > 0:
            pii = fs["pii_masked"]
            print(f"  PII masked: {pii['total']} (emails:{pii['emails']}, phones:{pii['phones']}, IPs:{pii['ips']})")

        if enable_content_classification:
            cc = fs["content_classification"]
            print(f"  Content: NSFW:{cc['nsfw']}, Non-NSFW:{cc['non_nsfw']}, Toxic:{cc['toxic']}, Non-Toxic:{cc['non_toxic']}")

        if enable_quality_classifier:
            qc = fs["quality_classifier"]
            print(f"  Quality (Fasttext): Positive:{qc['positive']}, Negative:{qc['negative']}")

        if enable_langid and fs["languages"]:
            top_langs = sorted(fs["languages"].items(), key=lambda x: x[1], reverse=True)[:3]
            lang_str = ", ".join([f"{get_language_name(l)}:{c}" for l, c in top_langs])
            print(f"  Top languages: {lang_str}")

    # ==================================================================
    # PROCESS FILES (parallel if num_workers > 1, else sequential)
    # ==================================================================
    effective_workers = min(num_workers, len(file_jobs))

    if effective_workers > 1:
        print(f"\n🚀 Processing {len(file_jobs)} files with {effective_workers} parallel workers...\n")

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx = {
                executor.submit(_process_file_worker, job): idx
                for idx, job in enumerate(file_jobs)
            }

            progress = tqdm(total=len(file_jobs), desc="Processing files", unit="file")

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                progress.update(1)

                _print_file_result(result, idx, len(file_jobs))

                if result["success"]:
                    output_files_this_run.append(result["output_path"])
                    _merge_stats(total_stats, result["stats"], enable_flags)

                    # HF upload (sequential — after each file completes)
                    if push_to_hf:
                        opath = result["output_path"]
                        remote_file_name = os.path.basename(opath)
                        remote_path = (
                            posixpath.join(hf_path_prefix, remote_file_name)
                            if hf_path_prefix else remote_file_name
                        )
                        try:
                            file_size = os.path.getsize(opath) if os.path.exists(opath) else 0
                            upload_url = upload_file_to_hf_dataset(
                                local_path=opath, repo_id=hf_repo_id,
                                token=hf_token, path_in_repo=remote_path,
                                commit_message=f"Upload processed file: {remote_file_name}",
                            )
                            total_stats["huggingface"]["uploaded_files"] += 1
                            total_stats["huggingface"]["uploaded_bytes"] += file_size
                            print(f"  Uploaded to HF: {upload_url}")
                            if delete_local_after_hf_upload and os.path.exists(opath):
                                os.remove(opath)
                                total_stats["huggingface"]["deleted_local_files"] += 1
                        except Exception:
                            total_stats["huggingface"]["upload_failed"] += 1
                            import traceback
                            traceback.print_exc()

            progress.close()
    else:
        # Sequential fallback (num_workers=1 or single file)
        print(f"\n📄 Processing {len(file_jobs)} files sequentially...\n")

        for idx, job in enumerate(tqdm(file_jobs, desc="Processing files")):
            result = _process_file_worker(job)
            _print_file_result(result, idx, len(file_jobs))

            if result["success"]:
                output_files_this_run.append(result["output_path"])
                _merge_stats(total_stats, result["stats"], enable_flags)

                if push_to_hf:
                    opath = result["output_path"]
                    remote_file_name = os.path.basename(opath)
                    remote_path = (
                        posixpath.join(hf_path_prefix, remote_file_name)
                        if hf_path_prefix else remote_file_name
                    )
                    try:
                        file_size = os.path.getsize(opath) if os.path.exists(opath) else 0
                        upload_url = upload_file_to_hf_dataset(
                            local_path=opath, repo_id=hf_repo_id,
                            token=hf_token, path_in_repo=remote_path,
                            commit_message=f"Upload processed file: {remote_file_name}",
                        )
                        total_stats["huggingface"]["uploaded_files"] += 1
                        total_stats["huggingface"]["uploaded_bytes"] += file_size
                        print(f"  Uploaded to HF: {upload_url}")
                        if delete_local_after_hf_upload and os.path.exists(opath):
                            os.remove(opath)
                            total_stats["huggingface"]["deleted_local_files"] += 1
                    except Exception:
                        total_stats["huggingface"]["upload_failed"] += 1
                        import traceback
                        traceback.print_exc()

            force_memory_cleanup()
    
    files_done_time = time.time()
    print_memory_status("\n  [AFTER ALL FILES] ")
    
    # ==================================================================
    # DEDUPLICATION STEP (runs only on files processed in THIS run)
    # ==================================================================
    if enable_deduplication:
        print("\n")
        print("=" * 60)
        print("RUNNING DEDUPLICATION (MinHash + LSH)")
        print("=" * 60)
        print(f"  Hashes: {dedup_num_hashes}  |  Bands: {dedup_num_bands}  |  "
              f"N-grams: {dedup_ngrams}  |  Threshold: {dedup_jaccard_threshold}")
        print_memory_status("  [BEFORE DEDUP] ")
        
        dedup_start = time.time()
        
        # Use only files processed in THIS run (not all files in output_dir)
        output_files = sorted(output_files_this_run)
        
        # Helper to count rows from output file (parquet or jsonl)
        def get_output_file_row_count(file_path: str) -> int:
            """Count rows in an output file without loading full text content."""
            if file_path.endswith('.parquet'):
                pf = pq.ParquetFile(file_path)
                return pf.metadata.num_rows if pf.metadata else 0
            else:
                # JSONL format fallback (loads only for counting)
                data = read_jsonl_compressed(file_path, text_column)
                return len(data.get(text_column, []))
        
        # --- DEBUG: Check actual row counts before deduplication -----------
        print(f"\n🔍 DEBUG: Checking row counts in {len(output_files)} files from THIS run:")
        total_debug_rows = 0
        for i, file_path in enumerate(output_files):
            num_rows = get_output_file_row_count(file_path)
            total_debug_rows += num_rows
            print(f"  [{i+1}] {os.path.basename(file_path)}: {num_rows:,} rows")
        print(f"  TOTAL DEBUG ROWS: {total_debug_rows:,}")
        expected_files = len(output_files)
        print(f"  Expected ({expected_files} files × up to {max_rows or 'ALL'} rows kept after filters)")
        print()
        
        # --- 1. Deduplicate directly from output file paths ----------
        if not output_files:
            print("  No output files from this run — skipping dedup.")
        else:
            dedup_result = minhash_lsh_deduplication_tabular_files(
                input_files=output_files,
                text_column=text_column,
                num_hashes=dedup_num_hashes,
                num_bands=dedup_num_bands,
                ngrams=dedup_ngrams,
                jaccard_threshold=dedup_jaccard_threshold,
                batch_size=batch_size,
                workers=num_workers,  # signature step is embarrassingly parallel
                verbose=True,
            )

            total_stats["deduplication"]["total_docs_before"] = dedup_result["total"]
            total_stats["deduplication"]["candidate_pairs"] = dedup_result["candidate_pairs_checked"]
            total_stats["deduplication"]["verified_pairs"] = dedup_result["verified_pairs"]
            total_stats["deduplication"]["total_docs_after"] = dedup_result["kept"]
            total_stats["deduplication"]["duplicates_removed"] = dedup_result["removed"]

            print(f"  Total documents across {len(output_files)} files: {dedup_result['total']:,}")
            print(f"  Candidate duplicate pairs: {dedup_result['candidate_pairs_checked']:,}")
            print(f"  Verified duplicate pairs (Jaccard ≥ {dedup_jaccard_threshold}): {dedup_result['verified_pairs']:,}")
            print(f"  Documents kept: {dedup_result['kept']:,}  |  Removed: {dedup_result['removed']:,}")

            # --- 2. Rewrite output files without duplicates -----------
            duplicates_removed = dedup_result["removed"]
            if duplicates_removed > 0:
                keep_rows_per_file = dedup_result["keep_rows_per_file"]

                files_rewritten = 0
                for file_idx, file_path in enumerate(output_files):
                    keep_rows = keep_rows_per_file.get(file_idx)

                    if file_path.endswith('.parquet'):
                        tbl = pq.read_table(file_path)
                        original_len = tbl.num_rows

                        if keep_rows is None:
                            keep_mask = [False] * original_len
                        else:
                            keep_mask = [r in keep_rows for r in range(original_len)]

                        kept_count = sum(keep_mask)

                        if kept_count < original_len:
                            mask_array = pa.array(keep_mask, type=pa.bool_())
                            filtered_tbl = tbl.filter(mask_array)
                            pq.write_table(filtered_tbl, file_path, compression=compression)
                            files_rewritten += 1
                            print(f"    Rewrote {os.path.basename(file_path)}: "
                                  f"{original_len} → {kept_count} rows "
                                  f"(-{original_len - kept_count})")
                            del filtered_tbl, mask_array

                        del tbl
                    else:
                        data = read_jsonl_compressed(file_path, text_column)
                        original_len = len(data.get(text_column, []))
                        keep_indices = keep_rows if keep_rows is not None else set()

                        filtered_data = {}
                        for key, values in data.items():
                            filtered_data[key] = [v for i, v in enumerate(values) if i in keep_indices]

                        kept_count = len(filtered_data.get(text_column, []))
                        if kept_count < original_len:
                            write_jsonl_compressed(file_path, filtered_data, compression=compression)
                            files_rewritten += 1
                            print(f"    Rewrote {os.path.basename(file_path)}: "
                                  f"{original_len} → {kept_count} rows "
                                  f"(-{original_len - kept_count})")

                        del data, filtered_data

                    gc.collect()

                total_stats["deduplication"]["files_rewritten"] = files_rewritten
                print(f"  Files rewritten: {files_rewritten}")
            else:
                print("  No duplicates found — output files unchanged.")

            del dedup_result
            gc.collect()
            
            dedup_elapsed = time.time() - dedup_start
            print(f"  Deduplication time: {dedup_elapsed:.2f}s")
            print_memory_status("  [AFTER DEDUP] ")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print final summary
    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETE - FINAL STATISTICS")
    print("=" * 60)
    print(f"Total files processed:    {total_stats['files_processed']:,}")
    print(f"Total rows processed:     {total_stats['total_rows']:,}")
    print(f"Total HTML detected:      {total_stats['html_detected']:,}")
    print(f"Total text extracted:     {total_stats['html_extracted']:,}")
    print(f"Total extraction failed:  {total_stats['extraction_failed']:,}")
    print(f"Total time:               {elapsed:.2f} seconds")
    if elapsed > 0:
        print(f"Processing rate:          {total_stats['total_rows'] / elapsed:.2f} rows/sec")
    print()

    # Print aggregate timing profile
    if "timings" in total_stats and total_stats["timings"]:
        t = total_stats["timings"]
        cpu_total = t.get("total_file", 0.0)
        print("-" * 60)
        print("⏱  AGGREGATE TIMING PROFILE (sum of CPU-seconds across all workers)")
        print("-" * 60)
        print(f"  Wall-clock time:       {elapsed:>10.2f}s")
        print(f"  Total CPU-seconds:     {cpu_total:>10.2f}s  (across {total_stats['files_processed']} files)")
        if num_workers > 1 and elapsed > 0:
            speedup = cpu_total / elapsed
            print(f"  Effective speedup:     {speedup:>10.2f}x  (ideal={num_workers}x)")
        print()
        print(f"  {'Step':<28} {'CPU-sec':>10} {'% of CPU':>10} {'Avg/file':>10}")
        print(f"  {'-'*60}")
        nfiles = max(total_stats['files_processed'], 1)
        for step_name, key in [
            ("I/O Read (parquet decode)", "io_read"),
            ("HTML Detection",           "html_detection"),
            ("HTML Extraction",          "html_extraction"),
            ("PII Masking",              "pii_masking"),
            ("Language ID (FastText)",   "langid"),
            ("Gopher Quality Filter",   "quality_filter"),
            ("Fasttext Quality Clf",    "quality_classifier"),
            ("Content Classification",  "content_classification"),
            ("I/O Write (parquet enc)",  "io_write"),
        ]:
            secs = t.get(key, 0.0)
            pct = (secs / cpu_total) * 100 if cpu_total > 0 else 0
            avg = secs / nfiles
            bar = "█" * int(pct / 2)
            print(f"  {step_name:<28} {secs:>10.2f} {pct:>9.1f}% {avg:>10.2f}  {bar}")
        accounted = sum(t.get(k, 0.0) for k in [
            "io_read", "html_detection", "html_extraction", "pii_masking",
            "langid", "quality_filter", "quality_classifier",
            "content_classification", "io_write"])
        overhead = cpu_total - accounted
        pct_oh = (overhead / cpu_total) * 100 if cpu_total > 0 else 0
        print(f"  {'Overhead / other':<28} {overhead:>10.2f} {pct_oh:>9.1f}%")
        print()
    
    # Print PII masking summary
    if enable_pii_masking:
        print("-" * 60)
        print("PII MASKING SUMMARY")
        print("-" * 60)
        pii = total_stats["pii_masked"]
        print(f"Email addresses masked:   {pii['emails']:,}")
        print(f"Phone numbers masked:     {pii['phones']:,}")
        print(f"IP addresses masked:      {pii['ips']:,}")
        print(f"Total PII masked:         {pii['total']:,}")
        print()
    
    # Print HTML type breakdown
    print("-" * 60)
    print("HTML TYPE BREAKDOWN")
    print("-" * 60)
    print(f"{'Type':<20} {'Count':>15} {'Percentage':>15}")
    print("-" * 60)
    
    total_content = sum(total_stats["html_types"].values())
    for html_type, count in sorted(
        total_stats["html_types"].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        percentage = (count / total_content * 100) if total_content > 0 else 0
        print(f"{html_type:<20} {count:>15,} {percentage:>14.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_content:>15,} {'100.00':>14}%")
    print()
    
    # Print language breakdown
    if enable_langid and total_stats["languages"]:
        print("-" * 60)
        print("LANGUAGE BREAKDOWN")
        print("-" * 60)
        print(f"{'Language':<25} {'Code':<8} {'Count':>12} {'Percentage':>12}")
        print("-" * 60)
        
        total_lang = sum(total_stats["languages"].values())
        for lang_code, count in sorted(
            total_stats["languages"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:30]:  # Show top 30 languages
            lang_name = get_language_name(lang_code)
            percentage = (count / total_lang * 100) if total_lang > 0 else 0
            print(f"{lang_name:<25} {lang_code:<8} {count:>12,} {percentage:>11.2f}%")
        
        if len(total_stats["languages"]) > 30:
            others = sum(v for k, v in sorted(
                total_stats["languages"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[30:])
            print(f"{'... others':<25} {'':<8} {others:>12,} {others/total_lang*100:>11.2f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<25} {'':<8} {total_lang:>12,} {'100.00':>11}%")
        print(f"Unique languages detected: {len(total_stats['languages'])}")
    
    # Print content classification summary
    if enable_content_classification:
        print("-" * 60)
        print("CONTENT CLASSIFICATION SUMMARY")
        print("-" * 60)
        cc = total_stats["content_classification"]
        total_classified = cc["nsfw"] + cc["non_nsfw"]
        print(f"NSFW content:             {cc['nsfw']:,} ({cc['nsfw']/total_classified*100:.2f}%)" if total_classified > 0 else f"NSFW content:             {cc['nsfw']:,}")
        print(f"Non-NSFW content:         {cc['non_nsfw']:,} ({cc['non_nsfw']/total_classified*100:.2f}%)" if total_classified > 0 else f"Non-NSFW content:         {cc['non_nsfw']:,}")
        print(f"Toxic content:            {cc['toxic']:,} ({cc['toxic']/total_classified*100:.2f}%)" if total_classified > 0 else f"Toxic content:            {cc['toxic']:,}")
        print(f"Non-Toxic content:        {cc['non_toxic']:,} ({cc['non_toxic']/total_classified*100:.2f}%)" if total_classified > 0 else f"Non-Toxic content:        {cc['non_toxic']:,}")
        print()
    
    # Print quality classifier summary (Fasttext)
    if enable_quality_classifier:
        print("-" * 60)
        print("QUALITY CLASSIFIER SUMMARY (Fasttext)")
        print("-" * 60)
        qc = total_stats["quality_classifier"]
        total_qc = qc["positive"] + qc["negative"]
        print(f"Positive (high) quality:  {qc['positive']:,} ({qc['positive']/total_qc*100:.2f}%)" if total_qc > 0 else f"Positive (high) quality:  {qc['positive']:,}")
        print(f"Negative (low) quality:   {qc['negative']:,} ({qc['negative']/total_qc*100:.2f}%)" if total_qc > 0 else f"Negative (low) quality:   {qc['negative']:,}")
        print(f"Kept (positive only):     {qc['positive']:,}")
        print()
    
    # Print filtering summary
    print("-" * 60)
    print("FILTERING SUMMARY (English only, Non-NSFW, Non-Toxic, Quality)")
    print("-" * 60)
    flt = total_stats["filtered"]
    total_processed = flt["kept"] + flt["total_filtered"]
    print(f"Total texts processed:    {total_processed:,}")
    print(f"Texts KEPT:               {flt['kept']:,} ({flt['kept']/total_processed*100:.2f}%)" if total_processed > 0 else f"Texts KEPT:               {flt['kept']:,}")
    print(f"Texts FILTERED OUT:       {flt['total_filtered']:,} ({flt['total_filtered']/total_processed*100:.2f}%)" if total_processed > 0 else f"Texts FILTERED OUT:       {flt['total_filtered']:,}")
    print()
    print("Filtered out due to:")
    print(f"  - Non-English language: {flt['non_english']:,}")
    print(f"  - Gopher quality:       {total_stats['quality_filter']['failed']:,}")
    print(f"  - Fasttext low quality:  {flt.get('low_quality_Fasttext', 0):,}")
    print(f"  - NSFW content:         {flt['nsfw']:,}")
    print(f"  - Toxic content:        {flt['toxic']:,}")
    print()
    
    # Print quality filter summary
    if enable_quality_filter:
        print("-" * 60)
        print("GOPHER QUALITY FILTER SUMMARY")
        print("-" * 60)
        qf = total_stats["quality_filter"]
        total_qf = qf["passed"] + qf["failed"]
        print(f"Texts passed quality:     {qf['passed']:,} ({qf['passed']/total_qf*100:.2f}%)" if total_qf > 0 else f"Texts passed quality:     {qf['passed']:,}")
        print(f"Texts failed quality:     {qf['failed']:,} ({qf['failed']/total_qf*100:.2f}%)" if total_qf > 0 else f"Texts failed quality:     {qf['failed']:,}")
        print()
        print("Failed quality checks breakdown:")
        for reason, count in sorted(qf["failed_reasons"].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  - {reason}: {count:,}")
        print()
    
    # Print deduplication summary
    if enable_deduplication:
        print("-" * 60)
        print("DEDUPLICATION SUMMARY (MinHash + LSH)")
        print("-" * 60)
        dd = total_stats["deduplication"]
        print(f"Documents before dedup:   {dd['total_docs_before']:,}")
        print(f"Documents after dedup:    {dd['total_docs_after']:,}")
        print(f"Duplicates removed:       {dd['duplicates_removed']:,}")
        if dd['total_docs_before'] > 0:
            pct = dd['duplicates_removed'] / dd['total_docs_before'] * 100
            print(f"Dedup removal rate:       {pct:.2f}%")
        print(f"Candidate pairs found:    {dd['candidate_pairs']:,}")
        print(f"Verified duplicate pairs: {dd['verified_pairs']:,}")
        print(f"Output files rewritten:   {dd['files_rewritten']:,}")
        print()

    # Print Hugging Face upload summary
    if push_to_hf:
        print("-" * 60)
        print("HUGGING FACE UPLOAD SUMMARY")
        print("-" * 60)
        hf_stats = total_stats["huggingface"]
        uploaded_gb = hf_stats["uploaded_bytes"] / (1024 ** 3)
        print(f"Files uploaded:           {hf_stats['uploaded_files']:,}")
        print(f"Upload failures:          {hf_stats['upload_failed']:,}")
        print(f"Total uploaded size:      {uploaded_gb:.2f} GB")
        print(f"Local files deleted:      {hf_stats['deleted_local_files']:,}")
        print()

    if total_stats["failed_files"]:
        print("-" * 60)
        print(f"FAILED FILES: {len(total_stats['failed_files'])} of {len(file_jobs)}")
        print("-" * 60)
        for path in total_stats["failed_files"]:
            print(f"  {path}")
        print()

    print("=" * 60)

    if metrics_json:
        run_config = {
            "input_glob": input_glob,
            "start_index": start_index,
            "max_files": max_files,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "enabled": {**enable_flags, "deduplication": enable_deduplication},
            "dedup_params": {
                "num_hashes": dedup_num_hashes,
                "num_bands": dedup_num_bands,
                "ngrams": dedup_ngrams,
                "jaccard_threshold": dedup_jaccard_threshold,
            },
            "push_to_hf": push_to_hf,
        }
        write_metrics(metrics_json, build_metrics(
            total_stats, run_config, start_time, files_done_time, end_time, dedup_elapsed,
        ))

    return total_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HTML to Text Extraction Pipeline with Language Identification"
    )
    parser.add_argument(
        "--input-glob", "-i",
        type=str,
        default=INPUT_GLOB,
        help="Glob pattern for input parquet files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to write processed files"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--start-index", "-s",
        type=int,
        default=5,
        help="Index of the first file to process (0-based, default: 0)"
    )
    parser.add_argument(
        "--text-column", "-c",
        type=str,
        default=TEXT_COLUMN,
        help="Column name containing text/HTML content"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=COMPRESSION,
        help="Output compression codec (e.g., zstd, snappy, gzip)"
    )
    parser.add_argument(
        "--no-langid",
        action="store_true",
        help="Disable language identification (faster processing)"
    )
    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Disable PII masking (emails, phones, IPs)"
    )
    parser.add_argument(
        "--no-content-classification",
        action="store_true",
        help="Disable NSFW and toxic content classification"
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable Gopher quality filtering"
    )
    parser.add_argument(
        "--no-quality-classifier",
        action="store_true",
        help="Disable NVIDIA Fasttext quality classifier"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable MinHash+LSH fuzzy deduplication"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel worker processes for file-level parallelism (default: 1 = sequential)"
    )
    # Dedup parameters are read from cs336_data.config.py (DEDUP_*)
    parser.add_argument(
        "--max-rows", "-r",
        type=int,
        default=-1,
        help="Maximum rows to process per file (default: 1000 for debugging, use -1 for all)"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Input file format: 'parquet' or 'jsonl' (supports .jsonl, .jsonl.gz, .jsonl.zst)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Output file format: 'parquet' or 'jsonl' (compressed with zstd)"
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Upload each processed file to a Hugging Face dataset repository"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HF dataset repo id (username/repo). Defaults to config.HUGGINGFACE_DATASET_REPO"
    )
    parser.add_argument(
        "--hf-path-prefix",
        type=str,
        default="",
        help="Optional folder prefix inside HF dataset repo (e.g. 'processed/train')"
    )
    parser.add_argument(
        "--keep-local-after-hf-upload",
        action="store_true",
        help="Keep local output files after successful HF upload"
    )
    
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Write run metrics (throughput, per-stage latency/time, docs removed per filter) to this JSON file"
    )
    
    args = parser.parse_args()
    
    # Handle max_rows: -1 means all rows
    max_rows = None if args.max_rows == -1 else args.max_rows
    
    # If using jsonl input, switch to JSON glob if available and not explicitly set
    input_glob = args.input_glob
    if args.input_format == "jsonl" and args.input_glob == INPUT_GLOB and INPUT_GLOB_JSON:
        input_glob = INPUT_GLOB_JSON
        print(f"Using JSONL input glob from config: {input_glob}")
    
    stats = run_pipeline(
        input_glob=input_glob,
        output_dir=args.output_dir,
        text_column=args.text_column,
        batch_size=args.batch_size,
        compression=args.compression,
        max_files=args.max_files,
        start_index=args.start_index,
        enable_langid=not args.no_langid,
        enable_pii_masking=not args.no_pii,
        enable_content_classification=not args.no_content_classification,
        enable_quality_filter=not args.no_quality_filter,
        enable_quality_classifier=not args.no_quality_classifier,
        enable_deduplication=not args.no_dedup,
        max_rows=max_rows,
        input_format=args.input_format,
        output_format=args.output_format,
        push_to_hf=args.push_to_hf,
        hf_repo_id=args.hf_repo_id,
        hf_path_prefix=args.hf_path_prefix,
        delete_local_after_hf_upload=not args.keep_local_after_hf_upload,
        num_workers=args.workers,
        metrics_json=args.metrics_json,
    )

    # Exit non-zero so SLURM marks the job FAILED instead of reporting a clean run.
    if stats["failed_files"] or stats["huggingface"]["upload_failed"]:
        sys.exit(1)
