"""
Machine-readable metrics for process_html_pipeline runs.

The pipeline collects raw measurements (per-document stage latencies, docs
entering each stage, bytes/chars in and out); this module turns them into a
JSON report used for pipeline ablations: throughput, latency percentiles,
time per stage, and how many documents each filter removed.
"""

import json
import os
import resource
import socket
import time
from array import array

import numpy as np

# Stages timed once per document (I/O is timed per batch, so it is excluded).
PER_DOC_STAGES = [
    "html_detection",
    "html_extraction",
    "pii_masking",
    "langid",
    "quality_filter",
    "quality_classifier",
    "content_classification",
    "document",  # end-to-end latency of one document through all stages
]


def new_latency_store() -> dict:
    return {stage: array("f") for stage in PER_DOC_STAGES}


def latency_summary(samples) -> dict | None:
    """Latency percentiles in milliseconds, or None when the stage never ran."""
    if len(samples) == 0:
        return None
    ms = np.frombuffer(samples, dtype=np.float32).astype(np.float64) * 1000.0
    return {
        "count": int(ms.size),
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def _peak_rss_gb() -> dict:
    # ru_maxrss is in KB on Linux.
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {
        "main_process_gb": self_kb / 1024 ** 2,
        "largest_worker_gb": children_kb / 1024 ** 2,
    }


def build_metrics(
    total_stats: dict,
    run_config: dict,
    start_time: float,
    files_done_time: float,
    end_time: float,
    dedup_seconds: float,
) -> dict:
    """Assemble the metrics report from the merged pipeline stats."""
    rows = total_stats["total_rows"]
    kept = total_stats["filtered"]["kept"]
    timings = total_stats.get("timings", {})
    worker_seconds = timings.get("total_file", 0.0)
    wall = end_time - start_time
    filter_wall = files_done_time - start_time
    dedup = total_stats["deduplication"]
    removed = total_stats["filtered"]
    gopher = total_stats["quality_filter"]
    docs_in = total_stats.get("stage_docs_in", {})

    # Documents removed by each stage, in pipeline order.
    removed_by_stage = {
        "empty_text": total_stats["html_types"].get("none", 0),
        "langid": removed["non_english"],
        "quality_filter": gopher["failed"],
        "quality_classifier": removed["low_quality_Fasttext"],
        "content_classification": removed["nsfw"] + removed["toxic"],
        "deduplication": dedup["duplicates_removed"],
    }
    total_removed = sum(removed_by_stage.values())

    stages = {}
    for stage in PER_DOC_STAGES:
        entry = {
            "cpu_seconds": timings.get(stage, worker_seconds if stage == "document" else 0.0),
            "docs_in": docs_in.get(stage, 0),
            "latency": latency_summary(total_stats["latencies"][stage]),
        }
        if stage in removed_by_stage:
            n = removed_by_stage[stage]
            entry["docs_removed"] = n
            entry["removed_pct_of_stage_input"] = 100.0 * n / entry["docs_in"] if entry["docs_in"] else 0.0
        if worker_seconds and stage != "document":
            entry["pct_of_worker_time"] = 100.0 * entry["cpu_seconds"] / worker_seconds
        stages[stage] = entry
    for io_stage in ("io_read", "io_write"):
        stages[io_stage] = {"cpu_seconds": timings.get(io_stage, 0.0)}

    return {
        "run": {
            **run_config,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_time)),
        },
        "totals": {
            "files_processed": total_stats["files_processed"],
            "files_failed": len(total_stats["failed_files"]),
            "rows_in": rows,
            "docs_kept_after_filters": kept,
            "docs_kept_final": kept - dedup["duplicates_removed"],
            "docs_removed": total_removed,
            "kept_pct": 100.0 * (kept - dedup["duplicates_removed"]) / rows if rows else 0.0,
            "input_bytes": total_stats.get("input_bytes", 0),
            "output_bytes": total_stats.get("output_bytes", 0),
            "chars_in": total_stats.get("chars_in", 0),
            "chars_out": total_stats.get("chars_out", 0),
        },
        "time": {
            "wall_seconds": wall,
            "filtering_wall_seconds": filter_wall,
            "dedup_seconds": dedup_seconds,
            "worker_seconds": worker_seconds,
            "parallel_speedup": worker_seconds / filter_wall if filter_wall > 0 else 0.0,
        },
        "throughput": {
            "rows_per_second_wall": rows / wall if wall > 0 else 0.0,
            "rows_per_second_filtering": rows / filter_wall if filter_wall > 0 else 0.0,
            "rows_per_second_per_worker": rows / worker_seconds if worker_seconds > 0 else 0.0,
            "mb_per_second_filtering": total_stats.get("input_bytes", 0) / 1e6 / filter_wall if filter_wall > 0 else 0.0,
        },
        "stages": stages,
        "removal_breakdown": {
            "by_stage": removed_by_stage,
            "pct_of_removed": {
                k: (100.0 * v / total_removed if total_removed else 0.0) for k, v in removed_by_stage.items()
            },
            "gopher_failed_checks": dict(gopher["failed_reasons"]),
            "nsfw": removed["nsfw"],
            "toxic": removed["toxic"],
        },
        "pii_masked": dict(total_stats["pii_masked"]),
        "html_types": dict(total_stats["html_types"]),
        "deduplication": {**dedup, "seconds": dedup_seconds},
        "memory": _peak_rss_gb(),
        "per_file": total_stats.get("per_file", []),
        "failed_files": total_stats["failed_files"],
        "huggingface": dict(total_stats["huggingface"]),
    }


def write_metrics(path: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {path}")
