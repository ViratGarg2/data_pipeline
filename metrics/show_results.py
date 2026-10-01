"""Recompute every number quoted in README.md from the evidence files in this repository.

    python metrics/show_results.py
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(path):
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


def section(title):
    print(f"\n=== {title}")


def production_run():
    section("Production run, shards 180-209 (metrics/production_run_180-209_summary.txt)")
    text = open(os.path.join(HERE, "production_run_180-209_summary.txt"), encoding="utf-8").read()
    for label, pattern in [
        ("rows processed", r"Total rows processed:\s+([\d,]+)"),
        ("wall seconds", r"Total time:\s+([\d.]+)"),
        ("rows/s", r"Processing rate:\s+([\d.]+)"),
        ("kept", r"Texts KEPT:\s+([\d,]+ \([\d.]+%\))"),
        ("removed: non-English", r"Non-English language:\s+([\d,]+)"),
        ("removed: Gopher", r"- Gopher quality:\s+([\d,]+)"),
        ("removed: quality classifier", r"Fasttext low quality:\s+([\d,]+)"),
        ("PII spans masked", r"Total PII masked:\s+([\d,]+)"),
    ]:
        m = re.search(pattern, text)
        print(f"  {label:28s} {m.group(1) if m else 'n/a'}")


def production_throughput():
    section("Production throughput, 8 workers per task (metrics/pipeline_shards_*.json)")
    chars = wall = rows = 0
    for path in sorted(glob.glob(os.path.join(HERE, "pipeline_shards_*.json"))):
        m = json.load(open(path))
        if m["totals"]["rows_in"] < 500_000:  # skip the single-shard task
            continue
        chars += m["totals"]["chars_in"]
        wall += m["time"]["filtering_wall_seconds"]
        rows += m["totals"]["rows_in"]
    print(f"  {chars / 1e9:.1f} GB of text, {rows / wall:.0f} rows/s, {wall / 60 / (chars / 1e9):.1f} min per GB of text, "
          f"{chars / 1e9 / (wall / 3600):.1f} GB/hour per task")
    m = json.load(open(os.path.join(HERE, "bench", "scale_w16.json")))
    print(f"  16 workers: {m['time']['filtering_wall_seconds'] / 60 / (m['totals']['chars_in'] / 1e9):.1f} min per GB of text")


def scaling():
    section("Worker scaling, 16 shards x 20k rows (metrics/bench/scale_w*.json)")
    base = None
    print(f"  {'workers':>7} {'wall s':>8} {'rows/s':>8} {'per worker':>10} {'speedup':>8} {'efficiency':>10}")
    for w in (1, 2, 4, 8, 16):
        m = load(f"bench/scale_w{w}.json")
        wall = m["time"]["filtering_wall_seconds"]
        base = base or wall
        sp = base / wall
        print(f"  {w:>7} {wall:>8.1f} {m['throughput']['rows_per_second_filtering']:>8.0f} "
              f"{m['throughput']['rows_per_second_per_worker']:>10.0f} {sp:>7.2f}x {100 * sp / w:>9.0f}%")


def ablation():
    section("Stage ablation, 8 workers, 8 shards x 20k rows (metrics/bench/stage_*.json)")
    prod = load("bench/stage_production.json")["time"]["wall_seconds"]
    for name in ("production", "no_pii", "no_langid", "no_gopher", "no_qclf", "plus_content",
                 "plus_dedup", "plus_dedup_v2"):
        m = load(f"bench/stage_{name}.json")
        wall = m["time"]["wall_seconds"]
        print(f"  {name:14s} wall {wall:8.1f} s  change {100 * (wall - prod) / prod:+8.1f}%  "
              f"kept {m['totals']['kept_pct']:5.1f}%  removed_by {m['removal_breakdown']['by_stage']}")


def latency():
    section("Per-document latency, production settings (metrics/bench/stage_production.json)")
    m = load("bench/stage_production.json")
    for stage, v in m["stages"].items():
        lat = v.get("latency")
        if not lat:
            continue
        share = v.get("pct_of_worker_time")
        share = f"{share:5.1f}%" if share is not None and stage != "document" else "  100%"
        print(f"  {stage:24s} docs_in {v['docs_in']:>7}  cpu {share}  "
              f"p50 {lat['p50_ms']:6.2f}  p95 {lat['p95_ms']:6.2f}  p99 {lat['p99_ms']:6.2f} ms")


def dedup():
    section("Dedup before/after (metrics/bench/stage_plus_dedup.json vs stage_plus_dedup_v2.json)")
    a, b = load("bench/stage_plus_dedup.json"), load("bench/stage_plus_dedup_v2.json")
    for label, fn in [
        ("dedup seconds", lambda m: f"{m['time']['dedup_seconds']:.1f}"),
        ("whole run seconds", lambda m: f"{m['time']['wall_seconds']:.1f}"),
        ("end-to-end rows/s", lambda m: f"{m['throughput']['rows_per_second_wall']:.0f}"),
        ("docs before -> after", lambda m: f"{m['deduplication']['total_docs_before']} -> {m['deduplication']['total_docs_after']}"),
        ("candidates / verified", lambda m: f"{m['deduplication']['candidate_pairs']} / {m['deduplication']['verified_pairs']}"),
        ("doc latency p50/p95/p99 ms", lambda m: "/".join(f"{m['stages']['document']['latency'][q]:.2f}" for q in ("p50_ms", "p95_ms", "p99_ms"))),
    ]:
        print(f"  {label:28s} before {fn(a):>22s}   after {fn(b):>22s}")
    print(f"  dedup speedup {a['time']['dedup_seconds'] / b['time']['dedup_seconds']:.1f}x, "
          f"whole run {a['time']['wall_seconds'] / b['time']['wall_seconds']:.1f}x")


def matched():
    section("Token-matched datasets (metrics/matched_data_summary.json)")
    s = load("matched_data_summary.json")
    for k in ("shards_complete", "filtered_tokens", "unfiltered_tokens", "pct_of_assignment_budget",
              "filtered_docs", "unfiltered_docs", "filter_configs"):
        print(f"  {k:28s} {s[k]}")
    print(f"  {'problems':28s} {len(s['problems'])}")


def baseline():
    section("Earlier model (inference_check/results/prev_model_summary_*.json)")
    for path in sorted(glob.glob(os.path.join(ROOT, "inference_check", "results", "prev_model_summary_*.json"))):
        s = json.load(open(path))
        print(f"  {os.path.basename(path)}: Paloma C4 loss {s['paloma_c4_val_loss']:.3f}, "
              f"perplexity {s['perplexity']:.1f}, {s['eval_tokens']:,} tokens on {s['device']}")


def training():
    section("Training: validation loss (metrics/train_25k_eval_curves.csv)")
    import csv
    rows = list(csv.DictReader(open(os.path.join(HERE, "train_25k_eval_curves.csv"))))
    for r in rows:
        print(f"  step {int(r['step']):>6}  filtered {r['filtered_val_loss']}  unfiltered {r['unfiltered_val_loss']}  gap {r['gap']}")
    gaps = [float(r["gap"]) for r in rows]
    print(f"  filtered ahead at {sum(g > 0 for g in gaps)}/{len(gaps)} evaluations; gap range {min(gaps):.3f}-{max(gaps):.3f}")
    final_unf = float(rows[-1]["unfiltered_val_loss"])
    pts = [(int(r["step"]), float(r["filtered_val_loss"])) for r in rows]
    for (s0, l0), (s1, l1) in zip(pts, pts[1:]):
        if l0 >= final_unf >= l1:
            step = s0 + (l0 - final_unf) / (l0 - l1) * (s1 - s0)
            print(f"  filtered reached the unfiltered final loss {final_unf:.3f} at ~{step:,.0f} steps -> "
                  f"{pts[-1][0] / step:.2f}x compute")


def comparison():
    for path in sorted(glob.glob(os.path.join(ROOT, "inference_check", "results", "compare_25k_summary_*.json"))):
        section(f"Final models ({os.path.relpath(path, ROOT)})")
        s = json.load(open(path))
        for arm, m in s["models"].items():
            for name in ("paloma_c4", "heldout_pile"):
                e = m[name]
                print(f"  {arm:10s} {name:12s} loss {e['loss']:.3f}  ppl {e['perplexity']:.1f}  "
                      f"next-token acc {100 * e['next_token_accuracy']:.2f}%")
            g = m["generation_gpt2_ppl"]
            print(f"  {arm:10s} generations: median GPT-2 ppl {g['median']:.2f} (n={g['n']})")
        print(f"  prompts where filtered was more natural to GPT-2: {s['prompts_where_filtered_more_fluent']}/50")


if __name__ == "__main__":
    production_run()
    production_throughput()
    scaling()
    ablation()
    latency()
    dedup()
    matched()
    training()
    comparison()
    baseline()
