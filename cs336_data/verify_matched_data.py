"""
Check that the filtered and unfiltered HF datasets built by prepare_matched_data.py
are complete and token-matched, and print their totals.

For every shard it checks that both repos have text, tokens and a manifest, that
the two manifests agree on the token count, and that each token file is exactly
2 bytes per token.

Usage:
  python cs336_data/verify_matched_data.py --shards 0-269 \\
      --filtered-repo ViratGarg/pile_filtered_13B --unfiltered-repo ViratGarg/pile_unfiltered_13B \\
      --summary-json metrics/matched_data_summary.json
"""

import argparse
import json
import os
import tempfile
from collections import Counter

from huggingface_hub import HfApi, snapshot_download

ASSIGNMENT_TOKENS = 100_000 * 128 * 2 * 512  # train_steps * batch * GPUs * context


def parse_shards(spec: str) -> list[int]:
    shards = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        shards.extend(range(int(lo), int(hi or lo) + 1))
    return shards


def repo_state(api: HfApi, repo: str, workdir: str) -> tuple[dict, dict]:
    """Returns ({path: size} for all files, {shard: manifest})."""
    sizes = {
        f.path: f.size
        for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
        if hasattr(f, "size")
    }
    local = snapshot_download(repo, repo_type="dataset", allow_patterns=["manifests/*.json"],
                              local_dir=os.path.join(workdir, repo.replace("/", "__")))
    manifests = {}
    for name in os.listdir(os.path.join(local, "manifests")):
        with open(os.path.join(local, "manifests", name)) as f:
            m = json.load(f)
        manifests[m["shard"]] = m
    return sizes, manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", default="0-269")
    parser.add_argument("--filtered-repo", default="ViratGarg/pile_filtered_13B")
    parser.add_argument("--unfiltered-repo", default="ViratGarg/pile_unfiltered_13B")
    parser.add_argument("--summary-json", help="Write the totals to this file")
    args = parser.parse_args()

    api = HfApi()
    shards = parse_shards(args.shards)
    with tempfile.TemporaryDirectory() as tmp:
        f_sizes, f_man = repo_state(api, args.filtered_repo, tmp)
        u_sizes, u_man = repo_state(api, args.unfiltered_repo, tmp)

    problems = []
    for shard in shards:
        name = f"train-{shard:05d}"
        for label, sizes, man in (("filtered", f_sizes, f_man), ("unfiltered", u_sizes, u_man)):
            for path in (f"text/{name}.parquet", f"tokens/{name}.bin", f"manifests/{name}.json"):
                if path not in sizes:
                    problems.append(f"{label}: missing {path}")
            if shard in man and f"tokens/{name}.bin" in sizes:
                expected = man[shard]["filtered" if label == "filtered" else "unfiltered"]["tokens"] * 2
                if sizes[f"tokens/{name}.bin"] != expected:
                    problems.append(f"{label}: tokens/{name}.bin is {sizes[f'tokens/{name}.bin']} bytes, expected {expected}")
        if shard in f_man and shard in u_man:
            ft, ut = f_man[shard]["filtered"]["tokens"], u_man[shard]["unfiltered"]["tokens"]
            if ft != ut or f_man[shard]["filtered"]["tokens"] != u_man[shard]["filtered"]["tokens"]:
                problems.append(f"{name}: filtered {ft:,} tokens vs unfiltered {ut:,}")

    complete = [s for s in shards if s in f_man and s in u_man]
    filtered_tokens = sum(f_man[s]["filtered"]["tokens"] for s in complete)
    unfiltered_tokens = sum(u_man[s]["unfiltered"]["tokens"] for s in complete)
    summary = {
        "shards_expected": len(shards),
        "shards_complete": len(complete),
        "filtered_tokens": filtered_tokens,
        "unfiltered_tokens": unfiltered_tokens,
        "assignment_tokens": ASSIGNMENT_TOKENS,
        "pct_of_assignment_budget": 100.0 * filtered_tokens / ASSIGNMENT_TOKENS,
        "filtered_docs": sum(f_man[s]["filtered"]["docs"] for s in complete),
        "unfiltered_docs": sum(u_man[s]["unfiltered"]["docs"] for s in complete),
        "unfiltered_mean_pct_of_raw_docs_sampled": (
            sum(u_man[s]["unfiltered"]["sampled_pct_of_raw_docs"] for s in complete) / len(complete) if complete else 0.0
        ),
        # Which pipeline stages produced each filtered shard (see prepare_matched_data.py).
        "filter_configs": dict(Counter(
            "with content classification" if "nsfw_label" in f_man[s]["filtered"]["pipeline_columns"]
            else "langid + PII + Gopher + fastText quality"
            for s in complete
        )),
        "bytes_on_hf": {"filtered": sum(f_sizes.values()), "unfiltered": sum(u_sizes.values())},
        "problems": problems,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=2))
    print(f"\n{len(problems)} problem(s)")
    for p in problems[:50]:
        print(" ", p)
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
