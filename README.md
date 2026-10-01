# Filtered Pile 13B

A cluster-scale pipeline that cleans web text for language-model pretraining, and a controlled experiment to measure what the cleaning is worth: two GPT-2-small runs on token-matched filtered and unfiltered data drawn from the same source files.

Built on [Stanford CS336 Assignment 4 (Data)](./data_assign.pdf). Every number below comes from a file in this repository; run `python metrics/show_results.py` to recompute them.

| | |
|---|---|
| Datasets | [ViratGarg/pile_filtered_13B](https://huggingface.co/datasets/ViratGarg/pile_filtered_13B), [ViratGarg/pile_unfiltered_13B](https://huggingface.co/datasets/ViratGarg/pile_unfiltered_13B) |
| Trained models (25k steps, every 2k-step checkpoint) | [ViratGarg/cs336-a4-filtered-25k](https://huggingface.co/ViratGarg/cs336-a4-filtered-25k), [ViratGarg/cs336-a4-unfiltered-25k](https://huggingface.co/ViratGarg/cs336-a4-unfiltered-25k) |
| Experiment tracking | [W&B project `cs336-data`](https://wandb.ai/gargvirat5-iiit-hyderabad/cs336-data) |

## Highlights

| | |
|---|---|
| **Perplexity on clean web text (Paloma C4)** | **31.7** with filtered data vs 35.1 without: 9.7% lower (loss 3.455 vs 3.558), same 162M model, same 3.3B tokens, same seed |
| **Compute efficiency** | The unfiltered run needs **≈1.43× the training compute** to reach the filtered run's loss |
| **More natural text** | Median GPT-2 perplexity of generations **9.74 vs 10.89**; filtered text reads more naturally on 30 of 50 prompts |
| **Pipeline throughput** | **≈970 documents/s** per 8-CPU task on Intel Xeon E5-2640 v4 nodes: **2.9 min per GB of raw text** (≈21 GB/hour), 1.6 min/GB with 16 workers ([hardware](#hardware)) |
| **Dedup optimisation** | MinHash-LSH dedup **15.3× faster** (73 min → 4.8 min) with bit-identical output |
| **Scale** | 22.0M Pile documents filtered across 19 parallel cluster tasks into two token-matched 13.59B-token datasets |

**Same prompt, both models** (seed and sampling identical; full text in the [CSV](inference_check/results/compare_25k_20261002_004204.csv), prompt 16):

> **Prompt:** The human heart pumps blood by
>
> **Filtered model** (GPT-2 ppl 6.3): …pumping it into the veins. This is called heart pump activity. The heart pump activity is a means of pumping blood to the lungs from a place in the body called the heart.
>
> **Unfiltered model** (GPT-2 ppl 16.4): …the end of the day. This is a real problem, and it will never pass. I think I have to mention that since I wrote this in 2009, I have been able to get a good feeling for your situation.

## Key results

| Result | Value | Evidence |
|---|---|---|
| Paloma C4 validation loss, filtered vs unfiltered (same model, tokens, seed) | **3.455 vs 3.558** (perplexity 31.7 vs 35.1) | `metrics/train_25k_eval_curves.csv` |
| Compute the unfiltered run needs to match the filtered run's loss | ≈1.4× (its final 3.558 = filtered at ~17.4k of 25k steps) | `metrics/train_25k_eval_curves.csv` |
| Held-out raw Pile loss, filtered vs unfiltered | 3.155 vs **2.646**: unfiltered wins on raw Pile text | `inference_check/results/compare_25k_summary_20261002_004204.json` |
| Pile documents processed | 21,979,350 (270 shards × 81,405) | `shards_complete` in `metrics/matched_data_summary.json`; `unfiltered.raw_docs_in_shard` in each dataset manifest |
| Tokens in each dataset, matched exactly | 13,589,978,522 | `metrics/matched_data_summary.json` |
| Filtering throughput, production settings, 8 workers per task | 970 rows/s; 2.9 min per GB of raw text (33.1 GB measured) | `metrics/pipeline_shards_*.json` |
| Dedup speedup after vectorising MinHash, identical output | 15.3× (4,390.8 s → 287.5 s) | `metrics/bench/stage_plus_dedup*.json` |
| Filtering speedup, 1 → 16 workers | 5.72× (304 → 1,739 rows/s) | `metrics/bench/scale_w*.json` |
| Median / p99 filtering latency per document | 2.45 ms / 61.71 ms | `metrics/bench/stage_production.json` |
| Cost of PII masking (runtime saved when disabled) | 44.6% of wall time | `metrics/bench/stage_no_pii.json` |
| Tests | 22 | `uv run pytest` |

## Pipeline

```mermaid
flowchart LR
    A[Pile shard / CC WET file] --> B[HTML detect + extract]
    B --> D[Language ID<br/>fastText lid.176]
    D --> E[Gopher rules]
    E --> F[Quality classifier<br/>fastText]
    F -.optional.-> G[NSFW / toxicity]
    G --> C[PII masking<br/>email, phone, IP]
    C -.optional.-> H[MinHash-LSH dedup]
    H --> I[GPT-2 tokenise<br/>+ token matching]
    I --> J[Hugging Face]
```

PII masking now runs last, on kept documents only, since it is the most expensive stage; the datasets and benchmarks below were produced with the earlier order (PII masking before language ID), and the new order has not been re-benchmarked yet. Inputs can be Pile parquet shards, JSONL or Common Crawl WET files (`--input-format wet`). Each shard is processed by its own worker process (`--workers`), and shards are spread across cluster nodes with SLURM job arrays. Every stage is timed per document and every removal is counted by reason; `--metrics-json` writes a report with throughput, per-stage CPU time, latency percentiles, removals by reason and peak memory ([`cs336_data/pipeline_metrics.py`](cs336_data/pipeline_metrics.py)).

### Where 2.44 million documents went

Production run on shards 180–209 (30 shards, 4 workers, 2,860.7 s, 853.7 rows/s). Evidence: `metrics/production_run_180-209_summary.txt`.

| Outcome | Documents | Share |
|---|---:|---:|
| Kept | 1,118,405 | 45.8% |
| Removed by Gopher rules | 752,714 | 30.8% |
| Removed by quality classifier | 306,082 | 12.5% |
| Removed as not English | 264,949 | 10.8% |

458,929 PII spans (emails, phone numbers, IP addresses) were masked.

## Benchmarks

Controlled runs on shards 300+ (not part of the training data) on the cluster nodes described in [Hardware](#hardware), with production filter settings unless stated.

### Hardware

All data processing ran as SLURM jobs on the `u22` partition of IIIT Hyderabad's Ada cluster (CPU details checked with `lscpu` on a node that ran production tasks):

| | Data processing (each cluster node) |
|---|---|
| CPU | 2 × Intel Xeon E5-2640 v4 @ 2.40 GHz (Broadwell, turbo up to 3.4 GHz) |
| Cores / threads | 20 cores, 40 hardware threads (2 threads per core) |
| Memory | 128 GB |
| Allocation per task | 8 SLURM CPUs (hardware threads) and 40 GB RAM for the filtering tasks behind the throughput figures (`jobs/matched_new_shards.sh`); 4 CPUs and 48 GB for the shards 180–209 run (`run_pipeline.sh`). One worker process per CPU; each worker peaks at ~3 GB (fastText models) |
| Storage | network home directory for code and models; node-local SSD scratch (~500 GB free) for staging, deleted after upload |
| Parallelism | up to 19 tasks at once, spread over about 10 nodes (several tasks per node); nodes were shared with other users' jobs |

Training ran separately on one NVIDIA RTX 3090 (24 GB, bf16) in a workstation with 16 CPU threads and 94 GB RAM.

### Worker scaling

16 shards × 20,000 rows. Evidence: `metrics/bench/scale_w{1,2,4,8,16}.json`.

| Workers | Wall time | Rows/s | Rows/s per worker | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,052.5 s | 304 | 304 | 1.00× | 100% |
| 2 | 721.9 s | 443 | 222 | 1.46× | 73% |
| 4 | 467.6 s | 684 | 173 | 2.25× | 56% |
| 8 | 285.5 s | 1,121 | 143 | 3.69× | 46% |
| 16 | 184.1 s | 1,739 | 115 | 5.72× | 36% |

Per-worker throughput falls as workers are added. Likely causes, not yet isolated: SLURM allocates hardware threads, so workers share physical cores; each worker loads its own 2.7 GB of fastText models from network storage; and the nodes are shared with other jobs.

### What each stage costs

One change at a time against production settings: 8 workers, 8 shards × 20,000 rows. Evidence: `metrics/bench/stage_*.json`.

| Variant | Wall time | Change | Docs kept | Effect on output |
|---|---:|---:|---:|---|
| Production settings | 180.8 s | | 45.2% | |
| No PII masking | 100.1 s | −44.6% | 45.3% | Same documents kept |
| No quality classifier | 133.1 s | −26.4% | 57.3% | 19,467 fewer removals |
| No Gopher rules | 147.4 s | −18.5% | 64.9% | Classifier removes 39,659 instead of 19,467 |
| No language ID | 164.7 s | −8.9% | 45.2% | Gopher removes 68,206 instead of 51,853 |
| Add NSFW / toxicity | 238.6 s | +32.0% | 45.0% | 225 more documents removed (0.14%) |
| Add MinHash-LSH dedup (original) | 4,569.7 s | +2,427% | 44.8% | 490 near-duplicates removed |
| Add MinHash-LSH dedup (optimised) | 431.0 s | +138% | 44.8% | Same 490 removed |

Findings:

- **PII masking is the most expensive filter**: 39.5% of CPU time. It ran on every document although only 45% survive filtering; it now runs after the filters (not yet re-benchmarked).
- **Language ID is largely redundant on the Pile**: without it, the Gopher rules reject non-English documents anyway and the kept share is unchanged.
- **NSFW / toxicity filtering is poor value here**: +32% runtime to remove 0.14% of documents.
- **The two quality filters overlap**: without Gopher, the classifier removes twice as many documents, yet the kept share still rises from 45.2% to 64.9%.

### Per-document latency by stage

Evidence: `metrics/bench/stage_production.json`.

| Stage | Docs in | CPU share | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| HTML detection | 160,000 | 0.4% | 0.03 ms | 0.06 ms | 0.16 ms |
| HTML extraction | 5,810 | 0.1% | 0.22 ms | 0.88 ms | 2.27 ms |
| PII masking | 159,511 | 39.5% | 1.08 ms | 11.98 ms | 35.76 ms |
| Language ID | 159,511 | 6.9% | 0.37 ms | 1.69 ms | 1.95 ms |
| Gopher rules | 143,078 | 22.1% | 0.67 ms | 7.52 ms | 22.16 ms |
| Quality classifier | 91,225 | 26.0% | 0.64 ms | 3.71 ms | 10.01 ms |
| **Whole document** | 160,000 | 100% | **2.45 ms** | **22.93 ms** | **61.71 ms** |

A CPU profile of one worker attributes about 38% of its time to Python regex calls (`metrics/bench/profile_tottime.txt`).

### Optimising MinHash dedup

Dedup was 24.5× slower than all filtering combined. The cause was the signature step in [`cs336_data/lsh/hashing.py`](cs336_data/lsh/hashing.py): for each of ~1,700 shingles per document and each of 256 hash functions, it computed `(a·x + b) mod (2^61 − 1)` with Python integers (about 434,000 operations per document at ~170 ns each), on a single core because the pipeline never passed its worker count to the dedup step.

Two changes:

1. **Vectorised, exact arithmetic.** The hash is computed for all shingles and hash functions at once with NumPy. Operands are split into 32-bit halves and reduced using 2^64 ≡ 8 and 2^61 ≡ 1 (mod 2^61 − 1), so every intermediate fits in 64 bits and signatures are bit-identical to the original ([`tests/test_minhash_vectorized.py`](tests/test_minhash_vectorized.py)).
2. **Parallel signatures.** `process_html_pipeline.py` now passes `--workers` to the dedup step.

Same benchmark before and after (8 workers, 8 shards × 20,000 rows). Evidence: `metrics/bench/stage_plus_dedup.json`, `metrics/bench/stage_plus_dedup_v2.json`.

| | Before | After |
|---|---:|---:|
| Dedup time | 4,390.8 s | 287.5 s (15.3× faster) |
| Whole run | 4,569.7 s | 431.0 s (10.6× faster) |
| End-to-end throughput | 35 rows/s | 371 rows/s |
| Documents in → kept | 72,247 → 71,757 | 72,247 → 71,757 |
| LSH candidates / confirmed duplicates | 3,818,117 / 490 | 3,818,117 / 490 |
| Filtering latency p50 / p95 / p99 | 2.48 / 23.06 / 61.88 ms | 2.36 / 21.92 / 58.60 ms (unchanged; node-to-node noise) |

The signature step is still about 87% of dedup time; per-document Unicode normalisation and one SHA-1 per shingle are the next targets. 99.99% of LSH candidates are false positives, but checking all 3.8M took only about 37 s.

## Token-matched datasets

For every source shard *i*, `pile_filtered_13B` holds the pipeline's output for shard *i*, and `pile_unfiltered_13B` holds a seeded random sample of raw documents from the same shard, drawn until it has exactly as many GPT-2 tokens (the last document is truncated). Filtering is the only difference between the two ([`cs336_data/prepare_matched_data.py`](cs336_data/prepare_matched_data.py)).

Evidence: `metrics/matched_data_summary.json`, produced by [`cs336_data/verify_matched_data.py`](cs336_data/verify_matched_data.py), which checks every file, every per-shard token count and every token file size on both repositories.

| | Filtered | Unfiltered |
|---|---:|---:|
| Source shards | 0–269 | 0–269 |
| GPT-2 tokens | 13,589,978,522 | 13,589,978,522 |
| Documents | 11,023,784 | 8,374,559 |
| Share of the assignment's 13.1B-token training budget | 103.7% | 103.7% |
| Verification problems | 0 | 0 |

251 shards were filtered with language ID, PII masking, Gopher rules and the quality classifier; 19 early shards also had NSFW/toxicity filtering. Each shard's manifest records its pipeline columns. Each repository contains `tokens/train-XXXXX.bin` (uint16, `<|endoftext|>` after every document), `text/train-XXXXX.parquet` and `manifests/train-XXXXX.json`.

## Training

Both runs use the assignment's GPT-2-small configuration (12 layers, d_model 768, 162M parameters, context 512, global batch 256 sequences, cosine schedule, seed 0); only the dataset differs. Shards stream from Hugging Face during training ([`cs336-basics/cs336_basics/streaming_data.py`](cs336-basics/cs336_basics/streaming_data.py)), so a run needs about 1 GB of disk instead of 27 GB.

Both runs trained for 25,000 steps (3.28B tokens each, ~17.7 h each on one RTX 3090): W&B runs [filtered_25k](https://wandb.ai/gargvirat5-iiit-hyderabad/cs336-data/runs/6eux2izo) and [unfiltered_25k](https://wandb.ai/gargvirat5-iiit-hyderabad/cs336-data/runs/2c8k5yoe). Every 2k-step checkpoint and the final model are on the Hub.

### Validation loss during training (Paloma C4)

Evidence: `metrics/train_25k_eval_curves.csv`, exported from W&B.

| Step | Tokens seen | Filtered | Unfiltered | Gap |
|---:|---:|---:|---:|---:|
| 2,000 | 0.26B | 4.082 | 4.199 | 0.117 |
| 6,000 | 0.79B | 3.803 | 3.911 | 0.108 |
| 10,000 | 1.31B | 3.702 | 3.850 | 0.148 |
| 14,000 | 1.84B | 3.606 | 3.712 | 0.106 |
| 18,000 | 2.36B | 3.545 | 3.654 | 0.109 |
| 22,000 | 2.88B | 3.485 | 3.582 | 0.097 |
| **25,000** | **3.28B** | **3.455** | **3.558** | **0.103** |

The filtered model is ahead at all 13 evaluations (gap 0.088–0.148). Each estimate averages 1,000 batches, so its uncertainty is roughly ±0.01, an order of magnitude below the gap. The unfiltered model's final loss is what the filtered model reached at about 17.4k steps (between 3.590 at 16k and 3.545 at 18k), so on this benchmark the filtered data is worth roughly 1.4× the compute. Each arm has one seed.

### Final models on two held-out sets, and generations

[`inference_check/compare_models.py`](inference_check/compare_models.py) evaluates both final models on the same 1.64M-token sample of each set (random 512-token windows) and generates from 50 prompts with the same seed per prompt (temperature 0.7, top-k 50, 100 new tokens). Each generation is scored by the original GPT-2 (124M) as a neutral fluency judge: the perplexity GPT-2 assigns to the generated continuation. Evidence: `inference_check/results/compare_25k_summary_20261002_004204.json`; all 50 prompts with both models' outputs are in [`inference_check/results/compare_25k_20261002_004204.csv`](inference_check/results/compare_25k_20261002_004204.csv).

| | Filtered | Unfiltered |
|---|---:|---:|
| Paloma C4: loss / perplexity | **3.457 / 31.7** | 3.558 / 35.1 |
| Paloma C4: next-token accuracy | **37.07%** | 36.00% |
| Held-out raw Pile (shard 1649, 4,000 docs): loss / perplexity | 3.155 / 23.5 | **2.646 / 14.1** |
| Held-out raw Pile: next-token accuracy | 44.47% | **50.40%** |
| Generations: median GPT-2 perplexity (lower = more natural) | **9.74** | 10.89 |
| Prompts where the filtered model's text was more natural to GPT-2 | 30 of 50 | 20 of 50 |
| Generations with >20% repeated 4-grams | 4 of 50 | 3 of 50 |

What this shows:

- **Filtering helps on clean web text and hurts on raw Pile text.** The filtered model is better on Paloma C4 and writes text GPT-2 finds more natural, but the unfiltered model is much better on held-out raw Pile. The filters removed most code, LaTeX, tables and other structured text; that content is common in raw Pile and very predictable (repetitive syntax), so the model that trained on it wins there. Filtering is a choice about the target distribution, not a free improvement.
- **Repetition is similar for both** (about 6% repeated 4-grams on average), so the GPT-2 fluency gap is not explained by one model looping more.

GPT-2 perplexity also rewards repetition (one filtered generation that loops on a single sentence scores 1.5), which is why the repeated-4-gram rate is reported alongside it. A same-prompt sample from both models is shown in [Highlights](#highlights).

### Throughput on the training GPU

Measured on the training GPU, an RTX 3090 in bf16 (evidence: `metrics/train_smoke_rtx3090.txt`, W&B run [12fd2vz3](https://wandb.ai/gargvirat5-iiit-hyderabad/cs336-data-smoke/runs/12fd2vz3)):

| | |
|---|---:|
| Tokens per optimizer step | 131,072 |
| Step time | ≈2.35 s |
| Throughput | ≈56k tokens/s |
| 120-step smoke test, Paloma C4 validation loss | 7.01 → 6.54 |

### Baseline: the earlier model

Trained on 0.82B tokens (batch 16 × 512 × 100k steps) on an RTX 2080 Ti in 10.2 h ([W&B run s1ml9n3n](https://wandb.ai/gargvirat5-iiit-hyderabad/cs336-data/runs/s1ml9n3n)). Re-evaluated with [`inference_check/infer_prev.py`](inference_check/infer_prev.py): Paloma C4 loss **3.797** (perplexity 44.6), matching the 3.81 logged during training. Evidence: `inference_check/results/prev_model_summary_20260927_172053.json`.

24 generations (8 prompts × 3 samples, temperature 0.7, top-k 50) are in [`inference_check/results/prev_model_samples_20260927_172053.csv`](inference_check/results/prev_model_samples_20260927_172053.csv). The text is grammatical and holds a register (news, recipe, encyclopedia) for a sentence or two, then drifts or loops, for example:

> The best way to cook pasta is to make it a part of the recipe, such as a salad, a salad or a salad.

The code prompt `def fibonacci(n):` produces no code, consistent with prose-only filtered training data.

## How to run

Commands run from the repository root. The SLURM scripts use the paths of the IIIT Ada cluster; change the paths at the top of each script for another cluster.

### 1. Environment, models and tests

```bash
uv sync
```

```bash
mkdir -p model
wget -O model/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
wget -O model/jigsaw_fasttext_bigrams_nsfw_final.bin https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin
wget -O model/jigsaw_fasttext_bigrams_hatespeech_final.bin https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin
```

```bash
cd urls && python train_quality_classifier.py --positive positive.txt --negative negative.txt --output quality_model && cd ..
```

```bash
uv run pytest
```

fastText 0.9.3 needs NumPy < 2.

### 2. Filter shards on one node, with a metrics report

```bash
python cs336_data/process_html_pipeline.py --start-index 300 --max-files 8 --max-rows 20000 --workers 8 --no-content-classification --output-dir out/ --metrics-json metrics/run.json
```

Add `--no-dedup` to skip MinHash-LSH, or `--push-to-hf --hf-repo-id USER/REPO` to upload each shard.

### 3. Production run on the cluster

```bash
sbatch run_pipeline.sh
sbatch jobs/matched_new_shards.sh
sbatch --array=0-8 jobs/matched_reuse_shards.sh
sbatch --export=ALL,SHARDS=136-155 jobs/matched_retry_resumable.sh
python cs336_data/verify_matched_data.py --summary-json metrics/matched_data_summary.json
```

`run_pipeline.sh` filters 30 shards and uploads them; the `matched_*` jobs build token-matched pairs from new and already-filtered shards; `matched_retry_resumable.sh` rebuilds and uploads with resumable, time-bounded retries.

### 4. Benchmarks

```bash
bash jobs/submit_benchmarks.sh
python metrics/show_results.py
```

### 5. Train one arm of the comparison

```bash
tmux new -s a4_filtered 'cs336-basics/scripts/launch_matched_run.sh filtered'
tmux new -s a4_unfiltered 'cs336-basics/scripts/launch_matched_run.sh unfiltered'
```

The launcher derives micro-batch, gradient accumulation and bf16/fp16 from the visible GPUs so the global batch stays at 256. Use `MICRO_BATCH=16` on smaller GPUs and Hydra overrides with `++`, e.g. `++training.train_steps=25000`.

### 6. Evaluate and sample a model

```bash
python inference_check/infer_prev.py /path/to/model_dir
```

It uses the GPU only when at least 3 GB is free and otherwise the CPU, and writes a CSV of prompts and generations plus a summary JSON to `inference_check/results/`.

To compare the two trained models (loss, perplexity and next-token accuracy on Paloma C4 and held-out raw Pile, plus 50 prompts scored by GPT-2):

```bash
python inference_check/compare_models.py work_dir/ /path/to/paloma_c4_100_val.bin
```

## Repository layout

| Path | Contents |
|---|---|
| `cs336_data/process_html_pipeline.py` | The filtering pipeline and its CLI |
| `cs336_data/pipeline_metrics.py` | Metrics report: throughput, latency percentiles, removals by reason |
| `cs336_data/lsh/` | MinHash-LSH near-duplicate detection (vectorised signatures) |
| `cs336_data/prepare_matched_data.py` | Builds token-matched filtered/unfiltered shard pairs and uploads them |
| `cs336_data/verify_matched_data.py` | Verifies both datasets file by file |
| `cs336-basics/` | Training code: model, streaming loader, configs, launcher |
| `jobs/` | SLURM scripts for dataset building and benchmarks |
| `metrics/` | Evidence for every number in this README, and `show_results.py` |
| `inference_check/` | Evaluation and sampling script, with saved results |
| `tests/` | Unit tests for every pipeline component |
| `urls/` | Quality-classifier data collection and training |

## Engineering notes

- Worker crashes were swallowed and jobs exited 0; failures are now listed and fail the SLURM job.
- `#SBATCH` directives placed after the first command were silently ignored; job scripts now declare resources correctly.
- fastText 0.9.3 breaks on NumPy 2 (`np.array(..., copy=False)`); NumPy is pinned below 2.
- Exact line dedup opened its output files in read mode; fixed, and its test passes again.
- Uploads through Hugging Face's Xet backend hung indefinitely from the cluster while plain LFS ran at full link speed; uploads now use LFS with per-attempt timeouts, resumable retries, and staging kept until success.
- Evaluation during training ran without `torch.no_grad()` and ran out of GPU memory; fixed without changing the computed loss.

## Acknowledgements

The assignment, handout and staff model implementation in `cs336-basics` come from Stanford CS336 (Spring 2025). NSFW and hate-speech classifiers are from [AllenAI Dolma](https://huggingface.co/allenai); language identification uses fastText `lid.176`. Source data is [EleutherAI's deduplicated Pile](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated). See also the [OLMo 3 paper](./olmo3.pdf).
