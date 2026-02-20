"""
Configuration for the data processing pipeline.
"""
HUGGINGFACE_KEY = "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"
# Dataset repo to upload processed files (format: "username/repo")
HUGGINGFACE_DATASET_REPO = ""
# Optional folder path inside the dataset repo, e.g. "processed/train"
HUGGINGFACE_DATASET_PATH_PREFIX = ""
# Glob pattern for input files
INPUT_GLOB = "/data3/dataset/the_pile_deduplicated/data/train-*-of-01650-*.parquet"

INPUT_GLOB_JSON = "/data3/dataset/eleuther-ai-the-pile-v-1/pile/train/*.jsonl.zst"
# Directory to save processed files
OUTPUT_DIR = "/home2/mehulag022/processed_data_html_extracted_lang"

# ---------------- DEDUPLICATION CONFIG ----------------
# MinHash + LSH settings used for fuzzy deduplication. These are read by
# the pipeline (process_html_pipeline.py) when deduplication is enabled.
# Modify these values to tune recall/precision tradeoffs.
DEDUPLICATION_ENABLED = True
DEDUP_NUM_HASHES = 256
DEDUP_NUM_BANDS = 32
DEDUP_NGRAMS = 5
DEDUP_JACCARD_THRESHOLD = 0.5
