import time
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
INPUT_GLOB = "/home2/mehulag022/processed_data_html_extracted_lang/train-00000-*.parquet"
# OUTPUT_DIR = "/data3/processed_dataset"
# BATCH_SIZE = 100
# COMPRESSION = "zstd"
# ----------------------------------------

# os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "parquet",
    data_files=INPUT_GLOB,
    split="train",
    streaming=True
)

total_replacements = 0
total_samples = 0
buffer = []
writer = None
part_id = 0
indx = 0
start_time = time.time()

for sample in tqdm(dataset):
    if "text" in sample and isinstance(sample["text"], str) and indx<4:

        print(sample["nsfw_label"],sample["toxic_score"],sample["text"][:1000])  # Print first 100 characters of the text field
        # count = sample["text"].count("?")
        # total_replacements += count
        indx += 1


print("\n==== FINAL STATS ====")
# print(f"Total samples processed : {total_replacements}")
print(f"Total documents processed : {indx}")
