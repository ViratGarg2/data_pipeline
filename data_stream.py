import time
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
INPUT_GLOB = "/data3/dataset/the_pile_deduplicated/data/train-*-of-01650-*.parquet"
OUTPUT_DIR = "/home2/mehulag022/processed_data"
BATCH_SIZE = 100
COMPRESSION = "zstd"
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

start_time = time.time()

for sample in tqdm(dataset):
    if "text" in sample and isinstance(sample["text"], str):
        count = sample["text"].count("?")
        if count > 0:
            # sample["text"] = sample["text"].replace("?", "/")
            total_replacements += count

    buffer.append(sample)
    total_samples += 1

    if len(buffer) == BATCH_SIZE:
        table = pa.Table.from_pylist(buffer)

        if writer is None:
            out_path = f"{OUTPUT_DIR}/part-{part_id:05d}.parquet"
            writer = pq.ParquetWriter(
                out_path,
                table.schema,
                compression=COMPRESSION
            )

        writer.write_table(table)
        buffer.clear()
        break

        # rotate file every batch (recommended)
        writer.close()
        writer = None
        part_id += 1

    if total_samples % 100 == 0:
        elapsed = time.time() - start_time
        print(
            f"Processed {total_samples:,} samples | "
            f"Replacements: {total_replacements:,} | "
            f"Time: {elapsed:.1f}s"
        )

# write remaining samples
if buffer:
    table = pa.Table.from_pylist(buffer)
    out_path = f"{OUTPUT_DIR}/part-{part_id:05d}.parquet"
    pq.write_table(table, out_path, compression=COMPRESSION)

end_time = time.time()

print("\n==== FINAL STATS ====")
print(f"Total samples processed : {total_samples:,}")
print(f"Total '?' replaced      : {total_replacements:,}")
print(f"Total time (seconds)    : {end_time - start_time:.2f}")
