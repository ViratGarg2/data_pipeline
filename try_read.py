import glob
import pyarrow.parquet as pq

pattern = "/data3/dataset/the_pile_deduplicated/data/train-*-of-01650-*.parquet"
files = sorted(glob.glob(pattern))

if not files:
    print(f"No files found matching: {pattern}")
else:
    print(f"Found {len(files)} file(s). Reading first 5 rows from: {files[0]}\n")
    table = pq.read_table(files[0]).slice(0, 5)
    for i, row in enumerate(table.to_pylist()):
        print(f"[{i}] {row}\n")
