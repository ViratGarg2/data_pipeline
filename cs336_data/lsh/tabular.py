"""MinHash+LSH deduplication for parquet/jsonl tabular outputs."""

import io
import json
import pickle
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .candidates import lsh_buckets
from .hashing import _generate_hash_params, compute_minhash_signature
from .similarity import jaccard_similarity
from .text import normalize_text, word_ngrams
from .union_find import UnionFind


def _truncate_to_max_words(text, max_words):
    """Keep only first ``max_words`` words from text for faster dedup."""
    if max_words is None:
        return text
    if max_words <= 0:
        return text

    parts = text.split(None, max_words)
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _open_text_reader(path):
    """Open JSONL-like files in text mode, including .zst."""
    path_str = str(path)
    if path_str.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required to read .zst files. Install with: pip install zstandard"
            ) from exc

        raw_f = open(path_str, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(raw_f)
        return io.TextIOWrapper(reader, encoding="utf-8")

    try:
        from xopen import xopen
    except ImportError:
        return open(path_str, "r", encoding="utf-8")
    return xopen(path_str, "rt")


def _iter_text_rows_from_tabular_files(
    input_files: List[Union[str, Path]],
    text_column: str,
    batch_size: int,
    max_rows_per_file: Optional[int] = None,
    max_words_per_row: Optional[int] = None,
    show_file_progress: bool = False,
) -> Iterator[Tuple[int, int, Optional[str]]]:
    """Yield `(file_idx, row_idx, text)` from parquet/jsonl files."""
    file_iter = input_files
    if show_file_progress:
        file_iter = tqdm(input_files, desc="Scanning files", unit="file", total=len(input_files), leave=False)

    for file_idx, raw_path in enumerate(file_iter):
        fpath = Path(raw_path)
        if str(fpath).endswith(".parquet"):
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "pyarrow is required to read parquet files in tabular deduplication mode."
                ) from exc

            pf = pq.ParquetFile(str(fpath))
            row_offset = 0
            for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
                if max_rows_per_file is not None and row_offset >= max_rows_per_file:
                    break
                texts = batch.column(0).to_pylist()
                if max_rows_per_file is not None:
                    remaining = max_rows_per_file - row_offset
                    if remaining <= 0:
                        break
                    if len(texts) > remaining:
                        texts = texts[:remaining]
                for in_batch_idx, text in enumerate(texts):
                    if text is not None:
                        if not isinstance(text, str):
                            text = str(text)
                        text = _truncate_to_max_words(text, max_words_per_row)
                    yield file_idx, row_offset + in_batch_idx, text
                row_offset += len(texts)
        else:
            valid_row_idx = 0
            with _open_text_reader(fpath) as f:
                try:
                    for line in f:
                        if max_rows_per_file is not None and valid_row_idx >= max_rows_per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = record.get(text_column)
                        if text is not None:
                            if not isinstance(text, str):
                                text = str(text)
                            text = _truncate_to_max_words(text, max_words_per_row)
                        yield file_idx, valid_row_idx, text
                        valid_row_idx += 1
                except UnicodeDecodeError as exc:
                    raise RuntimeError(
                        "Failed to decode {} as UTF-8 text. For .jsonl.zst files, "
                        "ensure zstandard is installed and the source file is valid.".format(fpath)
                    ) from exc


def minhash_lsh_deduplication_tabular_files(
    input_files: List[Union[str, Path]],
    text_column: str,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    batch_size: int = 1000,
    cache_dir: Optional[str] = None,
    max_rows_per_file: Optional[int] = None,
    max_words_per_row: Optional[int] = 10000,
    verbose: bool = True,
) -> dict:
    """Memory-efficient MinHash+LSH dedup for parquet/jsonl text columns."""
    assert num_hashes % num_bands == 0, (
        f"num_hashes ({num_hashes}) must be evenly divisible by "
        f"num_bands ({num_bands})"
    )

    temp_dir_created = False
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="minhash_tabular_cache_")
        temp_dir_created = True
    else:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    signatures_file = Path(cache_dir) / "signatures.pkl"
    normalized_file = Path(cache_dir) / "normalized_texts.txt"

    hash_params = _generate_hash_params(num_hashes)
    doc_locations = []  # type: List[Tuple[int, int]]
    text_offsets = []  # type: List[int]
    num_docs = 0

    try:
        if verbose:
            print("Step 1/3: Scanning rows and computing MinHash signatures...")

        row_iter = _iter_text_rows_from_tabular_files(
            input_files,
            text_column=text_column,
            batch_size=batch_size,
            max_rows_per_file=max_rows_per_file,
            max_words_per_row=max_words_per_row,
            show_file_progress=verbose,
        )
        if verbose:
            row_iter = tqdm(row_iter, desc="Scanning rows", unit="rows")

        with open(signatures_file, "wb") as sig_f, open(normalized_file, "w+", encoding="utf-8") as norm_f:
            sig_batch = []  # type: List[Tuple[int, List[int]]]

            for file_idx, row_idx, text in row_iter:
                if text is None:
                    continue

                normalized = normalize_text(text)
                shingles = word_ngrams(normalized, ngrams)
                signature = compute_minhash_signature(shingles, hash_params)

                text_offsets.append(norm_f.tell())
                norm_f.write(normalized + "\n")
                doc_locations.append((file_idx, row_idx))

                sig_batch.append((num_docs, signature))
                num_docs += 1

                if len(sig_batch) >= batch_size:
                    pickle.dump(sig_batch, sig_f)
                    sig_batch.clear()

            if sig_batch:
                pickle.dump(sig_batch, sig_f)

        if num_docs == 0:
            return {
                "total": 0,
                "kept": 0,
                "removed": 0,
                "candidate_pairs_checked": 0,
                "verified_pairs": 0,
                "keep_rows_per_file": {},
            }

        if verbose:
            print("Step 2/3: Finding LSH candidates...")

        def load_signatures_streaming() -> Iterator[Tuple[int, List[int]]]:
            with open(signatures_file, "rb") as f:
                try:
                    while True:
                        batch = pickle.load(f)
                        for doc_idx, signature in batch:
                            yield doc_idx, signature
                except EOFError:
                    pass

        signature_stream = load_signatures_streaming()
        if verbose:
            signature_stream = tqdm(signature_stream, desc="LSH bucketing", total=num_docs)
        candidate_pairs = lsh_buckets(signature_stream, num_bands=num_bands)

        if verbose:
            print(f"🎯 Found {len(candidate_pairs)} candidate pairs")

        if verbose:
            print("Step 3/3: Verifying candidates with Jaccard similarity...")

        uf = UnionFind(num_docs)
        verified_pairs = 0
        shingle_cache = OrderedDict()  # type: OrderedDict
        max_cache_entries = 4096

        with open(normalized_file, "r", encoding="utf-8") as norm_f:

            def get_shingles(doc_idx: int) -> Set[str]:
                if doc_idx in shingle_cache:
                    shingle_cache.move_to_end(doc_idx)
                    return shingle_cache[doc_idx]
                norm_f.seek(text_offsets[doc_idx])
                normalized = norm_f.readline().rstrip("\n")
                shingles = word_ngrams(normalized, ngrams)
                shingle_cache[doc_idx] = shingles
                if len(shingle_cache) > max_cache_entries:
                    shingle_cache.popitem(last=False)
                return shingles

            pair_iter = candidate_pairs
            if verbose:
                pair_iter = tqdm(pair_iter, desc="Jaccard verification", unit="pairs")

            for i, j in pair_iter:
                if jaccard_similarity(get_shingles(i), get_shingles(j)) >= jaccard_threshold:
                    uf.union(i, j)
                    verified_pairs += 1

        if verbose:
            print(f"🔗 Verified {verified_pairs} duplicate pairs")

        components = defaultdict(list)  # type: Dict[int, List[int]]
        doc_iter = range(num_docs)
        if verbose:
            doc_iter = tqdm(doc_iter, desc="Building components", unit="doc", leave=False)
        for idx in doc_iter:
            components[uf.find(idx)].append(idx)

        keep_doc_indices = sorted(min(members) for members in components.values())
        keep_rows_per_file = defaultdict(set)  # type: Dict[int, Set[int]]
        for doc_idx in keep_doc_indices:
            file_idx, row_idx = doc_locations[doc_idx]
            keep_rows_per_file[file_idx].add(row_idx)

        removed = num_docs - len(keep_doc_indices)
        return {
            "total": num_docs,
            "kept": len(keep_doc_indices),
            "removed": removed,
            "candidate_pairs_checked": len(candidate_pairs),
            "verified_pairs": verified_pairs,
            "keep_rows_per_file": keep_rows_per_file,
        }
    finally:
        if temp_dir_created and Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
