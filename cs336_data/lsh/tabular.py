"""MinHash+LSH deduplication for parquet/jsonl tabular outputs."""

import io
import json
import pickle
import shutil
import struct
import tempfile
import time
from array import array
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .hashing import (
    _generate_hash_params,
    compute_minhash_signature_from_ints,
    shingles_to_ints,
)
from .similarity import jaccard_similarity
from .text import normalize_text, word_ngrams
from .union_find import UnionFind


_WORKER_HASH_PARAMS = None  # type: Optional[List[Tuple[int, int]]]


def _init_signature_worker(hash_params):
    """Initializer for multiprocessing workers."""
    global _WORKER_HASH_PARAMS
    _WORKER_HASH_PARAMS = hash_params


def _compute_signature_components(text, ngrams, hash_params):
    """Compute normalized text, signature, and granular timing for one row."""
    norm_start = time.perf_counter()
    normalized = normalize_text(text)
    norm_seconds = time.perf_counter() - norm_start

    shingle_start = time.perf_counter()
    shingles = word_ngrams(normalized, ngrams)
    shingle_seconds = time.perf_counter() - shingle_start
    shingle_count = len(shingles)

    to_int_start = time.perf_counter()
    shingle_ints = shingles_to_ints(shingles)
    to_int_seconds = time.perf_counter() - to_int_start

    sig_core_start = time.perf_counter()
    signature = compute_minhash_signature_from_ints(shingle_ints, hash_params)
    sig_core_seconds = time.perf_counter() - sig_core_start

    return (
        normalized,
        signature,
        shingle_count,
        norm_seconds,
        shingle_seconds,
        to_int_seconds,
        sig_core_seconds,
    )


def _process_row_signature_worker(payload):
    """Multiprocessing worker payload handler."""
    file_idx, row_idx, text, ngrams = payload
    if _WORKER_HASH_PARAMS is None:
        raise RuntimeError("Worker hash params were not initialized.")

    (
        normalized,
        signature,
        shingle_count,
        norm_seconds,
        shingle_seconds,
        to_int_seconds,
        sig_core_seconds,
    ) = _compute_signature_components(text, ngrams, _WORKER_HASH_PARAMS)

    return (
        file_idx,
        row_idx,
        normalized,
        signature,
        shingle_count,
        norm_seconds,
        shingle_seconds,
        to_int_seconds,
        sig_core_seconds,
    )


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
    workers: int = 1,
    max_bucket_size: int = 5000,
    band_partitions: int = 128,
    verbose: bool = True,
) -> dict:
    """Memory-efficient MinHash+LSH dedup for parquet/jsonl text columns."""
    assert num_hashes % num_bands == 0, (
        f"num_hashes ({num_hashes}) must be evenly divisible by "
        f"num_bands ({num_bands})"
    )
    if workers is None:
        workers = 1
    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if max_bucket_size is not None:
        max_bucket_size = int(max_bucket_size)
        if max_bucket_size <= 1:
            raise ValueError("max_bucket_size must be > 1 or None")
    if band_partitions is None:
        band_partitions = 1
    band_partitions = int(band_partitions)
    if band_partitions <= 0:
        raise ValueError("band_partitions must be a positive integer")

    temp_dir_created = False
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="minhash_tabular_cache_")
        temp_dir_created = True
    else:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    signatures_file = Path(cache_dir) / "signatures.pkl"
    normalized_file = Path(cache_dir) / "normalized_texts.txt"

    hash_params = _generate_hash_params(num_hashes)
    # Compact arrays avoid large Python object overhead for big corpora.
    doc_file_indices = array("I")
    doc_row_indices = array("I")
    text_offsets = array("Q")
    num_docs = 0
    total_start = time.time()
    step_timings = {
        "step1_scan_and_signature_seconds": 0.0,
        "step1_normalization_seconds": 0.0,
        "step1_shingling_seconds": 0.0,
        "step1_shingle_to_int_seconds": 0.0,
        "step1_signature_core_seconds": 0.0,
        "step1_signature_seconds": 0.0,
        "step1_cache_write_seconds": 0.0,
        "step1_other_overhead_seconds": 0.0,
        "step1_docs_processed": 0,
        "step1_total_shingles": 0,
        "step2_lsh_bucketing_seconds": 0.0,
        "step2_band_partitions": band_partitions,
        "step2_skipped_large_buckets": 0,
        "step2_skipped_large_bucket_docs": 0,
        "step3_jaccard_verification_seconds": 0.0,
        "step4_component_reduce_seconds": 0.0,
        "total_seconds": 0.0,
    }

    try:
        step1_start = time.time()
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

            def _record_signature_row(file_idx, row_idx, normalized, signature):
                nonlocal num_docs
                cache_start = time.perf_counter()
                text_offsets.append(norm_f.tell())
                norm_f.write(normalized + "\n")
                doc_file_indices.append(file_idx)
                doc_row_indices.append(row_idx)

                sig_batch.append((num_docs, signature))
                num_docs += 1
                step_timings["step1_docs_processed"] += 1

                if len(sig_batch) >= batch_size:
                    pickle.dump(sig_batch, sig_f)
                    sig_batch.clear()
                step_timings["step1_cache_write_seconds"] += time.perf_counter() - cache_start

            if workers > 1:
                if verbose:
                    print("Using {} worker processes for scan+signature.".format(workers))
                pool_chunksize = max(1, min(batch_size, 128))
                work_iter = (
                    (file_idx, row_idx, text, ngrams)
                    for file_idx, row_idx, text in row_iter
                    if text is not None
                )
                with Pool(
                    processes=workers,
                    initializer=_init_signature_worker,
                    initargs=(hash_params,),
                ) as pool:
                    for (
                        file_idx,
                        row_idx,
                        normalized,
                        signature,
                        shingle_count,
                        norm_seconds,
                        shingle_seconds,
                        to_int_seconds,
                        sig_core_seconds,
                    ) in pool.imap(_process_row_signature_worker, work_iter, chunksize=pool_chunksize):
                        step_timings["step1_normalization_seconds"] += norm_seconds
                        step_timings["step1_shingling_seconds"] += shingle_seconds
                        step_timings["step1_total_shingles"] += shingle_count
                        step_timings["step1_shingle_to_int_seconds"] += to_int_seconds
                        step_timings["step1_signature_core_seconds"] += sig_core_seconds
                        step_timings["step1_signature_seconds"] += (to_int_seconds + sig_core_seconds)
                        _record_signature_row(file_idx, row_idx, normalized, signature)
            else:
                for file_idx, row_idx, text in row_iter:
                    if text is None:
                        continue
                    (
                        normalized,
                        signature,
                        shingle_count,
                        norm_seconds,
                        shingle_seconds,
                        to_int_seconds,
                        sig_core_seconds,
                    ) = _compute_signature_components(text, ngrams, hash_params)
                    step_timings["step1_normalization_seconds"] += norm_seconds
                    step_timings["step1_shingling_seconds"] += shingle_seconds
                    step_timings["step1_total_shingles"] += shingle_count
                    step_timings["step1_shingle_to_int_seconds"] += to_int_seconds
                    step_timings["step1_signature_core_seconds"] += sig_core_seconds
                    step_timings["step1_signature_seconds"] += (to_int_seconds + sig_core_seconds)
                    _record_signature_row(file_idx, row_idx, normalized, signature)

            if sig_batch:
                pickle.dump(sig_batch, sig_f)
        step_timings["step1_scan_and_signature_seconds"] = time.time() - step1_start
        granular_sum = (
            step_timings["step1_normalization_seconds"]
            + step_timings["step1_shingling_seconds"]
            + step_timings["step1_signature_seconds"]
            + step_timings["step1_cache_write_seconds"]
        )
        step_timings["step1_other_overhead_seconds"] = max(
            0.0,
            step_timings["step1_scan_and_signature_seconds"] - granular_sum,
        )

        if num_docs == 0:
            step_timings["total_seconds"] = time.time() - total_start
            return {
                "total": 0,
                "kept": 0,
                "removed": 0,
                "candidate_pairs_checked": 0,
                "verified_pairs": 0,
                "keep_rows_per_file": {},
                "total_shingles": 0,
                "timing": step_timings,
            }

        rows_per_band = num_hashes // num_bands
        active_band_partitions = 1 if (band_partitions <= 1 or num_docs < 1000000) else band_partitions
        step_timings["step2_band_partitions"] = active_band_partitions
        if verbose:
            print(
                "Step 2/3: LSH bucketing + candidate verification (streaming low-memory)"
                f" with {active_band_partitions} partition(s)..."
            )

        def load_signatures_streaming() -> Iterator[Tuple[int, List[int]]]:
            with open(signatures_file, "rb") as f:
                try:
                    while True:
                        batch = pickle.load(f)
                        for doc_idx, signature in batch:
                            yield doc_idx, signature
                except EOFError:
                    pass

        uf = UnionFind(num_docs)
        verified_pairs = 0
        candidate_pairs_checked = 0
        shingle_cache = OrderedDict()  # type: OrderedDict
        max_cache_entries = 4096
        skipped_large_buckets = 0
        skipped_large_bucket_docs = 0

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

            for band_idx in range(num_bands):
                band_start = band_idx * rows_per_band
                band_end = band_start + rows_per_band

                band_collision_buckets = 0
                band_step2_start = time.time()
                partition_paths = []  # type: List[Path]

                if active_band_partitions > 1:
                    record_struct = struct.Struct("<I" + ("Q" * rows_per_band))
                    band_partition_dir = Path(cache_dir) / f"band_partitions_{band_idx:03d}"
                    band_partition_dir.mkdir(parents=True, exist_ok=True)
                    partition_paths = [band_partition_dir / f"part_{p:04d}.bin" for p in range(active_band_partitions)]
                    partition_files = [open(str(p), "wb") for p in partition_paths]
                    try:
                        signature_stream = load_signatures_streaming()
                        if verbose:
                            signature_stream = tqdm(
                                signature_stream,
                                desc=f"LSH band {band_idx + 1}/{num_bands}",
                                total=num_docs,
                                leave=False,
                            )
                        for doc_idx, sig in signature_stream:
                            band_key = tuple(sig[band_start:band_end])
                            part_idx = (hash(band_key) & 0xFFFFFFFFFFFFFFFF) % active_band_partitions
                            partition_files[part_idx].write(record_struct.pack(doc_idx, *band_key))
                    finally:
                        for f in partition_files:
                            f.close()

                    step_timings["step2_lsh_bucketing_seconds"] += time.time() - band_step2_start

                    partition_iter = partition_paths
                    if verbose:
                        partition_iter = tqdm(
                            partition_paths,
                            desc=f"Bucket load band {band_idx + 1}/{num_bands}",
                            unit="part",
                            leave=False,
                        )

                    for part_path in partition_iter:
                        if not part_path.exists() or part_path.stat().st_size == 0:
                            if part_path.exists():
                                part_path.unlink()
                            continue

                        load_start = time.time()
                        band_buckets = {}  # type: Dict[Tuple[int, ...], Union[int, List[int]]]
                        with open(str(part_path), "rb") as pf:
                            while True:
                                record = pf.read(record_struct.size)
                                if not record:
                                    break
                                if len(record) != record_struct.size:
                                    raise RuntimeError(
                                        f"Corrupt partition file: {part_path} (partial record)"
                                    )
                                unpacked = record_struct.unpack(record)
                                doc_idx = unpacked[0]
                                band_key = unpacked[1:]
                                prev = band_buckets.get(band_key)
                                if prev is None:
                                    band_buckets[band_key] = doc_idx
                                elif isinstance(prev, int):
                                    if prev != doc_idx:
                                        band_buckets[band_key] = [prev, doc_idx]
                                else:
                                    prev.append(doc_idx)
                        step_timings["step2_lsh_bucketing_seconds"] += time.time() - load_start
                        part_path.unlink()

                        verify_start = time.time()
                        for members in band_buckets.values():
                            if isinstance(members, int):
                                continue
                            if max_bucket_size is not None and len(members) > max_bucket_size:
                                skipped_large_buckets += 1
                                skipped_large_bucket_docs += len(members)
                                continue
                            if len(members) <= 1:
                                continue

                            band_collision_buckets += 1
                            members.sort()
                            mlen = len(members)
                            for i_pos in range(mlen):
                                i = members[i_pos]
                                for j_pos in range(i_pos + 1, mlen):
                                    j = members[j_pos]
                                    candidate_pairs_checked += 1
                                    if uf.find(i) == uf.find(j):
                                        continue
                                    if jaccard_similarity(get_shingles(i), get_shingles(j)) >= jaccard_threshold:
                                        uf.union(i, j)
                                        verified_pairs += 1
                        step_timings["step3_jaccard_verification_seconds"] += time.time() - verify_start
                        band_buckets.clear()

                    if band_partition_dir.exists():
                        band_partition_dir.rmdir()
                else:
                    band_buckets = {}  # type: Dict[Tuple[int, ...], Union[int, List[int]]]
                    signature_stream = load_signatures_streaming()
                    if verbose:
                        signature_stream = tqdm(
                            signature_stream,
                            desc=f"LSH band {band_idx + 1}/{num_bands}",
                            total=num_docs,
                            leave=False,
                        )

                    for doc_idx, sig in signature_stream:
                        band_key = tuple(sig[band_start:band_end])
                        prev = band_buckets.get(band_key)
                        if prev is None:
                            band_buckets[band_key] = doc_idx
                        elif isinstance(prev, int):
                            if prev != doc_idx:
                                band_buckets[band_key] = [prev, doc_idx]
                        else:
                            prev.append(doc_idx)

                    step_timings["step2_lsh_bucketing_seconds"] += time.time() - band_step2_start

                    verify_start = time.time()
                    for members in band_buckets.values():
                        if isinstance(members, int):
                            continue
                        if max_bucket_size is not None and len(members) > max_bucket_size:
                            skipped_large_buckets += 1
                            skipped_large_bucket_docs += len(members)
                            continue
                        if len(members) <= 1:
                            continue

                        band_collision_buckets += 1
                        members.sort()
                        mlen = len(members)
                        for i_pos in range(mlen):
                            i = members[i_pos]
                            for j_pos in range(i_pos + 1, mlen):
                                j = members[j_pos]
                                candidate_pairs_checked += 1
                                if uf.find(i) == uf.find(j):
                                    continue
                                if jaccard_similarity(get_shingles(i), get_shingles(j)) >= jaccard_threshold:
                                    uf.union(i, j)
                                    verified_pairs += 1
                    step_timings["step3_jaccard_verification_seconds"] += time.time() - verify_start
                    band_buckets.clear()

                if verbose:
                    print(
                        f"  Band {band_idx + 1}/{num_bands}: "
                        f"collision buckets={band_collision_buckets:,}, "
                        f"candidates so far={candidate_pairs_checked:,}, "
                        f"verified so far={verified_pairs:,}"
                    )

        step_timings["step2_skipped_large_buckets"] = skipped_large_buckets
        step_timings["step2_skipped_large_bucket_docs"] = skipped_large_bucket_docs

        if verbose:
            print(f"🎯 Candidate pairs checked: {candidate_pairs_checked:,}")
            print(f"🔗 Verified duplicate pairs: {verified_pairs:,}")
            if skipped_large_buckets:
                print(
                    f"⚠ Skipped {skipped_large_buckets:,} oversized buckets "
                    f"(>{max_bucket_size} docs), covering {skipped_large_bucket_docs:,} docs."
                )

        step4_start = time.time()
        components = defaultdict(list)  # type: Dict[int, List[int]]
        doc_iter = range(num_docs)
        if verbose:
            doc_iter = tqdm(doc_iter, desc="Building components", unit="doc", leave=False)
        for idx in doc_iter:
            components[uf.find(idx)].append(idx)

        keep_doc_indices = sorted(min(members) for members in components.values())
        keep_rows_per_file = defaultdict(set)  # type: Dict[int, Set[int]]
        for doc_idx in keep_doc_indices:
            file_idx = doc_file_indices[doc_idx]
            row_idx = doc_row_indices[doc_idx]
            keep_rows_per_file[file_idx].add(row_idx)
        step_timings["step4_component_reduce_seconds"] = time.time() - step4_start

        removed = num_docs - len(keep_doc_indices)
        step_timings["total_seconds"] = time.time() - total_start
        return {
            "total": num_docs,
            "kept": len(keep_doc_indices),
            "removed": removed,
            "candidate_pairs_checked": candidate_pairs_checked,
            "verified_pairs": verified_pairs,
            "keep_rows_per_file": keep_rows_per_file,
            "total_shingles": step_timings.get("step1_total_shingles", 0),
            "timing": step_timings,
        }
    finally:
        if temp_dir_created and Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
