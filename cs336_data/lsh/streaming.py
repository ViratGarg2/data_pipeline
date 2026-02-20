"""Streaming MinHash+LSH deduplication for text files."""

import gc
import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from .candidates import lsh_buckets
from .hashing import _generate_hash_params, compute_minhash_signature
from .similarity import jaccard_similarity
from .text import normalize_text, word_ngrams
from .union_find import UnionFind


def minhash_lsh_deduplication_streaming(
    input_files: List[Union[str, Path]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Union[str, Path],
    batch_size: int = 1000,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Memory-efficient MinHash deduplication for large text-file datasets."""
    try:
        from xopen import xopen
    except ImportError as exc:
        raise ImportError("xopen is required. Install with: pip install xopen") from exc

    assert num_hashes % num_bands == 0, (
        f"num_hashes ({num_hashes}) must be evenly divisible by "
        f"num_bands ({num_bands})"
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_files = [Path(f) for f in input_files]
    num_docs = len(input_files)

    if num_docs == 0:
        return {"total": 0, "kept": 0, "removed": 0}

    temp_dir_created = False
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="minhash_cache_")
        temp_dir_created = True
    else:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        if verbose:
            print(f"📁 Using cache directory: {cache_dir}")
            print(f"🔢 Processing {num_docs} documents in batches of {batch_size}")

        hash_params = _generate_hash_params(num_hashes)
        signatures_file = Path(cache_dir) / "signatures.pkl"

        if verbose:
            print("🔄 Computing MinHash signatures...")

        with open(signatures_file, "wb") as sig_f:
            progress_bar = (
                tqdm(range(0, num_docs, batch_size), desc="Signature batches")
                if verbose
                else range(0, num_docs, batch_size)
            )

            for batch_start in progress_bar:
                batch_end = min(batch_start + batch_size, num_docs)
                batch_files = input_files[batch_start:batch_end]

                batch_signatures = []
                for i, fpath in enumerate(batch_files):
                    doc_idx = batch_start + i

                    with xopen(str(fpath), "rt") as f:
                        content = f.read()

                    normalized = normalize_text(content)
                    shingles = word_ngrams(normalized, ngrams)
                    signature = compute_minhash_signature(shingles, hash_params)

                    batch_signatures.append((doc_idx, signature))

                    del content, normalized, shingles

                pickle.dump(batch_signatures, sig_f)
                del batch_signatures
                gc.collect()

        if verbose:
            print("🔍 Finding LSH candidates...")

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
        gc.collect()

        if verbose:
            print("✅ Verifying candidates with Jaccard similarity...")

        def get_shingle_set_for_doc(doc_idx: int) -> Set[str]:
            fpath = input_files[doc_idx]
            with xopen(str(fpath), "rt") as f:
                content = f.read()
            normalized = normalize_text(content)
            return word_ngrams(normalized, ngrams)

        uf = UnionFind(num_docs)
        verified_pairs = 0

        progress_bar = tqdm(candidate_pairs, desc="Jaccard verification") if verbose else candidate_pairs

        for i, j in progress_bar:
            shingles_i = get_shingle_set_for_doc(i)
            shingles_j = get_shingle_set_for_doc(j)

            sim = jaccard_similarity(shingles_i, shingles_j)
            if sim >= jaccard_threshold:
                uf.union(i, j)
                verified_pairs += 1

            del shingles_i, shingles_j

        if verbose:
            print(f"🔗 Verified {verified_pairs} duplicate pairs")

        components = defaultdict(list)  # type: Dict[int, List[int]]
        for idx in range(num_docs):
            root = uf.find(idx)
            components[root].append(idx)

        docs_to_keep = set()  # type: Set[int]
        for members in components.values():
            docs_to_keep.add(min(members))

        if verbose:
            print("💾 Writing deduplicated files...")

        progress_bar = tqdm(sorted(docs_to_keep), desc="Writing files") if verbose else sorted(docs_to_keep)

        for idx in progress_bar:
            with xopen(str(input_files[idx]), "rt") as f:
                content = f.read()

            out_path = output_directory / input_files[idx].name
            with xopen(str(out_path), "wt") as f:
                f.write(content)

        removed = num_docs - len(docs_to_keep)

        if verbose:
            print(f"✅ Deduplication complete: {len(docs_to_keep)} kept, {removed} removed")

        return {
            "total": num_docs,
            "kept": len(docs_to_keep),
            "removed": removed,
            "candidate_pairs_checked": len(candidate_pairs),
            "verified_pairs": verified_pairs,
        }

    finally:
        if temp_dir_created and Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
