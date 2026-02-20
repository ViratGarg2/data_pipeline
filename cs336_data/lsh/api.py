"""Public API for MinHash+LSH deduplication."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from tqdm import tqdm

from .candidates import lsh_buckets
from .hashing import _generate_hash_params, compute_minhash_signature
from .similarity import jaccard_similarity
from .streaming import minhash_lsh_deduplication_streaming
from .text import normalize_text, word_ngrams
from .union_find import UnionFind


def minhash_lsh_deduplication_on_texts(
    input_texts: List[Optional[str]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    verbose: bool = True,
) -> dict:
    """Run MinHash+LSH deduplication directly on in-memory texts."""
    assert num_hashes % num_bands == 0, (
        f"num_hashes ({num_hashes}) must be evenly divisible by "
        f"num_bands ({num_bands})"
    )

    num_docs = len(input_texts)
    if num_docs == 0:
        return {
            "total": 0,
            "kept": 0,
            "removed": 0,
            "candidate_pairs_checked": 0,
            "verified_pairs": 0,
            "keep_indices": [],
        }

    hash_params = _generate_hash_params(num_hashes)
    signatures = []  # type: List[List[int]]
    shingle_sets = []  # type: List[Set[str]]

    text_iter = input_texts
    if verbose:
        text_iter = tqdm(text_iter, desc="Computing MinHash", unit="docs", leave=False)

    for text in text_iter:
        normalized = normalize_text(text or "")
        shingles = word_ngrams(normalized, ngrams)
        shingle_sets.append(shingles)
        signatures.append(compute_minhash_signature(shingles, hash_params))

    candidate_pairs = lsh_buckets(signatures, num_bands)

    uf = UnionFind(num_docs)
    verified_pairs = 0
    pair_iter = candidate_pairs
    if verbose:
        pair_iter = tqdm(pair_iter, desc="Verifying Jaccard", unit="pairs", leave=False)

    for i, j in pair_iter:
        if jaccard_similarity(shingle_sets[i], shingle_sets[j]) >= jaccard_threshold:
            uf.union(i, j)
            verified_pairs += 1

    components = defaultdict(list)  # type: Dict[int, List[int]]
    for idx in range(num_docs):
        components[uf.find(idx)].append(idx)

    keep_indices = sorted(min(members) for members in components.values())
    removed = num_docs - len(keep_indices)

    return {
        "total": num_docs,
        "kept": len(keep_indices),
        "removed": removed,
        "candidate_pairs_checked": len(candidate_pairs),
        "verified_pairs": verified_pairs,
        "keep_indices": keep_indices,
    }


def minhash_lsh_deduplication(
    input_files: Optional[list],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Optional[Union[str, Path]],
    auto_streaming: bool = True,
    streaming_threshold: int = 5000,
    batch_size: int = 1000,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    input_texts: Optional[List[Optional[str]]] = None,
) -> dict:
    """Perform fuzzy document deduplication with MinHash + LSH."""
    if input_texts is not None:
        return minhash_lsh_deduplication_on_texts(
            input_texts=input_texts,
            num_hashes=num_hashes,
            num_bands=num_bands,
            ngrams=ngrams,
            jaccard_threshold=jaccard_threshold,
            verbose=verbose,
        )

    if input_files is None or output_directory is None:
        raise ValueError("input_files and output_directory are required when input_texts is not provided")

    # Keep behavior aligned with prior implementation (always streaming for file inputs).
    _ = auto_streaming, streaming_threshold
    return minhash_lsh_deduplication_streaming(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
        batch_size=batch_size,
        cache_dir=cache_dir,
        verbose=verbose,
    )


def deduplicate_large_dataset(
    input_files: List[Union[str, Path]],
    output_directory: Union[str, Path],
    num_hashes: int = 256,
    num_bands: int = 32,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
    batch_size: int = 500,
    cache_dir: Optional[str] = None,
) -> dict:
    """Convenience wrapper for low-memory large-dataset deduplication."""
    return minhash_lsh_deduplication_streaming(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
        batch_size=batch_size,
        cache_dir=cache_dir,
        verbose=True,
    )
