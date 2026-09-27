"""Backward-compatible import surface for MinHash+LSH deduplication.

The implementation has been moved to ``cs336_data.lsh`` for modularity.
This module re-exports the same public API so existing imports continue to work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_data.lsh import (
    MAX_HASH,
    MERSENNE_PRIME,
    UnionFind,
    _generate_hash_params,
    _iter_text_rows_from_tabular_files,
    compute_minhash_signature,
    deduplicate_large_dataset,
    jaccard_similarity,
    lsh_buckets,
    minhash_lsh_deduplication,
    minhash_lsh_deduplication_on_texts,
    minhash_lsh_deduplication_streaming,
    minhash_lsh_deduplication_tabular_files,
    normalize_text,
    shingle_to_int,
    word_ngrams,
)

__all__ = [
    "MERSENNE_PRIME",
    "MAX_HASH",
    "normalize_text",
    "word_ngrams",
    "shingle_to_int",
    "_generate_hash_params",
    "compute_minhash_signature",
    "lsh_buckets",
    "jaccard_similarity",
    "UnionFind",
    "minhash_lsh_deduplication_streaming",
    "_iter_text_rows_from_tabular_files",
    "minhash_lsh_deduplication_tabular_files",
    "minhash_lsh_deduplication_on_texts",
    "minhash_lsh_deduplication",
    "deduplicate_large_dataset",
]
