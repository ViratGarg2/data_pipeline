"""Modular MinHash+LSH package.

This package contains the deduplication implementation split across focused
modules while keeping the same public API names used by the pipeline.
"""

from .api import (
    deduplicate_large_dataset,
    minhash_lsh_deduplication,
    minhash_lsh_deduplication_on_texts,
)
from .candidates import lsh_buckets
from .constants import MAX_HASH, MERSENNE_PRIME
from .hashing import _generate_hash_params, compute_minhash_signature, shingle_to_int
from .similarity import jaccard_similarity
from .streaming import minhash_lsh_deduplication_streaming
from .tabular import _iter_text_rows_from_tabular_files, minhash_lsh_deduplication_tabular_files
from .text import normalize_text, word_ngrams
from .union_find import UnionFind

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
