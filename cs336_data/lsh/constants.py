"""Shared constants for MinHash + LSH deduplication."""

# Mersenne prime 2^61 - 1, used as modulus for linear hash functions.
MERSENNE_PRIME: int = (1 << 61) - 1

# Legacy upper bound kept for compatibility.
MAX_HASH: int = (1 << 32) - 1
