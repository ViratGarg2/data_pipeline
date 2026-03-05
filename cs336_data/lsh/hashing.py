"""Deterministic hashing and MinHash signature computation."""

import hashlib
import struct
from typing import List, Set, Tuple

from .constants import MERSENNE_PRIME


def shingle_to_int(shingle: str) -> int:
    """Map a shingle to a deterministic 64-bit integer."""
    h = hashlib.sha1(shingle.encode("utf-8")).digest()[:8]
    return struct.unpack("<Q", h)[0]


def shingles_to_ints(shingle_set: Set[str]) -> List[int]:
    """Convert shingles to deterministic integer ids."""
    if not shingle_set:
        return []
    return [shingle_to_int(s) for s in shingle_set]


def _generate_hash_params(num_hashes: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Generate deterministic (a, b) parameters for linear hash family."""
    import random

    rng = random.Random(seed)
    params = []  # type: List[Tuple[int, int]]
    for _ in range(num_hashes):
        a = rng.randint(1, MERSENNE_PRIME - 1)
        b = rng.randint(0, MERSENNE_PRIME - 1)
        params.append((a, b))
    return params


def compute_minhash_signature(
    shingle_set: Set[str],
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """Compute MinHash signature for a shingle set."""
    shingle_ints = shingles_to_ints(shingle_set)
    return compute_minhash_signature_from_ints(shingle_ints, hash_params)


def compute_minhash_signature_from_ints(
    shingle_ints: List[int],
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """Compute MinHash signature from integer shingles."""
    num_hashes = len(hash_params)
    signature = [MERSENNE_PRIME] * num_hashes

    if not shingle_ints:
        return signature

    for s_int in shingle_ints:
        for i, (a, b) in enumerate(hash_params):
            h = (a * s_int + b) % MERSENNE_PRIME
            if h < signature[i]:
                signature[i] = h
    return signature
