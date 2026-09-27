"""Deterministic hashing and MinHash signature computation."""

import hashlib
import struct
from typing import List, Set, Tuple

import numpy as np

from .constants import MERSENNE_PRIME

_P = np.uint64(MERSENNE_PRIME)
_MASK32 = np.uint64(0xFFFFFFFF)
_MASK29 = np.uint64((1 << 29) - 1)
# Rows of the (shingles x hashes) matrix processed at once, to bound memory for long documents.
_SHINGLE_CHUNK = 2048


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


def _fold(v: np.ndarray) -> np.ndarray:
    """Partially reduce uint64 values mod 2^61 - 1 (2^61 = 1 mod p); result < p + 8."""
    return (v & _P) + (v >> np.uint64(61))


def _mulmod_mersenne61(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(x * a) mod (2^61 - 1) for uint64 x (any 64-bit value) and a < 2^61, without overflow.

    Splits both operands into 32-bit halves and uses 2^64 = 8 and 2^61 = 1 (mod p),
    so every intermediate product fits in 64 bits.
    """
    xh, xl = x >> np.uint64(32), x & _MASK32
    ah, al = a >> np.uint64(32), a & _MASK32
    high = _fold((ah * xh) * np.uint64(8))              # ah*xh*2^64, with ah*xh < 2^61
    mid = _fold(ah * xl) + _fold(al * xh)              # < 2^62
    mid = (mid >> np.uint64(29)) + ((mid & _MASK29) << np.uint64(32))  # mid*2^32 mod p
    low = _fold(al * xl)
    r = _fold(high + mid + low)
    return np.where(r >= _P, r - _P, r)


def compute_minhash_signature_from_ints(
    shingle_ints: List[int],
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """Compute MinHash signature from integer shingles.

    Vectorised over (shingle, hash function) with NumPy; returns exactly the values of
    ``_compute_minhash_signature_python``, i.e. min over shingles of (a*x + b) mod (2^61 - 1).
    """
    num_hashes = len(hash_params)
    if not shingle_ints:
        return [MERSENNE_PRIME] * num_hashes

    a = np.fromiter((p[0] for p in hash_params), dtype=np.uint64, count=num_hashes)
    b = np.fromiter((p[1] for p in hash_params), dtype=np.uint64, count=num_hashes)
    x = np.fromiter(shingle_ints, dtype=np.uint64, count=len(shingle_ints))

    signature = np.full(num_hashes, _P, dtype=np.uint64)
    for start in range(0, len(x), _SHINGLE_CHUNK):
        xc = x[start:start + _SHINGLE_CHUNK, None]
        h = _mulmod_mersenne61(xc, a[None, :]) + b[None, :]   # < 2p, fits in uint64
        h = np.where(h >= _P, h - _P, h)
        np.minimum(signature, h.min(axis=0), out=signature)
    return signature.tolist()


def _compute_minhash_signature_python(
    shingle_ints: List[int],
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """Pure-Python reference implementation (slow; kept for tests)."""
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
