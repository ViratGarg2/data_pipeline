import random

from cs336_data.lsh.constants import MERSENNE_PRIME
from cs336_data.lsh.hashing import (
    _compute_minhash_signature_python,
    _generate_hash_params,
    compute_minhash_signature_from_ints,
)


def test_vectorized_minhash_matches_reference():
    rng = random.Random(0)
    params = _generate_hash_params(256)
    cases = [[], [0], [2**64 - 1], [MERSENNE_PRIME], [MERSENNE_PRIME - 1, 2**61, 2**63, 1]]
    cases += [[rng.getrandbits(64) for _ in range(rng.randint(1, 3000))] for _ in range(20)]
    for shingle_ints in cases:
        assert compute_minhash_signature_from_ints(shingle_ints, params) == _compute_minhash_signature_python(
            shingle_ints, params
        )
