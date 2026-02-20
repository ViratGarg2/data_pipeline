"""LSH bucketing and candidate pair generation."""

import itertools
from collections import defaultdict
from typing import Dict, Iterator, List, Set, Tuple, Union


def lsh_buckets(
    signatures: Union[List[List[int]], Iterator[Tuple[int, List[int]]]],
    num_bands: int,
) -> Set[Tuple[int, int]]:
    """Identify candidate duplicate pairs via LSH banding."""
    if isinstance(signatures, list):
        if not signatures:
            return set()
        num_hashes = len(signatures[0])
        signature_iter = iter(enumerate(signatures))  # type: Iterator[Tuple[int, List[int]]]
    else:
        raw_iter = iter(signatures)
        first_item = next(raw_iter, None)
        if first_item is None:
            return set()
        first_doc_idx, first_sig = first_item
        num_hashes = len(first_sig)
        signature_iter = itertools.chain([(first_doc_idx, first_sig)], raw_iter)

    if num_hashes % num_bands != 0:
        raise ValueError(
            f"num_hashes ({num_hashes}) must be evenly divisible by num_bands ({num_bands})"
        )

    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(lambda: defaultdict(set))  # type: Dict[int, Dict[Tuple, Set[int]]]

    for doc_idx, sig in signature_iter:
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            band_key = tuple(sig[start : start + rows_per_band])
            buckets[band_idx][band_key].add(doc_idx)

    candidate_pairs = set()  # type: Set[Tuple[int, int]]
    for band_buckets in buckets.values():
        for members in band_buckets.values():
            if len(members) > 1:
                members_list = sorted(members)
                for i in range(len(members_list)):
                    for j in range(i + 1, len(members_list)):
                        candidate_pairs.add((members_list[i], members_list[j]))

    return candidate_pairs
