"""
Fuzzy Document Deduplication using MinHash and Locality-Sensitive Hashing (LSH)

This module implements scalable, memory-efficient fuzzy (near-duplicate) document
deduplication following the approach described in Penedo et al. (2023).

Overview
========
Exact deduplication only catches identical documents. In practice, many duplicates
differ by trivial edits—whitespace, punctuation, copyright headers, etc. MinHash
with LSH detects these *near*-duplicates efficiently.

Algorithm
=========

1. **Text Normalization** (improves recall, following Penedo et al., 2023):
   - Lowercase all text
   - Remove punctuation
   - Normalize whitespace (collapse runs of whitespace to a single space)
   - Remove accents (e.g. é → e) via NFD unicode normalization + stripping
     combining characters (Unicode category "Mn")

2. **Word N-gram Shingling**:
   Each document is converted to the set of its word-level n-grams. For example,
   with n=5 the document "the quick brown fox jumps over the lazy dog" produces
   the shingles {"the quick brown fox jumps", "quick brown fox jumps over", …}.

3. **MinHash Signature Generation**:
   For each document we compute a fixed-length signature vector of `num_hashes`
   values. Each slot uses an independent hash function (implemented as
   h(x) = (a*x + b) mod p, with random a, b and a large prime p). The minimum
   hash over all shingles becomes that slot's value. Two documents' Jaccard
   similarity is approximated by the fraction of slots where their signatures
   agree.

   Hashing details:
   - Each shingle is first mapped to a 64-bit integer via a fast, non-
     cryptographic hash (Python's built-in hash with a fixed seed via hashlib
     SHA-1 truncated to 64 bits for determinism).
   - We then apply `num_hashes` independent linear hash functions of the form
         h_i(x) = (a_i * x + b_i) mod MERSENNE_PRIME
     where MERSENNE_PRIME = 2^61 - 1 (a Mersenne prime that fits in 64 bits),
     a_i and b_i are randomly sampled integers in [1, MERSENNE_PRIME) and
     [0, MERSENNE_PRIME) respectively.
   - For each hash function i, the minimum hash value across all shingles in the
     document is stored as signature[i].

4. **Locality-Sensitive Hashing (LSH)**:
   The signature is divided into `num_bands` bands, each containing
   `num_hashes // num_bands` rows. Two documents that share an identical band
   are placed into the same bucket and become *candidate duplicates*.

   The probability that two documents with true Jaccard similarity s are
   identified as candidates is approximately  1 - (1 - s^r)^b  where
   r = rows per band and b = num_bands. This gives an S-curve threshold that
   can be tuned via the band/row ratio.

5. **Jaccard Verification**:
   For each pair of candidate duplicates, we compute the exact word-n-gram
   Jaccard similarity on the *normalized* text. Pairs exceeding the threshold
   are true duplicates.

6. **Duplicate Removal via Union-Find**:
   All verified duplicate pairs are merged into connected components using a
   Union-Find (disjoint set) data structure. From each component, we keep the
   document that appeared first in the input order (lowest file index, preserving
   the original filename order). All other members of the component are removed.

7. **Output**:
   Non-duplicate documents are written to the output directory, preserving their
   original filenames. Removed documents are simply not written.

Memory Efficiency for Large Corpora
====================================
- Documents are identified by integer index, not by content.
- MinHash signatures are stored as compact NumPy arrays (or plain lists if NumPy
  is unavailable).
- LSH buckets use tuples of hash values as keys (fixed size per band).
- The shingle sets for Jaccard verification are built on-the-fly for candidate
  pairs only, so at most two shingle sets are in memory at a time.
- Text normalization is done in a streaming fashion per document.

Usage:
    from cs336_data.minhash_lsh import minhash_lsh_deduplication

    minhash_lsh_deduplication(
        input_files=["file1.txt", "file2.txt", ...],
        num_hashes=100,
        num_bands=10,
        ngrams=5,
        jaccard_threshold=0.8,
        output_directory="/path/to/output",
    )
"""

import gc
import hashlib
import os
import pickle
import re
import shutil
import struct
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from tqdm import tqdm
from xopen import xopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Mersenne prime 2^61 - 1, fits in a 64-bit integer.  Used as the modulus for
# the family of linear hash functions that produce the MinHash signature.
MERSENNE_PRIME: int = (1 << 61) - 1

# Upper bound for the random coefficients a and b in the linear hash family.
MAX_HASH: int = (1 << 32) - 1


# ---------------------------------------------------------------------------
# Text normalisation helpers (following Penedo et al., 2023)
# ---------------------------------------------------------------------------
# Pre-compiled regex that matches any punctuation character (Unicode category P)
_PUNCT_RE = re.compile(r"[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]"
                        r"|[^\w\s]", re.UNICODE)
# Pre-compiled regex for collapsing whitespace
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for MinHash / Jaccard comparison.

    Steps (order matters):
    1. NFD unicode normalization
    2. Strip combining characters (accents) — Unicode category "Mn"
    3. Lowercase
    4. Remove punctuation
    5. Collapse whitespace to single spaces and strip leading/trailing
    """
    # 1. NFD decomposition
    text = unicodedata.normalize("NFD", text)
    # 2. Remove combining marks (accents)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # 3. Lowercase
    text = text.lower()
    # 4. Remove punctuation
    text = _PUNCT_RE.sub("", text)
    # 5. Collapse whitespace
    text = _SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

def word_ngrams(text: str, n: int) -> set[str]:
    """Return the set of word-level n-grams (shingles) for *text*.

    Args:
        text: Already-normalised text string.
        n: N-gram length in words.

    Returns:
        Set of shingle strings, e.g. {"the quick brown fox jumps", ...}.
    """
    words = text.split()
    if len(words) < n:
        # If the document is shorter than the n-gram length, use the whole
        # document as a single shingle so it is still comparable.
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# Deterministic shingle → integer hashing
# ---------------------------------------------------------------------------

def shingle_to_int(shingle: str) -> int:
    """Map a shingle string to a deterministic 64-bit integer.

    We use SHA-1 (truncated to 8 bytes) instead of Python's built-in ``hash``
    because ``hash`` is randomised across processes (PYTHONHASHSEED).  SHA-1 is
    deterministic and fast enough for our purposes.
    """
    h = hashlib.sha1(shingle.encode("utf-8")).digest()[:8]
    return struct.unpack("<Q", h)[0]


# ---------------------------------------------------------------------------
# MinHash signature computation
# ---------------------------------------------------------------------------

def _generate_hash_params(num_hashes: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generate deterministic (a, b) pairs for the linear hash family.

    Each hash function is  h(x) = (a * x + b) mod MERSENNE_PRIME.
    We derive a_i, b_i from a seeded PRNG so results are reproducible.
    """
    import random
    rng = random.Random(seed)
    params: list[tuple[int, int]] = []
    for _ in range(num_hashes):
        a = rng.randint(1, MERSENNE_PRIME - 1)
        b = rng.randint(0, MERSENNE_PRIME - 1)
        params.append((a, b))
    return params


def compute_minhash_signature(
    shingle_set: set[str],
    hash_params: list[tuple[int, int]],
) -> list[int]:
    """Compute the MinHash signature for a set of shingles.

    Args:
        shingle_set: Set of shingle strings.
        hash_params: List of (a, b) tuples for the hash family.

    Returns:
        List of length ``len(hash_params)`` containing the minimum hash value
        for each hash function.
    """
    num_hashes = len(hash_params)

    if not shingle_set:
        # Empty document → signature of MAX values (will not match anything)
        return [MERSENNE_PRIME] * num_hashes

    # Convert shingles to integers once
    shingle_ints = [shingle_to_int(s) for s in shingle_set]

    signature = [MERSENNE_PRIME] * num_hashes
    for s_int in shingle_ints:
        for i, (a, b) in enumerate(hash_params):
            h = (a * s_int + b) % MERSENNE_PRIME
            if h < signature[i]:
                signature[i] = h
    return signature


# ---------------------------------------------------------------------------
# LSH banding
# ---------------------------------------------------------------------------

def lsh_buckets(
    signatures: list[list[int]],
    num_bands: int,
) -> list[set[int]]:
    """Identify candidate duplicate pairs via LSH banding.

    Args:
        signatures: List of MinHash signatures (one per document).
        num_bands: Number of bands to split the signature into.

    Returns:
        List of sets, each set being a group of document indices that hashed
        to the same bucket in at least one band.  (Returned as the merged
        candidate groups for convenience.)
    """
    if not signatures:
        return []

    num_hashes = len(signatures[0])
    rows_per_band = num_hashes // num_bands

    # band_index → {bucket_key → set of doc indices}
    buckets: dict[int, dict[tuple, set[int]]] = defaultdict(lambda: defaultdict(set))

    for doc_idx, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            band_key = tuple(sig[start:start + rows_per_band])
            buckets[band_idx][band_key].add(doc_idx)

    # Collect all candidate pairs (doc_i, doc_j) with i < j
    candidate_pairs: set[tuple[int, int]] = set()
    for band_buckets in buckets.values():
        for members in band_buckets.values():
            if len(members) > 1:
                members_list = sorted(members)
                for i in range(len(members_list)):
                    for j in range(i + 1, len(members_list)):
                        candidate_pairs.add((members_list[i], members_list[j]))

    return candidate_pairs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exact Jaccard similarity on word n-grams
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute the Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) for grouping duplicates
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Memory-Efficient Streaming Implementation for Large Datasets
# ---------------------------------------------------------------------------

def minhash_lsh_deduplication_streaming(
    input_files: list[Union[str, Path]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Union[str, Path],
    batch_size: int = 1000,
    cache_dir: str = None,
    verbose: bool = True,
) -> dict:
    """Memory-efficient MinHash deduplication for large datasets.
    
    This streaming version processes files in batches and uses disk caching
    to handle datasets larger than available RAM. Peak memory usage is roughly:
    batch_size × avg_file_size × 3 (for processing) + LSH bucket overhead.
    
    Args:
        input_files: Paths to input text files (one document per file).
        num_hashes: Number of hash functions for the MinHash signature.
                     Must be evenly divisible by ``num_bands``.
        num_bands: Number of bands for LSH.
        ngrams: Word n-gram length for shingling.
        jaccard_threshold: Pairs with Jaccard similarity ≥ this value are
                            considered duplicates.
        output_directory: Directory to write deduplicated files.
        batch_size: Number of files to process in each batch (tune for your RAM).
        cache_dir: Directory for temporary files (default: creates temp dir).
        verbose: Whether to show progress bars and status messages.
        
    Returns:
        A dict with summary statistics (kept, removed, total, etc.).
    """
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
    
    # Create temporary directory for caching
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
        
        # ------------------------------------------------------------------
        # Step 1: Process files in streaming batches to compute signatures
        # ------------------------------------------------------------------
        hash_params = _generate_hash_params(num_hashes)
        signatures_file = Path(cache_dir) / "signatures.pkl"
        
        if verbose:
            print("🔄 Computing MinHash signatures...")
            
        with open(signatures_file, 'wb') as sig_f:
            progress_bar = tqdm(range(0, num_docs, batch_size), desc="Signature batches") if verbose else range(0, num_docs, batch_size)
            
            for batch_start in progress_bar:
                batch_end = min(batch_start + batch_size, num_docs)
                batch_files = input_files[batch_start:batch_end]
                
                # Process current batch
                batch_signatures = []
                for i, fpath in enumerate(batch_files):
                    doc_idx = batch_start + i
                    
                    # Read and process single file
                    with xopen(fpath, "rt") as f:
                        content = f.read()
                    
                    normalized = normalize_text(content)
                    shingles = word_ngrams(normalized, ngrams)
                    signature = compute_minhash_signature(shingles, hash_params)
                    
                    batch_signatures.append((doc_idx, signature))
                    
                    # Clear variables to free memory
                    del content, normalized, shingles
                
                # Write batch signatures to disk
                pickle.dump(batch_signatures, sig_f)
                del batch_signatures
                gc.collect()  # Force garbage collection
        
        # ------------------------------------------------------------------
        # Step 2: LSH candidate detection (streaming signature loading)
        # ------------------------------------------------------------------
        if verbose:
            print("🔍 Finding LSH candidates...")
        
        def load_signatures_streaming() -> Iterator[Tuple[int, List[int]]]:
            """Generator that yields (doc_idx, signature) without storing all in memory."""
            with open(signatures_file, 'rb') as f:
                try:
                    while True:
                        batch = pickle.load(f)
                        for doc_idx, signature in batch:
                            yield doc_idx, signature
                except EOFError:
                    pass
        
        # Build LSH buckets incrementally
        buckets: dict[int, dict[tuple, set[int]]] = defaultdict(lambda: defaultdict(set))
        rows_per_band = num_hashes // num_bands
        
        progress_bar = tqdm(load_signatures_streaming(), desc="LSH bucketing", total=num_docs) if verbose else load_signatures_streaming()
        
        for doc_idx, signature in progress_bar:
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                band_key = tuple(signature[start:start + rows_per_band])
                buckets[band_idx][band_key].add(doc_idx)
        
        # Extract candidate pairs
        candidate_pairs: set[tuple[int, int]] = set()
        for band_buckets in buckets.values():
            for members in band_buckets.values():
                if len(members) > 1:
                    members_list = sorted(members)
                    for i in range(len(members_list)):
                        for j in range(i + 1, len(members_list)):
                            candidate_pairs.add((members_list[i], members_list[j]))
        
        if verbose:
            print(f"🎯 Found {len(candidate_pairs)} candidate pairs")
        del buckets  # Free LSH buckets
        gc.collect()
        
        # ------------------------------------------------------------------
        # Step 3: Jaccard verification (on-demand file reading)
        # ------------------------------------------------------------------
        if verbose:
            print("✅ Verifying candidates with Jaccard similarity...")
        
        def get_shingle_set_for_doc(doc_idx: int) -> set[str]:
            """Load and compute shingle set for a single document on-demand."""
            fpath = input_files[doc_idx]
            with xopen(fpath, "rt") as f:
                content = f.read()
            normalized = normalize_text(content)
            return word_ngrams(normalized, ngrams)
        
        uf = UnionFind(num_docs)
        verified_pairs = 0
        
        progress_bar = tqdm(candidate_pairs, desc="Jaccard verification") if verbose else candidate_pairs
        
        for i, j in progress_bar:
            # Load shingle sets on-demand (only 2 at a time)
            shingles_i = get_shingle_set_for_doc(i)
            shingles_j = get_shingle_set_for_doc(j)
            
            sim = jaccard_similarity(shingles_i, shingles_j)
            if sim >= jaccard_threshold:
                uf.union(i, j)
                verified_pairs += 1
            
            # Immediately free memory
            del shingles_i, shingles_j
        
        if verbose:
            print(f"🔗 Verified {verified_pairs} duplicate pairs")
        
        # ------------------------------------------------------------------
        # Step 4: Determine documents to keep
        # ------------------------------------------------------------------
        components: dict[int, list[int]] = defaultdict(list)
        for idx in range(num_docs):
            root = uf.find(idx)
            components[root].append(idx)
        
        docs_to_keep: set[int] = set()
        for members in components.values():
            docs_to_keep.add(min(members))
        
        # ------------------------------------------------------------------
        # Step 5: Write kept documents (streaming)
        # ------------------------------------------------------------------
        if verbose:
            print("💾 Writing deduplicated files...")
            
        progress_bar = tqdm(sorted(docs_to_keep), desc="Writing files") if verbose else sorted(docs_to_keep)
        
        for idx in progress_bar:
            # Read original content on-demand
            with xopen(input_files[idx], "rt") as f:
                content = f.read()
            
            out_path = output_directory / input_files[idx].name
            with xopen(out_path, "wt") as f:
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
        # Cleanup temporary directory if we created it
        if temp_dir_created and Path(cache_dir).exists():
            shutil.rmtree(cache_dir)


# ---------------------------------------------------------------------------
# Original Implementation (kept for compatibility)
# ---------------------------------------------------------------------------

def minhash_lsh_deduplication(
    input_files: list,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Union[str, Path],
    auto_streaming: bool = True,
    streaming_threshold: int = 5000,
    batch_size: int = 1000,
    cache_dir: str = None,
    verbose: bool = True,
) -> dict:
    """Perform fuzzy document deduplication with MinHash + LSH.
    
    Automatically chooses between in-memory and streaming implementation
    based on dataset size to optimize for memory usage.

    Args:
        input_files: Paths to input text files (one document per file).
        num_hashes: Number of hash functions for the MinHash signature.
                     Must be evenly divisible by ``num_bands``.
        num_bands: Number of bands for LSH.
        ngrams: Word n-gram length for shingling.
        jaccard_threshold: Pairs with Jaccard similarity ≥ this value are
                            considered duplicates.
        output_directory: Directory to write deduplicated files.
        auto_streaming: If True, automatically use streaming for large datasets.
        streaming_threshold: Use streaming if num_files > this threshold.
        batch_size: Batch size for streaming mode.
        cache_dir: Cache directory for streaming mode.
        verbose: Whether to show progress information.

    Returns:
        A dict with summary statistics (kept, removed, total, etc.).
    """
    num_files = len(input_files)
    
    # Automatically choose implementation based on dataset size
    if auto_streaming and num_files > streaming_threshold:
        if verbose:
            print(f"🚀 Large dataset detected ({num_files} files > {streaming_threshold})")
            print(f"🔄 Using memory-efficient streaming implementation")
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
    else:
        if verbose and auto_streaming:
            print(f"📝 Small dataset ({num_files} files ≤ {streaming_threshold})")
            print(f"⚡ Using in-memory implementation for speed")
        return _minhash_lsh_deduplication_inmemory(
            input_files=input_files,
            num_hashes=num_hashes,
            num_bands=num_bands,
            ngrams=ngrams,
            jaccard_threshold=jaccard_threshold,
            output_directory=output_directory,
        )


def _minhash_lsh_deduplication_inmemory(
    input_files: list[Union[str, Path]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Union[str, Path],
) -> dict:
    """Original in-memory implementation (faster for small datasets).

    Args:
        input_files: Paths to input text files (one document per file).
        num_hashes: Number of hash functions for the MinHash signature.
                     Must be evenly divisible by ``num_bands``.
        num_bands: Number of bands for LSH.
        ngrams: Word n-gram length for shingling.
        jaccard_threshold: Pairs with Jaccard similarity ≥ this value are
                            considered duplicates.
        output_directory: Directory to write deduplicated files.

    Returns:
        A dict with summary statistics (kept, removed, total, etc.).
    """
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

    # ------------------------------------------------------------------
    # Step 1: Read documents and compute normalized text
    # ------------------------------------------------------------------
    raw_contents: list[str] = []
    normalized_texts: list[str] = []

    for fpath in input_files:
        with xopen(fpath, "rt") as f:
            content = f.read()
        raw_contents.append(content)
        normalized_texts.append(normalize_text(content))

    # ------------------------------------------------------------------
    # Step 2: Compute shingle sets and MinHash signatures
    # ------------------------------------------------------------------
    hash_params = _generate_hash_params(num_hashes)
    signatures: list[list[int]] = []
    # Cache shingle sets for later Jaccard verification
    shingle_sets: list[set[str]] = []

    for norm_text in normalized_texts:
        shingles = word_ngrams(norm_text, ngrams)
        shingle_sets.append(shingles)
        sig = compute_minhash_signature(shingles, hash_params)
        signatures.append(sig)

    # ------------------------------------------------------------------
    # Step 3: LSH to find candidate pairs
    # ------------------------------------------------------------------
    candidate_pairs = lsh_buckets(signatures, num_bands)

    # ------------------------------------------------------------------
    # Step 4: Verify candidates with exact Jaccard similarity
    # ------------------------------------------------------------------
    uf = UnionFind(num_docs)

    for i, j in candidate_pairs:
        sim = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        if sim >= jaccard_threshold:
            uf.union(i, j)

    # ------------------------------------------------------------------
    # Step 5: Determine which documents to keep
    #
    # For each connected component keep the document with the smallest
    # index (i.e. the one that appeared first in the input list).
    # ------------------------------------------------------------------
    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(num_docs):
        root = uf.find(idx)
        components[root].append(idx)

    docs_to_keep: set[int] = set()
    for members in components.values():
        # Keep the member with the smallest index
        docs_to_keep.add(min(members))

    # ------------------------------------------------------------------
    # Step 6: Write kept documents to the output directory
    # ------------------------------------------------------------------
    for idx in sorted(docs_to_keep):
        out_path = output_directory / input_files[idx].name
        with xopen(out_path, "wt") as f:
            f.write(raw_contents[idx])

    removed = num_docs - len(docs_to_keep)
    return {
        "total": num_docs,
        "kept": len(docs_to_keep),
        "removed": removed,
        "candidate_pairs_checked": len(candidate_pairs),
    }


# ---------------------------------------------------------------------------
# Convenience Functions for Different Use Cases
# ---------------------------------------------------------------------------

def deduplicate_large_dataset(
    input_files: list[Union[str, Path]],
    output_directory: Union[str, Path],
    num_hashes: int = 256,
    num_bands: int = 32,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
    batch_size: int = 500,
    cache_dir: str = None,
) -> dict:
    """Convenience function for deduplicating large datasets with memory constraints.
    
    Optimized defaults for large-scale deduplication:
    - Higher num_hashes (256) for better accuracy
    - More bands (32) for better candidate detection  
    - Smaller batch_size (500) for lower memory usage
    - Always uses streaming implementation
    
    Args:
        input_files: Paths to input text files
        output_directory: Where to write deduplicated files
        num_hashes: Number of hash functions (default: 256)
        num_bands: Number of LSH bands (default: 32) 
        ngrams: Word n-gram size (default: 5)
        jaccard_threshold: Similarity threshold (default: 0.8)
        batch_size: Files per batch (default: 500 for low memory)
        cache_dir: Temp cache directory (default: auto-generated)
        
    Returns:
        Dictionary with deduplication statistics
    """
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


def deduplicate_small_dataset(
    input_files: list[Union[str, Path]],
    output_directory: Union[str, Path],
    num_hashes: int = 100,
    num_bands: int = 10,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
) -> dict:
    """Convenience function for fast deduplication of small datasets.
    
    Uses in-memory implementation for maximum speed on datasets that fit in RAM.
    
    Args:
        input_files: Paths to input text files
        output_directory: Where to write deduplicated files
        num_hashes: Number of hash functions (default: 100)
        num_bands: Number of LSH bands (default: 10)
        ngrams: Word n-gram size (default: 5) 
        jaccard_threshold: Similarity threshold (default: 0.8)
        
    Returns:
        Dictionary with deduplication statistics
    """
    return _minhash_lsh_deduplication_inmemory(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
    )
