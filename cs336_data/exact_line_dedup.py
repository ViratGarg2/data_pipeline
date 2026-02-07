"""
Exact Line Deduplication Module

This module implements memory-efficient exact line deduplication across multiple files.
It uses cryptographic hashing (SHA-256) to reduce memory footprint while maintaining
accurate duplicate detection.

Hashing Method Details:
=======================

We use SHA-256 (Secure Hash Algorithm 256-bit) for hashing lines for several reasons:

1. **Fixed Output Size**: SHA-256 produces a 256-bit (32-byte) hash regardless of input
   length. This means a line with 10 characters and a line with 10,000 characters both
   produce the same size hash, dramatically reducing memory for long lines.

2. **Collision Resistance**: SHA-256 is cryptographically secure, meaning the probability
   of two different lines producing the same hash (a collision) is astronomically low
   (approximately 1 in 2^128 for a birthday attack). For practical purposes, this means
   we can treat hash equality as line equality.

3. **Deterministic**: The same input always produces the same hash, which is essential
   for accurate duplicate detection across multiple passes over the data.

4. **Memory Efficiency Example**:
   - Without hashing: Storing 1 million unique lines averaging 100 bytes each = ~100 MB
   - With SHA-256: Storing 1 million 32-byte hashes = ~32 MB
   - For longer lines (e.g., 1000 bytes average), savings are even more dramatic

Alternative Hashing Approaches Considered:
- MD5 (128-bit): Faster but has known collision vulnerabilities
- xxHash/MurmurHash: Even faster but not cryptographically secure, higher collision risk
- SHA-1 (160-bit): Deprecated due to collision vulnerabilities
- We chose SHA-256 as the best balance of security and performance for deduplication

Implementation Strategy:
========================
1. **Pass 1 - Counting**: Read all files and count the frequency of each line's hash
2. **Pass 2 - Filtering**: Rewrite each file, keeping only lines whose hash has count == 1

This two-pass approach ensures that:
- We identify ALL duplicates across the entire corpus before removing any
- Each file is processed independently in the second pass (parallelizable)
- Memory usage is bounded by the number of unique lines, not their lengths

Usage:
    from cs336_data.exact_line_dedup import exact_line_deduplication
    
    exact_line_deduplication(
        input_files=["file1.txt", "file2.txt"],
        output_directory="/path/to/output"
    )
"""

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Union

from xopen import xopen


def hash_line(line: str) -> bytes:
    return hashlib.sha256(line.encode('utf-8')).digest()


def count_line_frequencies(input_files: list[Union[str, Path]]) -> Counter:
    line_counts: Counter = Counter()
    
    for filepath in input_files:
        filepath = Path(filepath)
        
        # xopen automatically handles gzip, bz2, xz compression based on extension
        with xopen(filepath) as f:
            for line in f:
                # Strip the trailing newline for consistent hashing
                # (files may have different line ending conventions)
                line_stripped = line.rstrip('\n\r')
                line_hash = hash_line(line_stripped)
                line_counts[line_hash] += 1
    
    return line_counts


def exact_line_deduplication(
    input_files: list[Union[str, Path]],
    output_directory: Union[str, Path],
) -> dict:
    """
    Perform exact line deduplication across multiple files.
    
    This function removes all lines that appear more than once across the entire
    corpus of input files. A line is considered a "duplicate" if it appears in
    ANY file more than once (including the same file or different files).
    
    Algorithm:
    1. **Pass 1 (Counting)**: Read all files and count the frequency of each
       line using SHA-256 hashes as keys. This reduces memory usage for long lines.
    
    2. **Pass 2 (Filtering)**: For each input file, rewrite it to the output
       directory, keeping only lines whose hash has a count of exactly 1.
    
    Hashing Details:
    - We use SHA-256 for memory-efficient duplicate detection
    - SHA-256 produces a 256-bit (32-byte) hash for any input length
    - Collision probability is negligible (< 1 in 2^128 for birthday attack)
    - See module docstring for detailed rationale
    
    Args:
        input_files: List of paths to input files. Supports plain text files
            and compressed files (gzip, bz2, xz) via xopen. Files can have
            any extension; compression is detected automatically.
        output_directory: Directory to write deduplicated files. Each input
            file will be written to this directory with the same basename.
            The directory will be created if it doesn't exist.
    
    Returns:
        Dictionary with statistics about the deduplication:
        - 'total_lines': Total lines read across all files
        - 'unique_lines': Lines that appeared exactly once
        - 'duplicate_lines': Lines that appeared more than once
        - 'lines_removed': Total line occurrences removed
    
    Example:
        >>> exact_line_deduplication(
        ...     input_files=['a/doc1.txt', 'a/doc2.txt'],
        ...     output_directory='b/'
        ... )
        {'total_lines': 100, 'unique_lines': 80, 'duplicate_lines': 10, 'lines_removed': 20}
        
        # This creates b/doc1.txt and b/doc2.txt with duplicates removed
    
    Note:
        - Empty files are valid output (if all lines were duplicates)
        - Line order within each file is preserved
        - Trailing newlines are preserved in output files
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_lines': 0,
        'unique_lines': 0,
        'duplicate_lines': 0,
        'lines_removed': 0,
    }
    
    # Pass 1: Count all line frequencies using hashes
    line_counts = count_line_frequencies(input_files)
    
    # Calculate statistics from the count
    for hash_val, count in line_counts.items():
        stats['total_lines'] += count
        if count == 1:
            stats['unique_lines'] += 1
        else:
            stats['duplicate_lines'] += 1
            stats['lines_removed'] += count  # All occurrences will be removed
    
    # Pass 2: Rewrite each file, keeping only unique lines
    for filepath in input_files:
        filepath = Path(filepath)
        output_path = output_directory / filepath.name
        
        with xopen(filepath) as f_in:
            with xopen(output_path) as f_out:
                for line in f_in:
                    line_stripped = line.rstrip('\n\r')
                    line_hash = hash_line(line_stripped)
                    
                    # Only keep lines that appear exactly once in the corpus
                    if line_counts[line_hash] == 1:
                        # Write the line with a newline (preserving original content)
                        f_out.write(line_stripped + '\n')
    
    return stats

