"""
Gopher Quality Filtering Module

Implements quality filtering rules based on the Gopher paper to remove
low-quality documents from the dataset.

Rules:
1. Word count: 50-100,000 words
2. Mean word length: 3-10 characters
3. Symbol-to-word ratio: <= 0.1 for hash (#) and ellipsis (...)
4. Bullet point lines: <= 90% starting with bullet
5. Ellipsis lines: <= 30% ending with ellipsis
6. Alphabetic words: >= 80% of words contain at least one alphabetic character
7. Stop words: At least 2 of [the, be, to, of, and, that, have, with]
"""

import re
from typing import Tuple

# NLTK imports for better tokenization
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize


# Stop words required for English text detection
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}

# Bullet point patterns
BULLET_PATTERNS = re.compile(r"^[\s]*[-•●○▪▸►◦*→⁃‣⦿⦾]\s*", re.MULTILINE)

# Ellipsis pattern at end of line
ELLIPSIS_END_PATTERN = re.compile(r"\.{3}\s*$|…\s*$", re.MULTILINE)


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text using NLTK's word_tokenize for better handling of
    punctuation, contractions, and special characters.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens/words
    """
    try:
        tokens = word_tokenize(text)
        return tokens
    except Exception:
        # Fallback to simple split if NLTK fails
        return text.split()


def get_words_only(tokens: list[str]) -> list[str]:
    """
    Filter tokens to get only actual words (containing at least one letter).
    Excludes pure punctuation tokens.
    
    Args:
        tokens: List of tokens from tokenize_text
        
    Returns:
        List of word tokens only
    """
    return [t for t in tokens if any(c.isalpha() for c in t)]


def count_words(text: str) -> list[str]:
    """
    Split text into words using NLTK tokenizer.
    Returns only actual words (not pure punctuation).
    """
    tokens = tokenize_text(text)
    return get_words_only(tokens)


def get_word_count(text: str) -> int:
    """Get the number of words in the text."""
    return len(count_words(text))


def get_mean_word_length(text: str) -> float:
    """Calculate mean word length in characters."""
    words = count_words(text)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def get_symbol_to_word_ratio(text: str, symbol: str) -> float:
    """
    Calculate the ratio of a symbol to words.
    
    Args:
        text: Input text
        symbol: Symbol to count (e.g., '#' or '...')
    
    Returns:
        Ratio of symbol occurrences to word count
    """
    words = count_words(text)
    if not words:
        return 0.0
    
    if symbol == "...":
        # Count ellipsis (both ... and …)
        count = text.count("...") + text.count("…")
    else:
        count = text.count(symbol)
    
    return count / len(words)


def get_bullet_line_ratio(text: str) -> float:
    """
    Calculate the ratio of lines starting with a bullet point.
    
    Returns:
        Ratio of bullet lines to total lines
    """
    lines = text.split("\n")
    if not lines:
        return 0.0
    
    # Filter out empty lines for this calculation
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return 0.0
    
    bullet_count = 0
    for line in non_empty_lines:
        if BULLET_PATTERNS.match(line):
            bullet_count += 1
    
    return bullet_count / len(non_empty_lines)


def get_ellipsis_line_ratio(text: str) -> float:
    """
    Calculate the ratio of lines ending with an ellipsis.
    
    Returns:
        Ratio of ellipsis-ending lines to total lines
    """
    lines = text.split("\n")
    if not lines:
        return 0.0
    
    # Filter out empty lines for this calculation
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return 0.0
    
    ellipsis_count = 0
    for line in non_empty_lines:
        if line.strip().endswith("...") or line.strip().endswith("…"):
            ellipsis_count += 1
    
    return ellipsis_count / len(non_empty_lines)


def get_alphabetic_word_ratio(text: str) -> float:
    """
    Calculate the ratio of words containing at least one alphabetic character.
    Uses all tokens (including punctuation) for this calculation.
    
    Returns:
        Ratio of alphabetic words to total tokens
    """
    # Get all tokens (including punctuation) for this ratio
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0
    
    alpha_count = sum(1 for token in tokens if any(c.isalpha() for c in token))
    return alpha_count / len(tokens)


def count_stop_words(text: str) -> int:
    """
    Count how many of the required stop words appear in the text.
    Uses NLTK tokenization for better word boundary detection.
    
    Returns:
        Number of unique stop words found (0-8)
    """
    # Use NLTK tokenizer and convert to lowercase
    tokens = tokenize_text(text.lower())
    words_set = set(tokens)
    
    # Count unique stop words present
    return len(STOP_WORDS.intersection(words_set))


def gopher_quality_filter(text: str) -> Tuple[bool, dict]:
    """
    Apply Gopher quality filtering rules to a text.
    
    Args:
        text: Input text to filter
        
    Returns:
        Tuple of (passes_filter: bool, details: dict)
        - passes_filter: True if text passes all quality checks
        - details: Dictionary with all computed metrics and which checks failed
    """
    if not text or not isinstance(text, str):
        return False, {
            "passed": False,
            "reason": "empty_or_invalid",
            "word_count": 0,
            "mean_word_length": 0.0,
            "hash_ratio": 0.0,
            "ellipsis_ratio": 0.0,
            "bullet_line_ratio": 0.0,
            "ellipsis_line_ratio": 0.0,
            "alphabetic_word_ratio": 0.0,
            "stop_word_count": 0,
            "failed_checks": ["empty_or_invalid"],
        }
    
    # Compute all metrics
    word_count = get_word_count(text)
    mean_word_length = get_mean_word_length(text)
    hash_ratio = get_symbol_to_word_ratio(text, "#")
    ellipsis_ratio = get_symbol_to_word_ratio(text, "...")
    bullet_line_ratio = get_bullet_line_ratio(text)
    ellipsis_line_ratio = get_ellipsis_line_ratio(text)
    alphabetic_word_ratio = get_alphabetic_word_ratio(text)
    stop_word_count = count_stop_words(text)
    
    # Track failed checks
    failed_checks = []
    
    # Rule 1: Word count between 50 and 100,000
    if word_count < 50 or word_count > 100000:
        # print("Word count check failed:", word_count,text[:20])
        failed_checks.append("word_count")
    
    # Rule 2: Mean word length between 3 and 10
    if mean_word_length < 3 or mean_word_length > 10:
        failed_checks.append("mean_word_length")
    
    # Rule 3: Hash symbol ratio <= 0.1
    if hash_ratio > 0.1:
        failed_checks.append("hash_ratio")
    
    # Rule 4: Ellipsis symbol ratio <= 0.1
    if ellipsis_ratio > 0.1:
        failed_checks.append("ellipsis_ratio")
    
    # Rule 5: Bullet point lines <= 90%
    if bullet_line_ratio > 0.9:
        failed_checks.append("bullet_line_ratio")
    
    # Rule 6: Ellipsis ending lines <= 30%
    if ellipsis_line_ratio > 0.3:
        failed_checks.append("ellipsis_line_ratio")
    
    # Rule 7: Alphabetic words >= 80%
    if alphabetic_word_ratio < 0.8:
        failed_checks.append("alphabetic_word_ratio")
    
    # Rule 8: At least 2 stop words
    if stop_word_count < 2:
        failed_checks.append("stop_word_count")
    
    passes_filter = len(failed_checks) == 0
    
    details = {
        "passed": passes_filter,
        "reason": "passed" if passes_filter else failed_checks[0],
        "word_count": word_count,
        "mean_word_length": round(mean_word_length, 2),
        "hash_ratio": round(hash_ratio, 4),
        "ellipsis_ratio": round(ellipsis_ratio, 4),
        "bullet_line_ratio": round(bullet_line_ratio, 4),
        "ellipsis_line_ratio": round(ellipsis_line_ratio, 4),
        "alphabetic_word_ratio": round(alphabetic_word_ratio, 4),
        "stop_word_count": stop_word_count,
        "failed_checks": failed_checks,
    }
    
    return passes_filter,details


def gopher_quality_filter_batch(texts: list[str]) -> list[Tuple[bool, dict]]:
    """
    Apply Gopher quality filtering to a batch of texts.
    
    Args:
        texts: List of input texts to filter
        
    Returns:
        List of tuples (passes_filter, details) for each text
    """
    return [gopher_quality_filter(text) for text in texts]


# Summary statistics for filtering
def get_filter_stats(results: list[Tuple[bool, dict]]) -> dict:
    """
    Compute summary statistics from filtering results.
    
    Args:
        results: List of (passes_filter, details) tuples
        
    Returns:
        Dictionary with aggregated statistics
    """
    total = len(results)
    passed = sum(1 for passed, _ in results if passed)
    failed = total - passed
    
    # Count failures by reason
    failure_reasons = {
        "word_count": 0,
        "mean_word_length": 0,
        "hash_ratio": 0,
        "ellipsis_ratio": 0,
        "bullet_line_ratio": 0,
        "ellipsis_line_ratio": 0,
        "alphabetic_word_ratio": 0,
        "stop_word_count": 0,
        "empty_or_invalid": 0,
    }
    
    for _, details in results:
        for check in details.get("failed_checks", []):
            if check in failure_reasons:
                failure_reasons[check] += 1
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 2) if total > 0 else 0.0,
        "failure_reasons": failure_reasons,
    }
