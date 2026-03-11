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

# ── Fast regex-based tokenizer ──────────────────────────────
# Matches sequences of alphanumerics/underscores OR single non-whitespace chars.
# This is ~20-50× faster than nltk.word_tokenize while giving comparable
# word-level tokens for the metrics we compute.
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+(?:\.[0-9]+)*|\S")
_ALPHA_RE = re.compile(r"[A-Za-z]")

# Stop words required for English text detection
STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})

# Bullet point pattern (matches start of a line)
_BULLET_RE = re.compile(r"[-•●○▪▸►◦*→⁃‣⦿⦾]")

# ── Single-pass helper ──────────────────────────────────────

def _compute_all_metrics(text: str) -> dict:
    """
    Compute every Gopher quality metric in a **single pass** over the text.

    Returns a dict with keys:
        word_count, mean_word_length, hash_ratio, ellipsis_ratio,
        bullet_line_ratio, ellipsis_line_ratio, alphabetic_word_ratio,
        stop_word_count
    """

    # ── tokenize once (fast regex) ──
    tokens = _WORD_RE.findall(text)
    num_tokens = len(tokens)

    # ── word-level stats (one loop over tokens) ──
    num_words = 0           # tokens with ≥1 alpha char
    total_word_len = 0      # sum of char lengths of words
    num_alpha_tokens = 0    # tokens with ≥1 alpha char (same as num_words)
    stop_words_found = set()

    lower_text_tokens_set = None  # deferred

    for tok in tokens:
        has_alpha = _ALPHA_RE.search(tok) is not None
        if has_alpha:
            num_words += 1
            total_word_len += len(tok)
            num_alpha_tokens += 1
            tok_lower = tok.lower()
            if tok_lower in STOP_WORDS:
                stop_words_found.add(tok_lower)

    mean_word_length = (total_word_len / num_words) if num_words > 0 else 0.0

    # Alphabetic word ratio uses *all* tokens in the denominator
    alphabetic_word_ratio = (num_alpha_tokens / num_tokens) if num_tokens > 0 else 0.0

    # ── symbol counts (simple str.count — very fast in CPython) ──
    hash_count = text.count("#")
    ellipsis_count = text.count("...") + text.count("…")
    hash_ratio = (hash_count / num_words) if num_words > 0 else 0.0
    ellipsis_ratio = (ellipsis_count / num_words) if num_words > 0 else 0.0

    # ── line-level stats (single split, single loop) ──
    lines = text.split("\n")
    num_non_empty = 0
    bullet_count = 0
    ellipsis_line_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        num_non_empty += 1

        # Bullet check: first non-whitespace char is a bullet symbol
        if _BULLET_RE.match(stripped):
            bullet_count += 1

        # Ellipsis-ending check
        if stripped.endswith("...") or stripped.endswith("…"):
            ellipsis_line_count += 1

    bullet_line_ratio = (bullet_count / num_non_empty) if num_non_empty > 0 else 0.0
    ellipsis_line_ratio = (ellipsis_line_count / num_non_empty) if num_non_empty > 0 else 0.0

    return {
        "word_count": num_words,
        "mean_word_length": mean_word_length,
        "hash_ratio": hash_ratio,
        "ellipsis_ratio": ellipsis_ratio,
        "bullet_line_ratio": bullet_line_ratio,
        "ellipsis_line_ratio": ellipsis_line_ratio,
        "alphabetic_word_ratio": alphabetic_word_ratio,
        "stop_word_count": len(stop_words_found),
    }


# ── Public API (unchanged signatures) ──────────────────────

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

    # Compute ALL metrics in one pass
    m = _compute_all_metrics(text)
    
    # Track failed checks
    failed_checks = []
    
    # Rule 1: Word count between 50 and 100,000
    if m["word_count"] < 50 or m["word_count"] > 100000:
        failed_checks.append("word_count")
    
    # Rule 2: Mean word length between 3 and 10
    if m["mean_word_length"] < 3 or m["mean_word_length"] > 10:
        failed_checks.append("mean_word_length")
    
    # Rule 3: Hash symbol ratio <= 0.1
    if m["hash_ratio"] > 0.1:
        failed_checks.append("hash_ratio")
    
    # Rule 4: Ellipsis symbol ratio <= 0.1
    if m["ellipsis_ratio"] > 0.1:
        failed_checks.append("ellipsis_ratio")
    
    # Rule 5: Bullet point lines <= 90%
    if m["bullet_line_ratio"] > 0.9:
        failed_checks.append("bullet_line_ratio")
    
    # Rule 6: Ellipsis ending lines <= 30%
    if m["ellipsis_line_ratio"] > 0.3:
        failed_checks.append("ellipsis_line_ratio")
    
    # Rule 7: Alphabetic words >= 80%
    if m["alphabetic_word_ratio"] < 0.8:
        failed_checks.append("alphabetic_word_ratio")
    
    # Rule 8: At least 2 stop words
    if m["stop_word_count"] < 2:
        failed_checks.append("stop_word_count")
    
    passes_filter = len(failed_checks) == 0
    
    details = {
        "passed": passes_filter,
        "reason": "passed" if passes_filter else failed_checks[0],
        "word_count": m["word_count"],
        "mean_word_length": round(m["mean_word_length"], 2),
        "hash_ratio": round(m["hash_ratio"], 4),
        "ellipsis_ratio": round(m["ellipsis_ratio"], 4),
        "bullet_line_ratio": round(m["bullet_line_ratio"], 4),
        "ellipsis_line_ratio": round(m["ellipsis_line_ratio"], 4),
        "alphabetic_word_ratio": round(m["alphabetic_word_ratio"], 4),
        "stop_word_count": m["stop_word_count"],
        "failed_checks": failed_checks,
    }
    
    return passes_filter, details


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
