"""Text normalization and shingling helpers."""

import re
import unicodedata
from typing import Set

# Regex matching punctuation + non-word/non-space chars.
_PUNCT_RE = re.compile(
    r"[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]|[^\w\s]",
    re.UNICODE,
)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for MinHash/Jaccard comparison."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def word_ngrams(text: str, n: int) -> Set[str]:
    """Return set of word-level n-gram shingles."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
