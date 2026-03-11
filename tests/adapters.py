# Import directly from individual modules to avoid __init__.py issues
from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language
from cs336_data.pii_masking import mask_emails, mask_phone_numbers, mask_ip_addresses
from cs336_data.quality_filter import gopher_quality_filter
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.quality_classifier_fasttext import get_all_predictions
from cs336_data.exact_line_dedup import exact_line_deduplication
from cs336_data.minhash_lsh import minhash_lsh_deduplication


# Try to import quality classifier (using Dolma3 FastText model)
# try:
#     # from cs336_data.quality_classifier_dolma import classify_quality_with_category
#     _quality_classifier_available = True
# except (ImportError, FileNotFoundError, OSError):
#     _quality_classifier_available = False
#     classify_quality_with_category = None


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract plain text from HTML bytes.
    
    Args:
        html_bytes: Raw HTML content as bytes
        
    Returns:
        Extracted plain text as string, or None if extraction fails
    """
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[str, float]:
    """
    Identify the language of text.
    
    Args:
        text: Unicode string to identify
        
    Returns:
        Tuple of (language_code, confidence_score)
    """
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    """
    Mask email addresses in text.
    
    Args:
        text: Input text containing email addresses
        
    Returns:
        Tuple of (masked_text, number_of_emails_masked)
    """
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Mask phone numbers in text.
    
    Args:
        text: Input text containing phone numbers
        
    Returns:
        Tuple of (masked_text, number_of_phones_masked)
    """
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    """
    Mask IP addresses in text.
    
    Args:
        text: Input text containing IP addresses
        
    Returns:
        Tuple of (masked_text, number_of_ips_masked)
    """
    return mask_ip_addresses(text)


def run_gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher quality filtering rules.
    
    Args:
        text: Input text to filter
        
    Returns:
        True if text passes all quality checks, False otherwise
    """
    label,details = gopher_quality_filter(text)
    return label


def run_classify_nsfw(text: str) -> tuple[str, float]:
    """
    Classify text as NSFW or not.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence_score)
        - label: "nsfw" or "non-nsfw"
    """
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[str, float]:
    """
    Classify text as toxic or not.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence_score)
        - label: "toxic" or "non-toxic"
    """
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[str, float]:
    """
    Classify text quality using FastText model.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (category, confidence_score)
        - category: "wiki" (high quality) or "cc" (low quality)
    """
    # if not _quality_classifier_available:
    #     raise NotImplementedError(
    #         "FastText quality classifier not available. "
    #         "Please train a model first using train_quality_classifier.py"
    #     )
    output = get_all_predictions(text)
    if output["label"] == "positive":
        category = "wiki"
    else:
        category = "cc"
    return category, output["confidence"]

# Deduplication adapters
def run_exact_line_deduplication(input_files: list, output_directory) -> dict:
    return exact_line_deduplication(input_files=input_files, output_directory=output_directory)


def run_minhash_deduplication(
    input_files: list,
    output_directory,
    num_hashes: int = 100,
    num_bands: int = 10,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
) -> dict:
    """Remove near-duplicate documents using MinHash + LSH."""
    return minhash_lsh_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
    )