from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language, get_language_name
from cs336_data.pii_masking import mask_emails, mask_phone_numbers, mask_ip_addresses, mask_all_pii
from cs336_data.quality_filter import gopher_quality_filter,gopher_quality_filter_batch
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech, classify_content

# Optional batch imports (may not exist in older versions)
try:
    from cs336_data.langid import identify_language_batch
except ImportError:
    identify_language_batch = None

try:
    from cs336_data.toxicity import classify_nsfw_batch, classify_toxic_speech_batch, classify_content_batch
except ImportError:
    classify_nsfw_batch = None
    classify_toxic_speech_batch = None
    classify_content_batch = None

# Try to import quality classifier (using Dolma3 FastText model from Allen AI)
try:
    from cs336_data.quality_classifier_dolma import (
        classify_quality,
        classify_quality_batch,
        classify_quality_with_category,
        is_quality_acceptable,
        filter_by_quality,
    )
except (ImportError, FileNotFoundError):
    # Model not available, provide stub functions
    def classify_quality(text):
        raise NotImplementedError("Dolma3 FastText quality model not available.")
    
    def classify_quality_batch(texts):
        raise NotImplementedError("Dolma3 FastText quality model not available.")
    
    def classify_quality_with_category(text):
        raise NotImplementedError("Dolma3 FastText quality model not available.")
    
    def is_quality_acceptable(text):
        raise NotImplementedError("Dolma3 FastText quality model not available.")
    
    def filter_by_quality(text):
        raise NotImplementedError("Dolma3 FastText quality model not available.")

__all__ = [
    # Extract
    "extract_text_from_html_bytes",
    # Language ID
    "identify_language",
    # "identify_language_batch", 
    "get_language_name",
    # PII
    "mask_emails",
    "mask_phone_numbers",
    "mask_ip_addresses",
    "mask_all_pii",
    # Quality filter
    "gopher_quality_filter",
    "gopher_quality_filter_batch",
    # Toxicity
    "classify_nsfw",
    "classify_toxic_speech",
    "classify_content",
    # Quality classifier
    "classify_quality",
    "classify_quality_batch",
    "classify_quality_with_category",
    "is_quality_acceptable",
    "filter_by_quality",
]