"""
Quality Classifier Module using Allen AI Dolma3 FastText Model

Uses the allenai/dolma3-fasttext-quality-classifier model to classify text quality
into high or low quality categories.

This is a FastText-based model, making it much faster and lighter than transformer models.

Model: https://huggingface.co/allenai/dolma3-fasttext-quality-classifier

This module provides:
1. classify_quality() - Returns quality label and confidence score
2. is_quality_acceptable() - Returns True if quality is high
3. classify_quality_with_category() - Returns category compatible with test adapters
"""

import os
from pathlib import Path
from typing import Tuple

import fasttext

# Global model (lazy loaded)
_model = None

# Default model path - can be overridden
DEFAULT_MODEL_PATH = os.environ.get(
    "DOLMA_QUALITY_MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "dolma3_quality_classifier.bin")
)

# Alternative paths to check
MODEL_SEARCH_PATHS = [
    DEFAULT_MODEL_PATH,
    str(Path(__file__).parent.parent / "models" / "model.bin"),
    str(Path(__file__).parent.parent / "dolma3_quality_classifier.bin"),
    str(Path.home() / ".cache" / "dolma3_quality_classifier.bin"),
    "dolma3_quality_classifier.bin",
    "model.bin",
]


def download_model(output_path: str = None) -> str:
    """
    Download the Dolma3 quality classifier model from Hugging Face.
    
    Args:
        output_path: Where to save the model. If None, uses default path.
        
    Returns:
        Path to the downloaded model
    """
    if output_path is None:
        output_path = DEFAULT_MODEL_PATH
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Dolma3 quality classifier model...")
    print(f"This is a 4.3GB file and may take a while...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        model_path = hf_hub_download(
            repo_id="allenai/dolma3-fasttext-quality-classifier",
            filename="model.bin",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        
        # Rename if needed
        if model_path != output_path and Path(model_path).exists():
            Path(model_path).rename(output_path)
            model_path = output_path
        
        print(f"Model downloaded to: {model_path}")
        return model_path
        
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        print("\nAlternatively, download manually:")
        print("  wget https://huggingface.co/allenai/dolma3-fasttext-quality-classifier/resolve/main/model.bin")
        print(f"  mv model.bin {output_path}")
        raise


def find_model() -> str | None:
    """Find the model file in common locations."""
    for path in MODEL_SEARCH_PATHS:
        if Path(path).exists():
            return path
    return None


def _load_model(model_path: str = None):
    """Lazy load the FastText model."""
    global _model
    
    if _model is None:
        # Find model path
        if model_path is None:
            model_path = find_model()
        
        if model_path is None or not Path(model_path).exists():
            print("Dolma3 quality classifier model not found.")
            print(f"Searched in: {MODEL_SEARCH_PATHS}")
            print("\nTo download the model, run:")
            print("  python -c \"from cs336_data.quality_classifier_dolma import download_model; download_model()\"")
            print("\nOr download manually:")
            print("  wget https://huggingface.co/allenai/dolma3-fasttext-quality-classifier/resolve/main/model.bin -O models/dolma3_quality_classifier.bin")
            raise FileNotFoundError(f"Model not found. Please download the model first.")
        
        print(f"Loading Dolma3 quality classifier from: {model_path}")
        _model = fasttext.load_model(model_path)
        print("Model loaded successfully!")
    
    return _model


# Label mapping - Dolma3 uses __label__hq (high quality) and __label__lq (low quality)
LABEL_MAP = {
    "__label__hq": "High",
    "__label__lq": "Low",
    "hq": "High",
    "lq": "Low",
}


def classify_quality(text: str, model_path: str = None) -> Tuple[str, float]:
    """
    Classify the quality of text using Dolma3 FastText model.
    
    Args:
        text: Input text to classify
        model_path: Optional path to model file
        
    Returns:
        Tuple of (quality_label, confidence_score)
        - quality_label: "High" or "Low"
        - confidence_score: Probability score for the predicted label (0-1)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "Low", 0.0
    
    model = _load_model(model_path)
    
    # Clean text - FastText expects single line
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())
    
    # Get prediction
    labels, probs = model.predict(clean_text, k=2)
    
    primary_label = labels[0]
    primary_prob = probs[0]
    
    # Map to standard labels
    label = LABEL_MAP.get(primary_label, LABEL_MAP.get(primary_label.replace("__label__", ""), "Low"))
    
    return label, float(primary_prob)


def classify_quality_batch(texts: list[str], model_path: str = None) -> list[Tuple[str, float]]:
    """
    Classify quality for a batch of texts.
    
    Args:
        texts: List of input texts
        model_path: Optional path to model file
        
    Returns:
        List of (quality_label, confidence_score) tuples
    """
    if not texts:
        return []
    
    model = _load_model(model_path)
    results = []
    
    for text in texts:
        if not text or not isinstance(text, str) or not text.strip():
            results.append(("Low", 0.0))
            continue
        
        # Clean text
        clean_text = text.replace("\n", " ").replace("\r", " ")
        clean_text = " ".join(clean_text.split())
        
        labels, probs = model.predict(clean_text)
        label = LABEL_MAP.get(labels[0], LABEL_MAP.get(labels[0].replace("__label__", ""), "Low"))
        results.append((label, float(probs[0])))
    
    return results


def is_quality_acceptable(text: str, model_path: str = None) -> bool:
    """
    Check if text quality is acceptable (High quality).
    
    Args:
        text: Input text to check
        model_path: Optional path to model file
        
    Returns:
        True if text is high quality, False otherwise
    """
    label, _ = classify_quality(text, model_path)
    return label == "High"


def map_quality_to_category(label: str) -> str:
    """
    Map quality labels to test categories.
    
    Maps:
    - "High" -> "wiki" (high quality, similar to Wikipedia)
    - "Low" -> "cc" (low quality Common Crawl)
    
    Args:
        label: Quality label ("High" or "Low")
        
    Returns:
        Category string ("wiki" or "cc")
    """
    if label == "High":
        return "wiki"
    else:
        return "cc"


def classify_quality_with_category(text: str, model_path: str = None) -> Tuple[str, float]:
    """
    Classify text quality and return the mapped category.
    
    This is the function expected by the test adapters.
    
    Args:
        text: Input text to classify
        model_path: Optional path to model file
        
    Returns:
        Tuple of (category, confidence_score)
        - category: "wiki" (High quality) or "cc" (Low quality)
        - confidence_score: Probability score (0-1)
    """
    label, score = classify_quality(text, model_path)
    category = map_quality_to_category(label)
    return category, score


def filter_by_quality(text: str, model_path: str = None) -> Tuple[bool, str, float]:
    """
    Filter text based on quality - returns whether to keep the text.
    
    Args:
        text: Input text to filter
        model_path: Optional path to model file
        
    Returns:
        Tuple of (keep_text, quality_label, confidence)
        - keep_text: True if quality is High (should be kept)
        - quality_label: "High" or "Low"
        - confidence: Confidence score
    """
    label, confidence = classify_quality(text, model_path)
    keep = label == "High"
    return keep, label, confidence


def get_all_probabilities(text: str, model_path: str = None) -> dict:
    """
    Get probabilities for all quality classes.
    
    Args:
        text: Input text to classify
        model_path: Optional path to model file
        
    Returns:
        Dictionary with probabilities for each class
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"High": 0.0, "Low": 1.0}
    
    model = _load_model(model_path)
    
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())
    
    labels, probs = model.predict(clean_text, k=2)
    
    result = {"High": 0.0, "Low": 0.0}
    for label, prob in zip(labels, probs):
        mapped_label = LABEL_MAP.get(label, LABEL_MAP.get(label.replace("__label__", ""), "Low"))
        result[mapped_label] = float(prob)
    
    return result
