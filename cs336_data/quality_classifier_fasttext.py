"""
Quality Classifier Module using FastText Model

Uses a trained FastText model to classify text quality into two categories:
positive (high quality) and negative (low quality).

This module provides:
1. classify_quality() - Returns quality label and confidence score
2. is_quality_acceptable() - Returns True if quality is positive (high quality)
3. classify_quality_with_category() - Returns mapped category for test adapters

Usage:
    from cs336_data.quality_classifier_fasttext import classify_quality, is_quality_acceptable
    
    label, score = classify_quality("Some text to check")
    # label: "positive" or "negative"
    # score: confidence score (0-1)
    
    if is_quality_acceptable("Some text"):
        print("Text is high quality!")
"""

import os
from pathlib import Path
from typing import Tuple

# Global model (lazy loaded)
_model = None

# Default model path - can be overridden
DEFAULT_MODEL_PATH = os.environ.get(
    "FASTTEXT_QUALITY_MODEL",
    str(Path(__file__).parent.parent / "urls" / "quality_model.bin")
)


def _load_model(model_path: str | None = None):
    """Lazy load the FastText model."""
    global _model
    
    if _model is None:
        import fasttext
        
        path = model_path or DEFAULT_MODEL_PATH
        
        if not Path(path).exists():
            raise FileNotFoundError(
                f"FastText quality model not found at: {path}\n"
                f"Please train a model first using train_quality_classifier.py\n"
                f"Or set FASTTEXT_QUALITY_MODEL environment variable to the model path."
            )
        
        # print(f"Loading FastText quality classifier: {path}")
        _model = fasttext.load_model(path)
        # print("FastText quality classifier loaded successfully")
    
    return _model


def set_model_path(model_path: str):
    """
    Set a custom model path and reload the model.
    
    Args:
        model_path: Path to the FastText .bin model file
    """
    global _model, DEFAULT_MODEL_PATH
    DEFAULT_MODEL_PATH = model_path
    _model = None  # Force reload on next use


def classify_quality(text: str, model_path: str | None = None) -> Tuple[str, float]:
    """
    Classify the quality of text using FastText model.
    
    Args:
        text: Input text to classify
        model_path: Optional path to FastText model (uses default if not specified)
        
    Returns:
        Tuple of (quality_label, confidence_score)
        - quality_label: "positive" (high quality) or "negative" (low quality)
        - confidence_score: Probability score for the predicted label (0-1)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "negative", 0.0
    
    model = _load_model(model_path)
    
    # FastText expects single line, replace newlines with spaces
    text_clean = text.replace("\n", " ").replace("\r", " ")
    text_clean = " ".join(text_clean.split())  # Collapse multiple spaces
    
    # Get prediction
    labels, probs = model.predict(text_clean, k=1)
    
    # Extract label and probability
    label = labels[0].replace("__label__", "")
    prob = probs[0]
    
    return label, float(prob)

def is_quality_acceptable(text: str, model_path: str | None = None) -> bool:
    """
    Check if text quality is acceptable (positive/high quality).
    
    Args:
        text: Input text to check
        model_path: Optional path to FastText model
        
    Returns:
        True if text is classified as positive (high quality), False otherwise
    """
    label, _ = classify_quality(text, model_path)
    return label == "positive"

def classify_quality_with_category(text: str, model_path: str | None = None) -> Tuple[str, float]:
    """
    Classify text quality and return the mapped category.
    
    This is the function expected by the test adapters.
    
    Args:
        text: Input text to classify
        model_path: Optional path to FastText model
        
    Returns:
        Tuple of (category, confidence_score)
        - category: "wiki" (positive/high quality) or "cc" (negative/low quality)
        - confidence_score: Probability score (0-1)
    """
    label, score = classify_quality(text, model_path)
    if label == "positive":
        category = "wiki"
    else:
        category = "cc"
    # category = map_quality_to_category(label)
    return category, score


def filter_by_quality(text: str, model_path: str | None = None) -> Tuple[bool, str, float]:
    """
    Filter text based on quality - returns whether to keep the text.
    
    Args:
        text: Input text to filter
        model_path: Optional path to FastText model
        
    Returns:
        Tuple of (keep_text, quality_label, confidence)
        - keep_text: True if quality is positive (should be kept)
        - quality_label: "positive" or "negative"
        - confidence: Confidence score
    """
    label, confidence = classify_quality(text, model_path)
    keep = label == "positive"
    return keep, label, confidence


def get_all_predictions(text: str, model_path: str | None = None) -> dict:
    """
    Get detailed prediction information for a text.
    
    Args:
        text: Input text to classify
        model_path: Optional path to FastText model
        
    Returns:
        Dictionary with all prediction details
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "label": "negative",
            "category": "cc",
            "confidence": 0.0,
            "all_labels": [],
            "all_probs": [],
        }
    
    model = _load_model(model_path)
    
    # Clean text
    text_clean = text.replace("\n", " ").replace("\r", " ")
    text_clean = " ".join(text_clean.split())
    
    # Get all predictions
    labels, probs = model.predict(text_clean, k=2)  # Get both labels
    
    # Process results
    all_labels = [l.replace("__label__", "") for l in labels]
    all_probs = [float(p) for p in probs]
    
    top_label = all_labels[0]
    top_prob = all_probs[0]
    
    return {
        "label": top_label,
        # "category": map_quality_to_category(top_label),
        "confidence": top_prob,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }


# # CLI for testing
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Test FastText quality classifier")
#     parser.add_argument("--model", "-m", type=str, default=None, help="Path to model file")
#     parser.add_argument("--text", "-t", type=str, help="Text to classify")
#     parser.add_argument("--file", "-f", type=str, help="File with texts to classify (one per line)")
    
#     args = parser.parse_args()
    
#     if args.model:
#         set_model_path(args.model)
    
#     if args.text:
#         label, score = classify_quality(args.text)
#         category = map_quality_to_category(label)
#         print(f"Label: {label}")
#         print(f"Category: {category}")
#         print(f"Confidence: {score:.4f}")
#         print(f"Keep: {label == 'positive'}")
    
#     elif args.file:
#         with open(args.file, "r", encoding="utf-8") as f:
#             for i, line in enumerate(f, 1):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 label, score = classify_quality(line)
#                 category = map_quality_to_category(label)
#                 preview = line[:60] + "..." if len(line) > 60 else line
#                 print(f"[{i}] [{category}] ({score:.3f}): {preview}")
    
#     else:
#         # Interactive mode
#         print("FastText Quality Classifier - Interactive Mode")
#         print("Enter text to classify (Ctrl+D to exit):")
#         print("-" * 50)
        
#         try:
#             while True:
#                 text = input("\n> ")
#                 if not text.strip():
#                     continue
                
#                 result = get_all_predictions(text)
#                 print(f"  Label: {result['label']}")
#                 print(f"  Category: {result['category']}")
#                 print(f"  Confidence: {result['confidence']:.4f}")
#                 print(f"  Keep: {result['label'] == 'positive'}")
#         except EOFError:
#             print("\nGoodbye!")
