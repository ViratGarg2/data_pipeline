"""
Quality Classifier Module using NVIDIA DeBERTa Model

Uses the nvidia/quality-classifier-deberta model to classify text quality
into three categories: Low, Medium, High.

This module provides:
1. classify_quality() - Returns quality label and confidence score
2. is_quality_acceptable() - Returns True if quality is Medium or High
"""

import torch
from typing import Tuple

# Global model and tokenizer (lazy loaded)
_model = None
_tokenizer = None
_device = None


def _load_model():
    """Lazy load the model and tokenizer."""
    global _model, _tokenizer, _device
    
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "nvidia/quality-classifier-deberta"
        
        print(f"Loading quality classifier model: {model_name}")
        
        # Determine device
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
        
        print(f"Using device: {_device}")
        
        # Load tokenizer and model
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.to(_device)
        _model.eval()
        
        print("Quality classifier model loaded successfully")
    
    return _model, _tokenizer, _device


# Label mapping from model output
# The nvidia/quality-classifier-deberta model outputs:
# 0: Low, 1: Medium, 2: High
LABEL_MAP = {
    0: "Low",
    1: "Medium", 
    2: "High"
}

# Reverse mapping for convenience
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}


def classify_quality(text: str, max_length: int = 512) -> Tuple[str, float]:
    """
    Classify the quality of text using NVIDIA DeBERTa model.
    
    Args:
        text: Input text to classify
        max_length: Maximum token length for the model
        
    Returns:
        Tuple of (quality_label, confidence_score)
        - quality_label: "Low", "Medium", or "High"
        - confidence_score: Probability score for the predicted label (0-1)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "Low", 0.0
    
    model, tokenizer, device = _load_model()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
    
    label = LABEL_MAP.get(predicted_class, "Low")
    
    return label, confidence


def classify_quality_batch(texts: list[str], max_length: int = 512, batch_size: int = 32) -> list[Tuple[str, float]]:
    """
    Classify quality for a batch of texts.
    
    Args:
        texts: List of input texts
        max_length: Maximum token length
        batch_size: Number of texts to process at once
        
    Returns:
        List of (quality_label, confidence_score) tuples
    """
    if not texts:
        return []
    
    model, tokenizer, device = _load_model()
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Handle empty/None texts
        valid_indices = []
        valid_texts = []
        for j, text in enumerate(batch_texts):
            if text and isinstance(text, str) and text.strip():
                valid_indices.append(j)
                valid_texts.append(text)
        
        # Initialize batch results with defaults
        batch_results = [("Low", 0.0)] * len(batch_texts)
        
        if valid_texts:
            # Tokenize batch
            inputs = tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
                
                for idx, (orig_idx, pred_class) in enumerate(zip(valid_indices, predicted_classes)):
                    label = LABEL_MAP.get(pred_class.item(), "Low")
                    confidence = probs[idx, pred_class].item()
                    batch_results[orig_idx] = (label, confidence)
        
        results.extend(batch_results)
    
    return results


def is_quality_acceptable(text: str, min_quality: str = "Medium") -> bool:
    """
    Check if text quality meets the minimum threshold.
    
    Args:
        text: Input text to check
        min_quality: Minimum acceptable quality ("Low", "Medium", or "High")
        
    Returns:
        True if text quality is >= min_quality, False otherwise
    """
    quality_order = {"Low": 0, "Medium": 1, "High": 2}
    
    label, _ = classify_quality(text)
    
    return quality_order.get(label, 0) >= quality_order.get(min_quality, 1)


def map_quality_to_category(label: str) -> str:
    """
    Map quality labels to broader categories for filtering.
    
    Maps:
    - "High" -> "wiki" (high quality, similar to Wikipedia)
    - "Medium" -> "wiki" (acceptable quality)
    - "Low" -> "cc" (low quality Common Crawl)
    
    Args:
        label: Quality label ("Low", "Medium", "High")
        
    Returns:
        Category string ("wiki" or "cc")
    """
    if label in ("High", "Medium"):
        return "wiki"
    else:
        return "cc"


def classify_quality_with_category(text: str) -> Tuple[str, float]:
    """
    Classify text quality and return the mapped category.
    
    This is the function expected by the test adapters.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (category, confidence_score)
        - category: "wiki" (Medium/High quality) or "cc" (Low quality)
        - confidence_score: Probability score (0-1)
    """
    label, score = classify_quality(text)
    category = map_quality_to_category(label)
    return category, score


# Convenience function for pipeline integration
def filter_by_quality(text: str) -> Tuple[bool, str, float]:
    """
    Filter text based on quality - returns whether to keep the text.
    
    Args:
        text: Input text to filter
        
    Returns:
        Tuple of (keep_text, quality_label, confidence)
        - keep_text: True if quality is Medium or High (should be kept)
        - quality_label: "Low", "Medium", or "High"
        - confidence: Confidence score
    """
    label, confidence = classify_quality(text)
    keep = label in ("Medium", "High")
    return keep, label, confidence
