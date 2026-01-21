"""
NSFW and Toxic Speech Detection Module using FastText.

This module provides functions to classify text content for NSFW and toxic speech
using FastText classifiers trained on the Jigsaw dataset.
"""

import os
from pathlib import Path

import fasttext

# Suppress FastText warnings about loading model
fasttext.FastText.eprint = lambda x: None

# Default model paths - looks for models in the 'model' directory relative to this file
MODEL_PATH_HATESPEECH = Path(__file__).parent.parent / "model" / "jigsaw_fasttext_bigrams_hatespeech_final.bin"
MODEL_PATH_NSFW = Path(__file__).parent.parent / "model" / "jigsaw_fasttext_bigrams_nsfw_final.bin"

# Global model instances (lazy loaded)
_nsfw_model = None
_hatespeech_model = None


def _get_nsfw_model(model_path: str | Path | None = None) -> fasttext.FastText._FastText:
    """
    Get the FastText NSFW classification model (lazy loading).
    
    Args:
        model_path: Path to the FastText model file. If None, uses default path.
        
    Returns:
        Loaded FastText model
    """
    global _nsfw_model
    
    if _nsfw_model is None:
        if model_path is None:
            model_path = MODEL_PATH_NSFW
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"FastText NSFW model not found at {model_path}. "
                "Please ensure the model file exists."
            )
        
        _nsfw_model = fasttext.load_model(str(model_path))
    
    return _nsfw_model


def _get_hatespeech_model(model_path: str | Path | None = None) -> fasttext.FastText._FastText:
    """
    Get the FastText hate speech classification model (lazy loading).
    
    Args:
        model_path: Path to the FastText model file. If None, uses default path.
        
    Returns:
        Loaded FastText model
    """
    global _hatespeech_model
    
    if _hatespeech_model is None:
        if model_path is None:
            model_path = MODEL_PATH_HATESPEECH
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"FastText hate speech model not found at {model_path}. "
                "Please ensure the model file exists."
            )
        
        _hatespeech_model = fasttext.load_model(str(model_path))
    
    return _hatespeech_model


def classify_nsfw(text: str, model_path: str | Path | None = None) -> tuple[str, float]:
    """
    Classify text as NSFW or non-NSFW.
    
    Uses FastText classifier trained on Jigsaw dataset to predict whether
    the text contains NSFW content.
    
    Args:
        text: Unicode string to classify
        model_path: Optional path to FastText model file
        
    Returns:
        Tuple of (label, confidence_score)
        - label: 'nsfw' or 'non-nsfw'
        - confidence_score: Float between 0 and 1 representing confidence
    """
    if not text or not isinstance(text, str):
        return ("non-nsfw", 0.0)
    
    # FastText expects single line input, replace newlines with spaces
    text_clean = text.replace("\n", " ").replace("\r", " ")
    
    # Limit text length for performance
    text_clean = text_clean[:10000]
    
    model = _get_nsfw_model(model_path)
    
    # Get prediction - returns (labels, probabilities)
    predictions = model.predict(text_clean, k=2)
    
    if not predictions or not predictions[0]:
        return ("non-nsfw", 0.0)
    
    # Extract label and confidence
    label = predictions[0][0]
    confidence = float(predictions[1][0])
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    # Normalize label - FastText labels are typically like '__label__nsfw' or '__label__1'
    label_clean = label.replace("__label__", "").lower()
    
    # Map to expected output format
    if label_clean in ("nsfw", "1", "yes", "true", "positive"):
        # print(f"NSFW detected with confidence {confidence:.4f}")
        return ("nsfw", confidence)
    else:
        # print(f"Non-NSFW detected with confidence {confidence:.4f}")
        return ("non-nsfw", confidence)


def classify_toxic_speech(text: str, model_path: str | Path | None = None) -> tuple[str, float]:
    """
    Classify text as toxic or non-toxic speech.
    
    Uses FastText classifier trained on Jigsaw dataset to predict whether
    the text contains toxic/hate speech.
    
    Args:
        text: Unicode string to classify
        model_path: Optional path to FastText model file
        
    Returns:
        Tuple of (label, confidence_score)
        - label: 'toxic' or 'non-toxic'
        - confidence_score: Float between 0 and 1 representing confidence
    """
    if not text or not isinstance(text, str):
        return ("non-toxic", 0.0)
    
    # FastText expects single line input, replace newlines with spaces
    text_clean = text.replace("\n", " ").replace("\r", " ")
    
    # Limit text length for performance
    text_clean = text_clean[:10000]
    
    model = _get_hatespeech_model(model_path)
    
    # Get prediction - returns (labels, probabilities)
    predictions = model.predict(text_clean, k=2)
    
    if not predictions or not predictions[0]:
        return ("non-toxic", 0.0)
    
    # Extract label and confidence
    label = predictions[0][0]
    confidence = float(predictions[1][0])
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    # Normalize label - FastText labels are typically like '__label__toxic' or '__label__1'
    label_clean = label.replace("__label__", "").lower()
    
    # Map to expected output format
    if label_clean in ("toxic", "hate", "hatespeech", "1", "yes", "true", "positive"):
        # print(f"Toxic detected with confidence {confidence:.4f}")
        return ("toxic", confidence)
    else:
        # print(f"Non-toxic detected with confidence {confidence:.4f}")
        return ("non-toxic", confidence)


def classify_content(text: str) -> dict:
    """
    Classify text for both NSFW and toxic content.
    
    Convenience function that runs both classifiers and returns results.
    
    Args:
        text: Unicode string to classify
        
    Returns:
        Dictionary with classification results:
        {
            'nsfw_label': str,
            'nsfw_score': float,
            'toxic_label': str,
            'toxic_score': float,
        }
    """
    nsfw_label, nsfw_score = classify_nsfw(text)
    toxic_label, toxic_score = classify_toxic_speech(text)
    
    return {
        'nsfw_label': nsfw_label,
        'nsfw_score': nsfw_score,
        'toxic_label': toxic_label,
        'toxic_score': toxic_score,
    }