"""
Language Identification Module using FastText.

This module provides functions to identify the language of text content
using the FastText language identification classifier (lid.176.bin).
"""

import os
from pathlib import Path

import fasttext

# Suppress FastText warnings about loading model
fasttext.FastText.eprint = lambda x: None

# Default model path - looks for model in the 'model' directory relative to this file
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model" / "lid.176.bin"

# Global model instance (lazy loaded)
_model = None


def _get_model(model_path: str | Path | None = None) -> fasttext.FastText._FastText:
    """
    Get the FastText language identification model (lazy loading).
    
    Args:
        model_path: Path to the FastText model file. If None, uses default path.
        
    Returns:
        Loaded FastText model
    """
    global _model
    
    if _model is None:
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"FastText language ID model not found at {model_path}. "
                "Please download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html"
            )
        
        _model = fasttext.load_model(str(model_path))
    
    return _model


def identify_language(text: str, model_path: str | Path | None = None) -> tuple[str, float]:
    """
    Identify the main language present in the given text.
    
    Uses FastText's language identification classifier to predict the language
    and return a confidence score between 0 and 1.
    
    Args:
        text: Unicode string to identify the language of
        model_path: Optional path to FastText model file
        
    Returns:
        Tuple of (language_code, confidence_score)
        - language_code: ISO 639-1 language code (e.g., 'en', 'zh', 'es')
        - confidence_score: Float between 0 and 1 representing confidence
    """
    if not text or not isinstance(text, str):
        return ("unknown", 0.0)
    
    # FastText expects single line input, replace newlines with spaces
    text_clean = text.replace("\n", " ").replace("\r", " ")
    
    # Limit text length for performance (first 1000 chars is usually enough)
    text_clean = text_clean[:10000]
    
    model = _get_model(model_path)
    
    # Get prediction - returns (labels, probabilities)
    # Labels are in format '__label__en', '__label__zh', etc.
    predictions = model.predict(text_clean, k=1)
    
    if not predictions or not predictions[0]:
        return ("unknown", 0.0)
    
    # Extract language code from label (remove '__label__' prefix)
    label = predictions[0][0]
    language_code = label.replace("__label__", "")
    
    # Get confidence score (probability)
    confidence = float(predictions[1][0])
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    return (language_code, confidence)


def identify_language_top_k(
    text: str, 
    k: int = 5, 
    model_path: str | Path | None = None
) -> list[tuple[str, float]]:
    """
    Identify the top-k languages present in the given text.
    
    Args:
        text: Unicode string to identify the language of
        k: Number of top predictions to return
        model_path: Optional path to FastText model file
        
    Returns:
        List of tuples (language_code, confidence_score) sorted by confidence
    """
    if not text or not isinstance(text, str):
        return [("unknown", 0.0)]
    
    # FastText expects single line input
    text_clean = text.replace("\n", " ").replace("\r", " ")[:10000]
    
    model = _get_model(model_path)
    
    predictions = model.predict(text_clean, k=k)
    
    if not predictions or not predictions[0]:
        return [("unknown", 0.0)]
    
    results = []
    for label, prob in zip(predictions[0], predictions[1]):
        language_code = label.replace("__label__", "")
        confidence = max(0.0, min(1.0, float(prob)))
        results.append((language_code, confidence))
    
    return results


# Language code to name mapping for common languages
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "ko": "Korean",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "bn": "Bengali",
    "fa": "Persian",
    "unknown": "Unknown",
}


def get_language_name(language_code: str) -> str:
    """
    Get the human-readable name for a language code.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        Human-readable language name
    """
    return LANGUAGE_NAMES.get(language_code, language_code)


if __name__ == "__main__":
    # Test the language identification
    test_texts = [
        ("Hello, this is a test in English.", "English"),
        ("Bonjour, ceci est un test en français.", "French"),
        ("Hola, esto es una prueba en español.", "Spanish"),
        ("欢迎来到我们的网站", "Chinese"),
        ("これは日本語のテストです。", "Japanese"),
        ("Привет, это тест на русском языке.", "Russian"),
    ]
    
    print("Language Identification Test")
    print("=" * 60)
    
    for text, expected in test_texts:
        lang, score = identify_language(text)
        lang_name = get_language_name(lang)
        print(f"Text: {text[:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Detected: {lang_name} ({lang}) with confidence {score:.4f}")
        print()
