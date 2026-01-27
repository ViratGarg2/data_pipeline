#!/usr/bin/env python3
"""
Interactive tool to test the FastText quality classifier model.

Usage:
    # Interactive mode - enter texts manually
    python test_quality_model.py --model quality_classifier.bin
    
    # Test with a specific text
    python test_quality_model.py --model quality_classifier.bin --text "Your text here"
    
    # Test with texts from a file (one per line)
    python test_quality_model.py --model quality_classifier.bin --file texts_to_test.txt
    
    # Run built-in test examples
    python test_quality_model.py --model quality_classifier.bin --examples
"""

import argparse
from pathlib import Path

import fasttext


def load_model(model_path: str) -> fasttext.FastText._FastText:
    """Load the FastText model from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = fasttext.load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"  Labels: {model.labels}")
    print()
    return model


def predict_quality(model: fasttext.FastText._FastText, text: str, verbose: bool = True) -> tuple[str, float]:
    """
    Predict the quality of a text.
    
    Args:
        model: Loaded FastText model
        text: Text to classify
        verbose: Whether to print results
        
    Returns:
        Tuple of (label, confidence)
    """
    # Clean text for prediction (single line)
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())  # Collapse whitespace
    
    # Get prediction
    labels, probs = model.predict(clean_text, k=2)  # Get top 2 predictions
    
    primary_label = labels[0].replace("__label__", "")
    primary_prob = probs[0]
    
    if verbose:
        print("-" * 70)
        print(f"TEXT: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"\nPREDICTION: {primary_label.upper()}")
        print(f"CONFIDENCE: {primary_prob:.4f} ({primary_prob:.2%})")
        
        if len(labels) > 1:
            alt_label = labels[1].replace("__label__", "")
            alt_prob = probs[1]
            print(f"\nAlternative: {alt_label} ({alt_prob:.4f})")
        
        # Quality interpretation
        if primary_label == "positive":
            print(f"\n✅ HIGH QUALITY - This text appears to be well-written/informative")
        else:
            print(f"\n❌ LOW QUALITY - This text appears to be low quality/spam-like")
        print("-" * 70)
    
    return primary_label, primary_prob


def run_interactive(model: fasttext.FastText._FastText):
    """Run interactive mode where user can input texts."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Enter text to classify (or 'quit' to exit, 'multiline' for multi-line input)")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'multiline':
                print("Enter multiple lines (type 'END' on a new line when done):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                user_input = "\n".join(lines)
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            predict_quality(model, user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def run_examples(model: fasttext.FastText._FastText):
    """Run prediction on built-in example texts."""
    examples = [
        # High quality examples
        (
            "The theory of evolution by natural selection, first formulated in Darwin's book "
            "On the Origin of Species in 1859, is the process by which organisms change over time "
            "as a result of changes in heritable physical or behavioral traits."
        ),
        (
            "Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being "
            "explicitly programmed. Machine learning focuses on the development of computer "
            "programs that can access data and use it to learn for themselves."
        ),
        (
            "The French Revolution was a period of radical political and societal change in France "
            "that began with the Estates General of 1789 and ended with the formation of the French "
            "Consulate in November 1799. Many of its ideas are considered fundamental principles "
            "of liberal democracy."
        ),
        
        # Low quality examples
        (
            "CLICK HERE NOW!!! FREE MONEY BEST DEALS LIMITED TIME OFFER "
            "BUY NOW CHEAP PRICES AMAZING DISCOUNTS!!!"
        ),
        (
            "asdfasdf jkljkl random text spam spam spam click click click "
            "free free free buy now sale sale sale"
        ),
        (
            "Copyright 2024 All Rights Reserved. Privacy Policy. Terms of Service. "
            "Contact Us. About. FAQ. Home. Menu. Login. Register. Search."
        ),
        (
            "Re: Re: Re: FWD: FWD: You won't believe this!!! "
            "Share with 10 friends or bad luck for 7 years!!!"
        ),
        
        # Borderline examples
        (
            "This is a simple sentence. It has basic structure. "
            "The words are common. Nothing special here."
        ),
        (
            "Hello world. This is a test message. "
            "I am writing some text to see how the model performs."
        ),
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING BUILT-IN EXAMPLES")
    print("=" * 70)
    
    results = {"positive": 0, "negative": 0}
    
    for i, text in enumerate(examples, 1):
        print(f"\n[Example {i}/{len(examples)}]")
        label, prob = predict_quality(model, text)
        results[label] += 1
        print()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print(f"Classified as POSITIVE (high quality): {results['positive']}")
    print(f"Classified as NEGATIVE (low quality): {results['negative']}")


def test_from_file(model: fasttext.FastText._FastText, file_path: str):
    """Test model on texts from a file (one text per line)."""
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nReading texts from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts to classify\n")
    
    results = {"positive": 0, "negative": 0}
    predictions = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}]")
        label, prob = predict_quality(model, text)
        results[label] += 1
        predictions.append((text[:50], label, prob))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total texts: {len(texts)}")
    print(f"Positive (high quality): {results['positive']} ({results['positive']/len(texts):.1%})")
    print(f"Negative (low quality): {results['negative']} ({results['negative']/len(texts):.1%})")
    
    # Show confidence distribution
    print("\n" + "-" * 70)
    print("PREDICTIONS BY CONFIDENCE:")
    print("-" * 70)
    
    sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    print("\nMost confident predictions:")
    for text, label, prob in sorted_preds[:5]:
        print(f"  [{label}] {prob:.3f}: {text}...")
    
    print("\nLeast confident predictions:")
    for text, label, prob in sorted_preds[-5:]:
        print(f"  [{label}] {prob:.3f}: {text}...")


def main():
    parser = argparse.ArgumentParser(
        description="Test FastText quality classifier model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="quality_classifier.bin",
        help="Path to the FastText model file (default: quality_classifier.bin)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Single text to classify"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="File with texts to classify (one per line)"
    )
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Run built-in example texts"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have trained the model first:")
        print("  python train_quality_classifier.py --positive positive.txt --negative negative.txt")
        return 1
    
    # Determine mode
    if args.text:
        # Single text mode
        predict_quality(model, args.text)
    elif args.file:
        # File mode
        test_from_file(model, args.file)
    elif args.examples:
        # Examples mode
        run_examples(model)
    else:
        # Default to interactive mode
        run_interactive(model)
    
    return 0


if __name__ == "__main__":
    exit(main())
