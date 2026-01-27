#!/usr/bin/env python3
"""
Interactive tool to test the NVIDIA DeBERTa quality classifier model.

This uses the nvidia/quality-classifier-deberta model which classifies text
into three quality categories: Low, Medium, High.

Usage:
    # Interactive mode - enter texts manually
    python test_nvidia_quality_model.py
    
    # Test with a specific text
    python test_nvidia_quality_model.py --text "Your text here"
    
    # Test with texts from a file (one per line)
    python test_nvidia_quality_model.py --file texts_to_test.txt
    
    # Run built-in test examples
    python test_nvidia_quality_model.py --examples
    
    # Compare with FastText model (if available)
    python test_nvidia_quality_model.py --compare --fasttext-model quality_classifier.bin
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Global model cache
_model = None
_tokenizer = None
_device = None

# Label mapping
LABEL_MAP = {
    0: "Low",
    1: "Medium", 
    2: "High"
}


def load_model(force_cpu: bool = False):
    """Load the NVIDIA DeBERTa quality classifier model."""
    global _model, _tokenizer, _device
    
    if _model is None:
        model_name = "nvidia/quality-classifier-deberta"
        
        print(f"Loading model: {model_name}")
        print("This may take a moment on first run (downloading model)...")
        
        # Determine device
        if force_cpu:
            _device = torch.device("cpu")
            print("Forcing CPU mode")
        elif torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                _device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"CUDA available but failed ({e}), falling back to CPU")
                _device = torch.device("cpu")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device("mps")
            print("Using Apple Silicon (MPS)")
        else:
            _device = torch.device("cpu")
            print("Using CPU")
        
        # Load tokenizer and model
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.to(_device)
        _model.eval()
        
        print("Model loaded successfully!\n")
    
    return _model, _tokenizer, _device


def predict_quality(text: str, verbose: bool = True) -> tuple[str, float, dict]:
    """
    Predict the quality of a text using NVIDIA DeBERTa model.
    
    Args:
        text: Text to classify
        verbose: Whether to print results
        
    Returns:
        Tuple of (label, confidence, all_probabilities)
    """
    model, tokenizer, device = load_model()
    
    # Clean and truncate text for display
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        # Get all probabilities
        all_probs = {
            LABEL_MAP[i]: probs[0, i].item() 
            for i in range(len(LABEL_MAP))
        }
    
    label = LABEL_MAP[predicted_class]
    
    if verbose:
        print("-" * 70)
        print(f"TEXT: {clean_text[:200]}{'...' if len(clean_text) > 200 else ''}")
        print(f"\n{'='*30} PREDICTION {'='*30}")
        print(f"\n  Quality Level: {label.upper()}")
        print(f"  Confidence:    {confidence:.4f} ({confidence:.2%})")
        print(f"\n  All Probabilities:")
        print(f"    Low:    {all_probs['Low']:.4f} ({all_probs['Low']:.2%})")
        print(f"    Medium: {all_probs['Medium']:.4f} ({all_probs['Medium']:.2%})")
        print(f"    High:   {all_probs['High']:.4f} ({all_probs['High']:.2%})")
        
        # Quality interpretation with visual indicator
        if label == "High":
            print(f"\n  ✅ HIGH QUALITY - Excellent, well-written content")
        elif label == "Medium":
            print(f"\n  ⚠️  MEDIUM QUALITY - Acceptable content")
        else:
            print(f"\n  ❌ LOW QUALITY - Poor quality, consider filtering")
        
        # Recommendation for pipeline
        if label in ("High", "Medium"):
            print(f"\n  📋 Pipeline: KEEP (Medium/High quality passes filter)")
        else:
            print(f"\n  📋 Pipeline: DISCARD (Low quality filtered out)")
        
        print("-" * 70)
    
    return label, confidence, all_probs


def run_interactive():
    """Run interactive mode where user can input texts."""
    print("\n" + "=" * 70)
    print("NVIDIA DeBERTa QUALITY CLASSIFIER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nEnter text to classify. Commands:")
    print("  'quit' or 'q' - Exit")
    print("  'multi' - Enter multi-line text (end with 'END' on its own line)")
    print("  'clear' - Clear screen")
    print("=" * 70 + "\n")
    
    # Pre-load model
    load_model()
    
    while True:
        try:
            user_input = input("\n📝 Enter text: ").strip()
            
            if user_input.lower() in ('quit', 'q'):
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                print("\033[H\033[J")  # Clear screen
                continue
            
            if user_input.lower() == 'multi':
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
            
            predict_quality(user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def run_examples():
    """Run prediction on built-in example texts."""
    examples = [
        # Expected HIGH quality
        {
            "text": (
                "The theory of evolution by natural selection, first formulated in Darwin's book "
                "On the Origin of Species in 1859, is the process by which organisms change over time "
                "as a result of changes in heritable physical or behavioral traits. Changes that allow "
                "an organism to better adapt to its environment will help it survive and have more offspring."
            ),
            "expected": "High",
            "description": "Scientific explanation (Wikipedia-style)"
        },
        {
            "text": (
                "Machine learning is a subset of artificial intelligence that provides systems "
                "the ability to automatically learn and improve from experience without being "
                "explicitly programmed. Machine learning focuses on the development of computer "
                "programs that can access data and use it to learn for themselves. The process "
                "begins with observations or data, such as examples, direct experience, or instruction."
            ),
            "expected": "High",
            "description": "Technical article about ML"
        },
        {
            "text": (
                "The French Revolution was a period of radical political and societal change in France "
                "that began with the Estates General of 1789 and ended with the formation of the French "
                "Consulate in November 1799. Many of its ideas are considered fundamental principles "
                "of liberal democracy, while the phrase 'Liberté, égalité, fraternité' reappeared "
                "in other revolts and became the motto of the French Republic."
            ),
            "expected": "High", 
            "description": "Historical article"
        },
        
        # Expected LOW quality
        {
            "text": (
                "CLICK HERE NOW!!! FREE MONEY BEST DEALS LIMITED TIME OFFER "
                "BUY NOW CHEAP PRICES AMAZING DISCOUNTS!!! ACT FAST!!!"
            ),
            "expected": "Low",
            "description": "Spam/advertising content"
        },
        {
            "text": (
                "asdfasdf jkljkl random text spam spam spam click click click "
                "free free free buy now sale sale sale best best best"
            ),
            "expected": "Low",
            "description": "Random/gibberish text"
        },
        {
            "text": (
                "Copyright 2024 All Rights Reserved. Privacy Policy. Terms of Service. "
                "Contact Us. About. FAQ. Home. Menu. Login. Register. Search. Sitemap."
            ),
            "expected": "Low",
            "description": "Boilerplate/navigation text"
        },
        {
            "text": (
                "Re: Re: Re: FWD: FWD: You won't believe this!!! "
                "Share with 10 friends or bad luck for 7 years!!! "
                "This really works I tried it!!!"
            ),
            "expected": "Low",
            "description": "Chain mail/forwarded spam"
        },
        
        # Expected MEDIUM quality (borderline)
        {
            "text": (
                "This is a simple blog post about my day. I woke up early and had breakfast. "
                "Then I went to work and had some meetings. After work, I went to the gym. "
                "It was a pretty normal day overall. Tomorrow I plan to do something similar."
            ),
            "expected": "Medium",
            "description": "Simple blog post"
        },
        {
            "text": (
                "Product Review: I bought this phone last week and it's pretty good. "
                "The camera is decent and battery life is okay. Screen is nice and bright. "
                "Overall I would recommend it if you're looking for a mid-range phone."
            ),
            "expected": "Medium",
            "description": "Basic product review"
        },
        {
            "text": (
                "Here's a quick recipe for pasta: Boil water, add pasta, cook for 10 minutes. "
                "Drain and add sauce. You can use any sauce you like. Serve with cheese. "
                "This is a simple meal that anyone can make."
            ),
            "expected": "Medium",
            "description": "Simple recipe/how-to"
        },
    ]
    
    print("\n" + "=" * 70)
    print("NVIDIA DeBERTa QUALITY CLASSIFIER - EXAMPLE TESTS")
    print("=" * 70)
    
    # Pre-load model
    load_model()
    
    results = {"High": 0, "Medium": 0, "Low": 0}
    correct = 0
    total = len(examples)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"[Example {i}/{total}] {example['description']}")
        print(f"Expected: {example['expected']}")
        print(f"{'='*70}")
        
        label, confidence, all_probs = predict_quality(example["text"])
        results[label] += 1
        
        if label == example["expected"]:
            correct += 1
            print(f"✅ CORRECT!")
        else:
            print(f"❌ MISMATCH - Expected {example['expected']}, got {label}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal examples: {total}")
    print(f"Correct predictions: {correct}/{total} ({correct/total:.1%})")
    print(f"\nDistribution:")
    print(f"  High:   {results['High']} ({results['High']/total:.1%})")
    print(f"  Medium: {results['Medium']} ({results['Medium']/total:.1%})")
    print(f"  Low:    {results['Low']} ({results['Low']/total:.1%})")
    print(f"\nNote: Expected results are subjective. The model's judgment may differ.")


def test_from_file(file_path: str):
    """Test model on texts from a file (one text per line)."""
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nReading texts from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts to classify\n")
    
    # Pre-load model
    load_model()
    
    results = {"High": 0, "Medium": 0, "Low": 0}
    predictions = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}]")
        label, confidence, all_probs = predict_quality(text)
        results[label] += 1
        predictions.append({
            "text": text[:50],
            "label": label,
            "confidence": confidence,
            "all_probs": all_probs
        })
    
    # Summary
    total = len(texts)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal texts: {total}")
    print(f"\nQuality Distribution:")
    print(f"  High:   {results['High']:4d} ({results['High']/total:6.1%}) - KEEP")
    print(f"  Medium: {results['Medium']:4d} ({results['Medium']/total:6.1%}) - KEEP")
    print(f"  Low:    {results['Low']:4d} ({results['Low']/total:6.1%}) - DISCARD")
    
    keep_count = results['High'] + results['Medium']
    print(f"\nPipeline Result:")
    print(f"  Would KEEP:    {keep_count} texts ({keep_count/total:.1%})")
    print(f"  Would DISCARD: {results['Low']} texts ({results['Low']/total:.1%})")
    
    # Show examples by category
    print("\n" + "-" * 70)
    print("SAMPLE PREDICTIONS BY CATEGORY:")
    print("-" * 70)
    
    for category in ["High", "Medium", "Low"]:
        cat_preds = [p for p in predictions if p["label"] == category]
        if cat_preds:
            print(f"\n{category} Quality Examples:")
            for p in cat_preds[:3]:
                print(f"  [{p['confidence']:.3f}] {p['text']}...")


def compare_with_fasttext(text: str, fasttext_model_path: str):
    """Compare NVIDIA model prediction with FastText model."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"\nText: {text[:100]}{'...' if len(text) > 100 else ''}\n")
    
    # NVIDIA prediction
    print("NVIDIA DeBERTa Model:")
    nvidia_label, nvidia_conf, nvidia_probs = predict_quality(text, verbose=False)
    print(f"  Label: {nvidia_label}")
    print(f"  Confidence: {nvidia_conf:.4f}")
    print(f"  All probs: Low={nvidia_probs['Low']:.3f}, Medium={nvidia_probs['Medium']:.3f}, High={nvidia_probs['High']:.3f}")
    
    # FastText prediction (if available)
    try:
        import fasttext
        if Path(fasttext_model_path).exists():
            print(f"\nFastText Model ({fasttext_model_path}):")
            ft_model = fasttext.load_model(fasttext_model_path)
            clean_text = text.replace("\n", " ").replace("\r", " ")
            clean_text = " ".join(clean_text.split())
            labels, probs = ft_model.predict(clean_text)
            ft_label = labels[0].replace("__label__", "")
            ft_conf = probs[0]
            print(f"  Label: {ft_label}")
            print(f"  Confidence: {ft_conf:.4f}")
        else:
            print(f"\nFastText model not found at: {fasttext_model_path}")
    except ImportError:
        print("\nFastText not installed, skipping comparison")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA DeBERTa quality classifier model"
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
        help="Run in interactive mode (default if no other option)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with FastText model"
    )
    parser.add_argument(
        "--fasttext-model",
        type=str,
        default="quality_classifier.bin",
        help="Path to FastText model for comparison"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (use if CUDA errors occur)"
    )
    
    args = parser.parse_args()
    
    # Pre-load model with CPU flag if specified
    if args.cpu:
        load_model(force_cpu=True)
    
    # Determine mode
    if args.text:
        if args.compare:
            compare_with_fasttext(args.text, args.fasttext_model)
        else:
            predict_quality(args.text)
    elif args.file:
        test_from_file(args.file)
    elif args.examples:
        run_examples()
    else:
        # Default to interactive mode
        run_interactive()
    
    return 0


if __name__ == "__main__":
    exit(main())
