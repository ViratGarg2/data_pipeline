#!/usr/bin/env python3
"""
Train a FastText classifier to distinguish between high-quality (positive) and low-quality (negative) text.

Usage:
    python train_quality_classifier.py --positive positive.txt --negative negative.txt --output quality_model
    python train_quality_classifier.py --positive positive.txt --negative negative.txt --output quality_model --positive-samples 500
    
Input format (FastText labeled format):
    __label__positive This is a high quality article...
    __label__negative Buy cheap products now click here...
"""

import argparse
import random
import tempfile
from pathlib import Path

import fasttext


def load_samples(file_path: str, label: str | None = None) -> list[str]:
    """
    Load samples from a FastText format file.
    
    Args:
        file_path: Path to the file
        label: Optional label to filter by (e.g., 'positive', 'negative')
        
    Returns:
        List of lines (samples)
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # If label filter specified, check if line has that label
            if label:
                if line.startswith(f"__label__{label}"):
                    samples.append(line)
            else:
                samples.append(line)
    
    return samples


def prepare_training_data(
    positive_file: str,
    negative_file: str,
    positive_samples: int | None = None,
    negative_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Prepare training data by loading and sampling from positive and negative files.
    
    Args:
        positive_file: Path to positive samples file
        negative_file: Path to negative samples file
        positive_samples: Number of positive samples to use (None for all)
        negative_samples: Number of negative samples to use (None for all)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, all_samples_before_split)
    """
    random.seed(seed)
    
    # Load samples
    print(f"Loading positive samples from: {positive_file}")
    positive = load_samples(positive_file)
    print(f"  Found {len(positive)} positive samples")
    
    print(f"Loading negative samples from: {negative_file}")
    negative = load_samples(negative_file)
    print(f"  Found {len(negative)} negative samples")
    
    # Sample if needed
    if positive_samples is not None and positive_samples < len(positive):
        print(f"  Randomly sampling {positive_samples} positive samples")
        positive = random.sample(positive, positive_samples)
    
    if negative_samples is not None and negative_samples < len(negative):
        print(f"  Randomly sampling {negative_samples} negative samples")
        negative = random.sample(negative, negative_samples)
    
    # Combine and shuffle
    all_samples = positive + negative
    random.shuffle(all_samples)
    
    print(f"\nTotal training samples: {len(all_samples)}")
    print(f"  Positive: {len(positive)}")
    print(f"  Negative: {len(negative)}")
    
    return all_samples, (positive, negative)


def train_fasttext_classifier(
    train_samples: list[str],
    output_model: str,
    lr: float = 0.1,
    epoch: int = 25,
    word_ngrams: int = 2,
    dim: int = 100,
    min_count: int = 1,
    verbose: int = 2,
) -> fasttext.FastText._FastText:
    """
    Train a FastText supervised classifier.
    
    Args:
        train_samples: List of labeled training samples
        output_model: Path to save the model (without extension)
        lr: Learning rate
        epoch: Number of training epochs
        word_ngrams: Max length of word n-grams
        dim: Dimension of word vectors
        min_count: Minimum word count
        verbose: Verbosity level
        
    Returns:
        Trained FastText model
    """
    # Write training data to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for sample in train_samples:
            f.write(sample + "\n")
        train_file = f.name
    
    print(f"\nTraining FastText classifier...")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epoch}")
    print(f"  Word n-grams: {word_ngrams}")
    print(f"  Dimensions: {dim}")
    
    try:
        # Train the model
        model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            dim=dim,
            minCount=min_count,
            verbose=verbose,
        )
        
        # Save the model
        model.save_model(f"{output_model}.bin")
        print(f"\nModel saved to: {output_model}.bin")
        
        return model
        
    finally:
        # Clean up temp file
        Path(train_file).unlink(missing_ok=True)


def evaluate_model(
    model: fasttext.FastText._FastText,
    positive_samples: list[str],
    negative_samples: list[str],
) -> dict:
    """
    Evaluate the trained model on the training data.
    
    Args:
        model: Trained FastText model
        positive_samples: List of positive samples
        negative_samples: List of negative samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Test on positive samples
    pos_correct = 0
    for sample in positive_samples:
        # Remove label from sample for prediction
        text = sample.replace("__label__positive ", "").replace("__label__negative ", "")
        pred_label, pred_prob = model.predict(text)
        if pred_label[0] == "__label__positive":
            pos_correct += 1
    
    # Test on negative samples
    neg_correct = 0
    for sample in negative_samples:
        text = sample.replace("__label__positive ", "").replace("__label__negative ", "")
        pred_label, pred_prob = model.predict(text)
        if pred_label[0] == "__label__negative":
            neg_correct += 1
    
    pos_accuracy = pos_correct / len(positive_samples) if positive_samples else 0
    neg_accuracy = neg_correct / len(negative_samples) if negative_samples else 0
    total_correct = pos_correct + neg_correct
    total_samples = len(positive_samples) + len(negative_samples)
    overall_accuracy = total_correct / total_samples if total_samples else 0
    
    metrics = {
        "positive_accuracy": pos_accuracy,
        "negative_accuracy": neg_accuracy,
        "overall_accuracy": overall_accuracy,
        "positive_correct": pos_correct,
        "positive_total": len(positive_samples),
        "negative_correct": neg_correct,
        "negative_total": len(negative_samples),
    }
    
    print(f"\nTraining Set Evaluation:")
    print(f"  Positive accuracy: {pos_correct}/{len(positive_samples)} = {pos_accuracy:.2%}")
    print(f"  Negative accuracy: {neg_correct}/{len(negative_samples)} = {neg_accuracy:.2%}")
    print(f"  Overall accuracy:  {total_correct}/{total_samples} = {overall_accuracy:.2%}")
    
    return metrics


def test_predictions(model: fasttext.FastText._FastText, texts: list[str]):
    """
    Test the model on a few sample texts and print predictions.
    """
    print("\nSample predictions:")
    print("-" * 60)
    
    for text in texts:
        # Truncate for display
        display_text = text[:80] + "..." if len(text) > 80 else text
        pred_label, pred_prob = model.predict(text)
        label = pred_label[0].replace("__label__", "")
        prob = pred_prob[0]
        print(f"  [{label}] ({prob:.3f}): {display_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a FastText quality classifier"
    )
    parser.add_argument(
        "--positive", "-p",
        type=str,
        required=True,
        help="Path to positive samples file (FastText format)"
    )
    parser.add_argument(
        "--negative", "-n",
        type=str,
        required=True,
        help="Path to negative samples file (FastText format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="quality_classifier",
        help="Output model path (without .bin extension). Default: quality_classifier"
    )
    parser.add_argument(
        "--positive-samples",
        type=int,
        default=None,
        help="Number of positive samples to use (default: 500)"
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=None,
        help="Number of negative samples to use (default: all)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=200,
        help="Dimension of word vectors (default: 200)"
    )
    parser.add_argument(
        "--word-ngrams",
        type=int,
        default=2,
        help="Max length of word n-grams (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.positive).exists():
        print(f"Error: Positive samples file not found: {args.positive}")
        return 1
    
    if not Path(args.negative).exists():
        print(f"Error: Negative samples file not found: {args.negative}")
        return 1
    
    print("=" * 60)
    print("FastText Quality Classifier Training")
    print("=" * 60)
    
    # Prepare training data
    train_samples, (positive, negative) = prepare_training_data(
        positive_file=args.positive,
        negative_file=args.negative,
        positive_samples=args.positive_samples,
        negative_samples=args.negative_samples,
        seed=args.seed,
    )
    
    # Train model
    model = train_fasttext_classifier(
        train_samples=train_samples,
        output_model=args.output,
        lr=args.lr,
        epoch=args.epoch,
        word_ngrams=args.word_ngrams,
        dim=args.dim,
    )
    
    # Evaluate model
    evaluate_model(model, positive, negative)
    
    # Test with some sample texts
    test_texts = [
        "This is a well-written article about machine learning and artificial intelligence.",
        "BUY NOW CLICK HERE FREE MONEY BEST DEALS LIMITED TIME OFFER!!!",
        "The history of philosophy begins with the ancient Greeks who sought to understand the nature of reality.",
        "asdfasdf jkljkl random text spam spam spam click click click",
    ]
    test_predictions(model, test_texts)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to: {args.output}.bin")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
