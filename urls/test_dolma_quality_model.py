
"""
Interactive tool to test the Allen AI Dolma3 FastText quality classifier model.

This uses the allenai/dolma3-fasttext-quality-classifier model which classifies text
into two quality categories: High (hq) and Low (lq).

Model: https://huggingface.co/allenai/dolma3-fasttext-quality-classifier

Usage:
    # Download the model first (4.3GB)
    python test_dolma_quality_model.py --download
    
    # Interactive mode - enter texts manually
    python test_dolma_quality_model.py
    
    # Test with a specific text
    python test_dolma_quality_model.py --text "Your text here"
    
    # Test with texts from a file (one per line)
    python test_dolma_quality_model.py --file texts_to_test.txt
    
    # Run built-in test examples
    python test_dolma_quality_model.py --examples
    
    # Check system resources
    python test_dolma_quality_model.py --check-resources
"""

import argparse
import os
import sys
import resource
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Global model cache
_model = None

# Default model paths to search
MODEL_SEARCH_PATHS = [
    "models/dolma3_quality_classifier.bin",
    "models/model.bin",
    "dolma3_quality_classifier.bin",
    "model.bin",
    str(Path(__file__).parent.parent / "models" / "dolma3_quality_classifier.bin"),
    str(Path(__file__).parent.parent / "models" / "model.bin"),
    str(Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--dolma3-fasttext-quality-classifier" / "snapshots"),
]

# Label mapping
LABEL_MAP = {
    "__label__hq": "High",
    "__label__lq": "Low",
}


def check_resources():
    """Check system resources and limits."""
    print("=" * 70)
    print("SYSTEM RESOURCE CHECK")
    print("=" * 70)
    
    # Check memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\nMemory:")
        print(f"  Total:     {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used:      {mem.percent}%")
    except ImportError:
        print("\nMemory: (install psutil for details)")
        # Fallback to /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line or 'MemAvailable' in line or 'MemFree' in line:
                        print(f"  {line.strip()}")
        except:
            print("  Unable to read memory info")
    
    # Check ulimits
    print(f"\nResource Limits (ulimit):")
    limits = [
        ("RLIMIT_AS", resource.RLIMIT_AS, "Address space"),
        ("RLIMIT_DATA", resource.RLIMIT_DATA, "Data segment"),
        ("RLIMIT_RSS", resource.RLIMIT_RSS, "Resident set size"),
        ("RLIMIT_MEMLOCK", resource.RLIMIT_MEMLOCK, "Locked memory"),
        ("RLIMIT_NOFILE", resource.RLIMIT_NOFILE, "Open files"),
    ]
    
    for name, limit_type, desc in limits:
        try:
            soft, hard = resource.getrlimit(limit_type)
            soft_str = f"{soft / (1024**3):.1f} GB" if soft != resource.RLIM_INFINITY else "unlimited"
            hard_str = f"{hard / (1024**3):.1f} GB" if hard != resource.RLIM_INFINITY else "unlimited"
            print(f"  {desc:20s}: soft={soft_str}, hard={hard_str}")
        except Exception as e:
            print(f"  {desc:20s}: Error - {e}")
    
    # Check model file
    model_path = find_model()
    print(f"\nModel:")
    if model_path:
        size_gb = Path(model_path).stat().st_size / (1024**3)
        print(f"  Path: {model_path}")
        print(f"  Size: {size_gb:.2f} GB")
        print(f"\n  ⚠️  Loading this model requires ~{size_gb * 2:.1f} GB RAM")
    else:
        print("  Not found!")
    
    # Check if running in SLURM
    print(f"\nCluster Environment:")
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_PER_TASK']
    for var in slurm_vars:
        val = os.environ.get(var, 'not set')
        print(f"  {var}: {val}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
If the model loading is being killed:

1. Request more memory in your SLURM job:
   srun --mem=16G --time=01:00:00 --pty bash
   
2. Or use a batch script with memory allocation:
   #SBATCH --mem=16G
   
3. Or try the smaller quantized model (if available)

4. Check your cluster's memory limits:
   ulimit -a
   
5. Monitor memory during loading:
   watch -n 1 free -h
""")
    print("=" * 70)


def find_model() -> str | None:
    """Find the model file in common locations."""
    for path in MODEL_SEARCH_PATHS:
        p = Path(path)
        if p.exists():
            if p.is_file():
                return str(p)
            elif p.is_dir():
                # Check for model.bin in directory
                model_file = p / "model.bin"
                if model_file.exists():
                    return str(model_file)
                # Check subdirectories (for HF cache)
                for subdir in p.iterdir():
                    if subdir.is_dir():
                        model_file = subdir / "model.bin"
                        if model_file.exists():
                            return str(model_file)
    return None


def download_model(output_dir: str = "models") -> str:
    """Download the Dolma3 model from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Downloading Dolma3 quality classifier from Hugging Face...")
    print("Model: allenai/dolma3-fasttext-quality-classifier")
    print("Size: ~4.3GB - this may take a while...")
    print()
    
    model_path = hf_hub_download(
        repo_id="allenai/dolma3-fasttext-quality-classifier",
        filename="model.bin",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"\nModel downloaded to: {model_path}")
    return model_path


def load_model(model_path: str = None):
    """Load the Dolma3 FastText model with progress tracking."""
    global _model
    
    if _model is not None:
        return _model
    
    if model_path is None:
        model_path = find_model()
    
    if model_path is None or not Path(model_path).exists():
        print("=" * 60)
        print("ERROR: Dolma3 model not found!")
        print("=" * 60)
        print("\nSearched in:")
        for p in MODEL_SEARCH_PATHS:
            print(f"  - {p}")
        print("\nTo download the model (~4.3GB), run:")
        print("  python test_dolma_quality_model.py --download")
        print("\nOr download manually:")
        print("  mkdir -p models")
        print("  wget https://huggingface.co/allenai/dolma3-fasttext-quality-classifier/resolve/main/model.bin -O models/model.bin")
        print("=" * 60)
        sys.exit(1)
    
    model_size = Path(model_path).stat().st_size / (1024**3)
    print(f"Loading model from: {model_path}")
    print(f"Model size: {model_size:.2f} GB")
    print(f"Estimated RAM needed: ~{model_size * 2:.1f} GB")
    print("Loading... (this may take 30-60 seconds)")
    sys.stdout.flush()
    
    try:
        import fasttext
        # Suppress FastText warnings
        fasttext.FastText.elideLevels = 0
        
        _model = fasttext.load_model(model_path)
        print("✅ Model loaded successfully!\n")
        return _model
        
    except MemoryError:
        print("\n❌ MemoryError: Not enough RAM to load the model")
        print("   Try requesting more memory: srun --mem=16G ...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)


def predict_quality(text: str, verbose: bool = True) -> tuple[str, float, dict]:
    """
    Predict the quality of a text using Dolma3 FastText model.
    
    Args:
        text: Text to classify
        verbose: Whether to print results
        
    Returns:
        Tuple of (label, confidence, all_probabilities)
    """
    model = load_model()
    
    # Clean and prepare text
    clean_text = text.replace("\n", " ").replace("\r", " ")
    clean_text = " ".join(clean_text.split())
    
    # Get predictions for both classes
    labels, probs = model.predict(clean_text, k=2)
    
    primary_label = LABEL_MAP.get(labels[0], labels[0])
    primary_prob = probs[0]
    
    # Build all probabilities dict
    all_probs = {"High": 0.0, "Low": 0.0}
    for label, prob in zip(labels, probs):
        mapped = LABEL_MAP.get(label, label)
        all_probs[mapped] = float(prob)
    
    if verbose:
        print("-" * 70)
        print(f"TEXT: {clean_text[:200]}{'...' if len(clean_text) > 200 else ''}")
        print(f"\n{'='*30} PREDICTION {'='*30}")
        print(f"\n  Quality Level: {primary_label.upper()}")
        print(f"  Confidence:    {primary_prob:.4f} ({primary_prob:.2%})")
        print(f"\n  All Probabilities:")
        print(f"    High Quality: {all_probs['High']:.4f} ({all_probs['High']:.2%})")
        print(f"    Low Quality:  {all_probs['Low']:.4f} ({all_probs['Low']:.2%})")
        
        # Quality interpretation
        if primary_label == "High":
            print(f"\n  ✅ HIGH QUALITY - Good content, keep in dataset")
            print(f"\n  📋 Pipeline: KEEP")
        else:
            print(f"\n  ❌ LOW QUALITY - Poor content, filter out")
            print(f"\n  📋 Pipeline: DISCARD")
        
        print("-" * 70)
    
    return primary_label, primary_prob, all_probs


def run_interactive():
    """Run interactive mode where user can input texts."""
    print("\n" + "=" * 70)
    print("DOLMA3 FASTTEXT QUALITY CLASSIFIER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nEnter text to classify. Commands:")
    print("  'quit' or 'q' - Exit")
    print("  'multi' - Enter multi-line text (end with 'END' on its own line)")
    print("=" * 70 + "\n")
    
    # Pre-load model
    load_model()
    
    while True:
        try:
            user_input = input("\n📝 Enter text: ").strip()
            
            if user_input.lower() in ('quit', 'q'):
                print("Goodbye!")
                break
            
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
    ]
    
    print("\n" + "=" * 70)
    print("DOLMA3 FASTTEXT QUALITY CLASSIFIER - EXAMPLE TESTS")
    print("=" * 70)
    
    # Pre-load model
    load_model()
    
    results = {"High": 0, "Low": 0}
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
            print(f"⚠️  DIFFERENT - Expected {example['expected']}, got {label}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal examples: {total}")
    print(f"Matching expected: {correct}/{total} ({correct/total:.1%})")
    print(f"\nDistribution:")
    print(f"  High Quality: {results['High']} ({results['High']/total:.1%})")
    print(f"  Low Quality:  {results['Low']} ({results['Low']/total:.1%})")


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
    
    results = {"High": 0, "Low": 0}
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}]")
        label, confidence, all_probs = predict_quality(text)
        results[label] += 1
    
    # Summary
    total = len(texts)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal texts: {total}")
    print(f"\nQuality Distribution:")
    print(f"  High Quality: {results['High']:4d} ({results['High']/total:6.1%}) - KEEP")
    print(f"  Low Quality:  {results['Low']:4d} ({results['Low']/total:6.1%}) - DISCARD")


def main():
    parser = argparse.ArgumentParser(
        description="Test Dolma3 FastText quality classifier model"
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
        "--download",
        action="store_true",
        help="Download the model from Hugging Face (~4.3GB)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model file (optional, will search common locations)"
    )
    parser.add_argument(
        "--check-resources",
        action="store_true",
        help="Check system resources and limits"
    )
    
    args = parser.parse_args()
    
    # Handle resource check
    if args.check_resources:
        check_resources()
        return 0
    
    # Handle download
    if args.download:
        download_model()
        print("\nModel ready! You can now run tests:")
        print("  python test_dolma_quality_model.py --examples")
        return 0
    
    # Set model path if specified
    if args.model:
        global MODEL_SEARCH_PATHS
        MODEL_SEARCH_PATHS = [args.model] + MODEL_SEARCH_PATHS
    
    # Determine mode
    if args.text:
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