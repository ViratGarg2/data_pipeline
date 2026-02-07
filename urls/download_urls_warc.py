#!/usr/bin/env python3
"""
Download URLs from positive_repo.txt file and store output in WARC format.

Usage:
    python download_urls_warc.py --input urls/positive_repo.txt --output output.warc --num-urls 100
    python download_urls_warc.py --input urls/positive_repo.txt --output output.warc  # Up to 1000 URLs
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


MAX_LINES_PER_FILE = 1000  # Maximum number of lines to read from each file


def extract_urls_from_file(file_path: str, num_urls: int | None = None) -> list[str]:
    """
    Extract URLs from a text file (one URL per line).
    
    Args:
        file_path: Path to the input file
        num_urls: Maximum number of URLs to extract (None for all, but capped at MAX_LINES_PER_FILE)
        
    Returns:
        List of URLs
    """
    urls = []
    
    # Cap the number of URLs to MAX_LINES_PER_FILE
    if num_urls is None:
        num_urls = MAX_LINES_PER_FILE
    else:
        num_urls = min(num_urls, MAX_LINES_PER_FILE)
    
    total_lines = num_urls
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Extracting URLs", unit="urls")):
            if i >= num_urls:
                break
            
            url = line.strip()
            if url:  # Skip empty lines
                urls.append(url)
    
    return urls


def download_urls_to_warc(
    urls: list[str],
    output_warc: str,
    timeout: int = 5,
    verbose: bool = False,
) -> None:
    """
    Download URLs using wget and save to WARC format.
    
    Args:
        urls: List of URLs to download
        output_warc: Output WARC file path (without .warc.gz extension)
        timeout: Timeout in seconds for each request
        verbose: Print verbose output
    """
    if not urls:
        print("No URLs to download")
        return
    
    print(f"Downloading {len(urls)} URLs to {output_warc}.warc.gz")
    
    # Create a temporary file with URLs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as url_file:
        url_file.write("\n".join(urls))
        url_file_path = url_file.name
    
    try:
        # Build wget command
        cmd = [
            "wget",
            f"--timeout={timeout}",
            "--tries=1",  # Only try once per URL
            "-i", url_file_path,  # Input file with URLs
            f"--warc-file={output_warc}",  # Output WARC file
            "-O", "/dev/null",  # Don't save individual files
            "--no-check-certificate",  # Skip SSL verification issues
            "--progress=dot:mega" if not verbose else "",  # Show progress dots
        ]
        
        # Remove empty strings from command
        cmd = [c for c in cmd if c]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Downloading {len(urls)} URLs...")
        
        # Run wget with real-time output for progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Track progress by counting completed URLs in output
        completed = 0
        pbar = tqdm(total=len(urls), desc="Downloading", unit="urls")
        
        for line in process.stdout:
            if verbose:
                print(line, end="")
            # wget outputs "Saving to:" or similar for each URL
            if "Saving to:" in line or "saved" in line.lower() or "ERROR" in line or "failed" in line.lower():
                completed += 1
                pbar.update(1)
        
        pbar.close()
        process.wait()
        
        if process.returncode != 0 and process.returncode != 8:
            # Return code 8 means some URLs failed, which is expected
            print(f"wget finished with return code: {process.returncode}")
        
        # Check if WARC file was created
        warc_path = Path(f"{output_warc}.warc.gz")
        if warc_path.exists():
            size_mb = warc_path.stat().st_size / (1024 * 1024)
            print(f"Successfully created {warc_path} ({size_mb:.2f} MB)")
        else:
            print(f"Warning: WARC file not found at {warc_path}")
            
    finally:
        # Clean up temp file
        Path(url_file_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download URLs from a text file and store in WARC format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input text file containing URLs (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="downloaded_urls",
        help="Output WARC file name (without extension). Default: downloaded_urls"
    )
    parser.add_argument(
        "--num-urls", "-n",
        type=int,
        default=None,
        help="Number of URLs to download (default: up to 1000)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=5,
        help="Timeout in seconds for each request (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--filter-robots",
        action="store_true",
        help="Filter out robots.txt URLs"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Reading URLs from: {args.input}")
    
    # Extract URLs
    urls = extract_urls_from_file(args.input, args.num_urls)
    print(f"Found {len(urls)} URLs")
    
    # Apply filters
    if args.filter_robots:
        original_count = len(urls)
        urls = [u for u in urls if not u.endswith("/robots.txt")]
        print(f"Filtered out {original_count - len(urls)} robots.txt URLs, {len(urls)} remaining")
    
    # Download to WARC
    download_urls_to_warc(
        urls=urls,
        output_warc=args.output,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())


if __name__ == "__main__":
    exit(main())
