#!/usr/bin/env python3
"""
Download URLs from CDX file and store output in WARC format.

Usage:
    python download_urls_warc.py --input urls/cdx-00000 --output output.warc --num-urls 100
    python download_urls_warc.py --input urls/cdx-00000 --output output.warc  # All URLs
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def extract_urls_from_cdx(cdx_path: str, num_urls: int | None = None) -> list[str]:
    """
    Extract URLs from a CDX file.
    
    Args:
        cdx_path: Path to the CDX file
        num_urls: Maximum number of URLs to extract (None for all)
        
    Returns:
        List of URLs
    """
    urls = []
    
    # Count total lines for progress bar if processing all
    if num_urls is None:
        with open(cdx_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = num_urls
    
    with open(cdx_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Extracting URLs", unit="urls")):
            if num_urls is not None and i >= num_urls:
                break
            
            try:
                # CDX format: key timestamp {json}
                # Find the JSON part (starts with {)
                json_start = line.find("{")
                if json_start == -1:
                    continue
                    
                json_str = line[json_start:].strip()
                data = json.loads(json_str)
                
                if "url" in data:
                    urls.append(data["url"])
            except json.JSONDecodeError:
                continue
    
    return urls


def download_urls_to_warc(
    urls: list[str],
    output_warc: str,
    timeout: int = 5,
    verbose: bool = False,
    batch_size: int = 100,
) -> None:
    """
    Download URLs using wget and save to WARC format.
    
    Args:
        urls: List of URLs to download
        output_warc: Output WARC file path (without .warc.gz extension)
        timeout: Timeout in seconds for each request
        verbose: Print verbose output
        batch_size: Number of URLs to download per batch (for progress tracking)
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
        description="Download URLs from CDX file and store in WARC format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CDX file containing URLs"
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
        help="Number of URLs to download (default: all)"
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
    parser.add_argument(
        "--filter-status",
        type=str,
        default=None,
        help="Only include URLs with this HTTP status (e.g., '200')"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Reading URLs from: {args.input}")
    
    # Extract URLs
    urls = extract_urls_from_cdx(args.input, args.num_urls)
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
