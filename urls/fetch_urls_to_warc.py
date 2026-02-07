#!/usr/bin/env python3
"""
Fetch URLs and save to WARC format with progress tracking.

Replicates the functionality of:
wget --tries=1 --dns-timeout=2 --connect-timeout=2 --read-timeout=3 \
     --timeout=3 --max-redirect=3 --inet4-only \
     -i subsampled_positive_urls.txt --warc-file=subsampled_positive_urls -O /dev/null

Usage:
    python fetch_urls_to_warc.py --input subsampled_positive_urls.txt --output subsampled_positive_urls
    python fetch_urls_to_warc.py --input urls.txt --output output --num-urls 100
"""

import argparse
import subprocess
import re
from pathlib import Path
from collections import Counter

from tqdm import tqdm


def fetch_urls_to_warc(
    input_file: str,
    output_warc: str,
    num_urls: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Fetch URLs using wget and save to WARC format with progress tracking.
    
    Args:
        input_file: Path to file containing URLs (one per line)
        output_warc: Output WARC file name (without .warc.gz extension)
        num_urls: Maximum number of URLs to fetch (None for all)
        verbose: Print verbose output
        
    Returns:
        Dictionary with statistics about the fetch operation
    """
    # Read URLs from file
    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if num_urls is not None:
        urls = urls[:num_urls]
    
    total_urls = len(urls)
    print(f"Total URLs to fetch: {total_urls}")
    
    if total_urls == 0:
        print("No URLs to fetch")
        return {"total": 0, "success": 0, "failed": 0, "status_codes": {}}
    
    # Create a temporary file with the (possibly limited) URLs
    temp_url_file = f"{output_warc}_temp_urls.txt"
    with open(temp_url_file, "w", encoding="utf-8") as f:
        f.write("\n".join(urls))
    
    # Build wget command matching the original parameters
    cmd = [
        "wget",
        "--tries=1",
        "--dns-timeout=2",
        "--connect-timeout=2",
        "--read-timeout=3",
        "--timeout=3",
        "--max-redirect=3",
        "--inet4-only",
        "-i", temp_url_file,
        f"--warc-file={output_warc}",
        "-O", "/dev/null",
        "--no-check-certificate",
        "--server-response",  # To capture HTTP status codes
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"\nFetching {total_urls} URLs to {output_warc}.warc.gz\n")
    
    # Statistics tracking
    status_codes = Counter()
    success_count = 0
    failed_count = 0
    current_url = ""
    
    # Progress bar
    pbar = tqdm(total=total_urls, desc="Fetching URLs", unit="url")
    
    try:
        # Run wget with real-time output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Regex patterns for parsing wget output
        url_pattern = re.compile(r"--\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}--  (https?://\S+)")
        status_pattern = re.compile(r"HTTP/[\d.]+ (\d{3})")
        error_patterns = [
            "failed:",
            "ERROR",
            "Unable to",
            "Giving up",
            "Connection timed out",
            "Connection refused",
            "No route to host",
            "Name or service not known",
        ]
        
        for line in process.stdout:
            line = line.strip()
            
            if verbose:
                print(line)
            
            # Check for URL being fetched
            url_match = url_pattern.search(line)
            if url_match:
                current_url = url_match.group(1)
            
            # Check for HTTP status code
            status_match = status_pattern.search(line)
            if status_match:
                status_code = int(status_match.group(1))
                status_codes[status_code] += 1
                
                if 200 <= status_code < 400:
                    success_count += 1
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "200": status_codes.get(200, 0),
                    "OK": success_count,
                    "Fail": failed_count
                })
            
            # Check for error patterns (for URLs that don't get HTTP response)
            if any(err in line for err in error_patterns):
                if "HTTP" not in line:  # Don't double count HTTP errors
                    # Check if this is a new failure (not already counted)
                    if "failed:" in line.lower() or "giving up" in line.lower():
                        failed_count += 1
                        status_codes["connection_error"] += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "200": status_codes.get(200, 0),
                            "OK": success_count,
                            "Fail": failed_count
                        })
        
        pbar.close()
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        process.terminate()
        pbar.close()
    
    finally:
        # Clean up temp file
        Path(temp_url_file).unlink(missing_ok=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    print(f"\nTotal URLs attempted: {total_urls}")
    print(f"Successful (2xx/3xx): {success_count}")
    print(f"Failed:               {failed_count}")
    print(f"Success rate:         {success_count/total_urls*100:.1f}%")
    
    print(f"\nStatus Code Distribution:")
    for code, count in sorted(status_codes.items(), key=lambda x: (-x[1], str(x[0]))):
        if code == 200:
            print(f"  {code}: {count} ✓")
        elif isinstance(code, int) and 200 <= code < 400:
            print(f"  {code}: {count}")
        else:
            print(f"  {code}: {count} ✗")
    
    # Check WARC file
    warc_path = Path(f"{output_warc}.warc.gz")
    if warc_path.exists():
        size_mb = warc_path.stat().st_size / (1024 * 1024)
        print(f"\nWARC file created: {warc_path} ({size_mb:.2f} MB)")
    else:
        print(f"\nWarning: WARC file not found at {warc_path}")
    
    print("=" * 60)
    
    return {
        "total": total_urls,
        "success": success_count,
        "failed": failed_count,
        "status_codes": dict(status_codes),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch URLs and save to WARC format with progress tracking"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input file containing URLs (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output WARC file name (without .warc.gz extension)"
    )
    parser.add_argument(
        "--num-urls", "-n",
        type=int,
        default=None,
        help="Maximum number of URLs to fetch (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose wget output"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Fetch URLs
    stats = fetch_urls_to_warc(
        input_file=args.input,
        output_warc=args.output,
        num_urls=args.num_urls,
        verbose=args.verbose,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
