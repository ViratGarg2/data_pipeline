#!/usr/bin/env python3
"""
Read and extract content from WARC files.

Usage:
    # Preview first 5 records (raw HTML)
    python read_warc.py --input downloaded_urls.warc.gz --num-records 5
    
    # Extract plain text and preview
    python read_warc.py --input downloaded_urls.warc.gz --extract-text -n 5
    
    # Save as JSONL
    python read_warc.py --input downloaded_urls.warc.gz -e --output extracted.jsonl
    
    # Save as FastText format with label (for training classifier)
    python read_warc.py --input positive_urls.warc.gz -e --output positive.txt --label positive
    python read_warc.py --input negative_urls.warc.gz -e --output negative.txt --label negative
    
    # Filter to keep only English texts (useful for negative samples)
    python read_warc.py --input negative_urls.warc.gz -e --output negative.txt --label negative --english-only
    
    # Then combine for FastText training:
    # cat positive.txt negative.txt | shuf > train.txt
    # fasttext supervised -input train.txt -output model
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Generator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language


def parse_warc_records(warc_path: str) -> Generator[dict, None, None]:
    """
    Parse WARC file and yield records with headers and content.
    
    Args:
        warc_path: Path to WARC file (.warc or .warc.gz)
        
    Yields:
        Dictionary with warc_type, url, headers, and content
    """
    path = Path(warc_path)
    
    # Open with gzip if compressed
    if path.suffix == ".gz" or str(path).endswith(".warc.gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")
    
    try:
        while True:
            # Read WARC header
            warc_headers = {}
            
            # First line should be WARC/1.0
            line = f.readline()
            if not line:
                break  # End of file
            
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("WARC/"):
                continue
            
            # Read WARC headers until empty line
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    break  # Empty line marks end of headers
                
                if ":" in line:
                    key, value = line.split(":", 1)
                    warc_headers[key.strip()] = value.strip()
            
            # Get content length
            content_length = int(warc_headers.get("Content-Length", 0))
            
            # Read the content block
            content_bytes = f.read(content_length) if content_length > 0 else b""
            
            # Skip trailing newlines between records
            f.readline()  # Usually \r\n
            f.readline()  # Usually \r\n
            
            # Parse the content based on WARC type
            warc_type = warc_headers.get("WARC-Type", "")
            target_uri = warc_headers.get("WARC-Target-URI", "").strip("<>")
            
            if warc_type == "response" and content_bytes:
                # HTTP response: headers + body
                # Find the separator between HTTP headers and body
                separator_idx = content_bytes.find(b"\r\n\r\n")
                if separator_idx == -1:
                    separator_idx = content_bytes.find(b"\n\n")
                
                if separator_idx != -1:
                    http_headers = content_bytes[:separator_idx].decode("utf-8", errors="replace")
                    body = content_bytes[separator_idx + 4:]  # Skip \r\n\r\n
                    
                    # Try to decode body as text
                    try:
                        body_text = body.decode("utf-8", errors="replace")
                    except:
                        body_text = f"[Binary content: {len(body)} bytes]"
                else:
                    http_headers = ""
                    body = content_bytes
                    body_text = content_bytes.decode("utf-8", errors="replace")
                    body_text = body_text[:1000]
                
                yield {
                    "warc_type": warc_type,
                    "url": target_uri,
                    "warc_headers": warc_headers,
                    "http_headers": http_headers,
                    "content": body_text,
                    "content_bytes": body,  # Raw bytes for text extraction
                    "content_length": len(body_text),
                }
            
            elif warc_type == "request":
                # HTTP request - usually not needed for content extraction
                yield {
                    "warc_type": warc_type,
                    "url": target_uri,
                    "warc_headers": warc_headers,
                    "content": content_bytes.decode("utf-8", errors="replace"),
                    "content_length": content_length,
                }
            
            elif warc_type == "warcinfo":
                # WARC metadata
                yield {
                    "warc_type": warc_type,
                    "url": "",
                    "warc_headers": warc_headers,
                    "content": content_bytes.decode("utf-8", errors="replace"),
                    "content_length": content_length,
                }
    
    finally:
        f.close()


def extract_html_content(
    warc_path: str, 
    output_path: str | None = None, 
    num_records: int | None = None,
    extract_text: bool = False,
    label: str | None = None,
    english_only: bool = False,
) -> list[dict]:
    """
    Extract HTML content from WARC file.
    
    Args:
        warc_path: Path to WARC file
        output_path: Optional path to save as JSONL or FastText format
        num_records: Maximum number of response records to extract
        extract_text: If True, extract plain text from HTML using resiliparse
        label: Optional label for FastText format (e.g., "positive" or "negative")
        english_only: If True, only keep English texts (using langid)
        
    Returns:
        List of extracted records
    """
    records = []
    response_count = 0
    filtered_non_english = 0
    
    print(f"Reading WARC file: {warc_path}")
    if extract_text:
        print("Text extraction enabled (using resiliparse)")
    if english_only:
        print("English-only filter enabled (using langid)")
    print("-" * 80)
    
    for record in parse_warc_records(warc_path):
        if record["warc_type"] == "response":
            response_count += 1
            
            if num_records is not None and response_count > num_records:
                break
            
            has_content_bytes = "content_bytes" in record
            
            # Get content - either extracted text or raw HTML
            if extract_text and has_content_bytes:
                # Use resiliparse to extract plain text from HTML
                raw_bytes = record["content_bytes"]
                extracted = extract_text_from_html_bytes(raw_bytes)
                
                if extracted and extracted.strip():
                    content = extracted.strip()
                    content_type = "extracted_text"
                else:
                    # Skip empty extractions in extract_text mode
                    continue
            else:
                content = record["content"]
                content_type = "raw_html"
            
            # Filter non-English texts if requested
            if english_only and content:
                lang_code, lang_score = identify_language(content)
                if lang_code != "en":
                    filtered_non_english += 1
                    continue  # Skip non-English texts
            
            records.append({
                "url": record["url"],
                "content": content,
                "content_type": content_type,
                "content_length": len(content) if content else 0,
                "label": label,
            })
            
            # Print output based on mode
            if extract_text:
                # Clean output: just show the full text
                print(f"\n[{response_count}] URL: {record['url']}")
                print(f"{'-' * 40}")
                # print(content)
                print(f"{'-' * 40}\n")
            else:
                # Verbose output with metadata
                if content:
                    content_preview = content[:500].replace("\n", " ")
                    print(f"\n[{response_count}] URL: {record['url']}")
                    print(f"    Content type: {content_type}")
                    print(f"    Content length: {len(content)} chars")
                    print(f"    Preview: {content_preview}...")
                else:
                    print(f"\n[{response_count}] URL: {record['url']}")
                    print(f"    Content: [Empty or failed to extract]")
    
    print(f"\nTotal response records processed: {response_count}")
    print(f"Records with extracted content: {len(records)}")
    if english_only:
        print(f"Filtered non-English texts: {filtered_non_english}")
    
    # Save to file if output path provided
    if output_path and records:
        output_file = Path(output_path)
        
        if output_file.suffix == ".txt":
            # FastText format: __label__<label> <text on single line>
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    # Replace newlines with spaces for FastText single-line format
                    text = record["content"].replace("\n", " ").replace("\r", " ")
                    # Collapse multiple spaces
                    text = " ".join(text.split())
                    
                    if label:
                        f.write(f"__label__{label} {text}\n")
                    else:
                        f.write(f"{text}\n")
            print(f"Saved {len(records)} records to {output_path} (FastText format)")
        
        else:
            # JSONL format
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved {len(records)} records to {output_path} (JSONL format)")
    
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Read and extract content from WARC files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input WARC file (.warc or .warc.gz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file path (optional)"
    )
    parser.add_argument(
        "--num-records", "-n",
        type=int,
        default=None,
        help="Number of response records to extract (default: all)"
    )
    parser.add_argument(
        "--extract-text", "-e",
        action="store_true",
        help="Extract plain text from HTML using resiliparse (removes HTML tags)"
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        default=None,
        help="Label for FastText format (e.g., 'positive', 'negative'). Use with .txt output file."
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Only keep English texts (using FastText langid)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Extract content
    extract_html_content(
        warc_path=args.input,
        output_path=args.output,
        num_records=args.num_records,
        extract_text=args.extract_text,
        label=args.label,
        english_only=args.english_only,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
