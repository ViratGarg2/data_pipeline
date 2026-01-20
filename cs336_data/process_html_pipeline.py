"""
HTML to Text Processing Pipeline with Language Identification

This script processes parquet files containing text data, detects HTML content,
extracts plain text from HTML using the extract_text_from_html_bytes function,
identifies the language using FastText, and outputs processed data to new parquet files.

Usage:
    python -m cs336_data.process_html_pipeline
    OR
    python process_html_pipeline.py --max-files 10
"""

import glob
import os
import re
import sys
import time
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Handle imports for both module and direct execution
try:
    from cs336_data.extract import extract_text_from_html_bytes
    from cs336_data.langid import identify_language, get_language_name
except ModuleNotFoundError:
    # When running directly from the cs336_data directory
    from extract import extract_text_from_html_bytes
    from langid import identify_language, get_language_name


# ---------------- CONFIG ----------------
INPUT_GLOB = "/data3/dataset/the_pile_deduplicated/data/train-*-of-01650-*.parquet"
OUTPUT_DIR = "/home2/mehulag022/processed_data_html_extracted_lang"
TEXT_COLUMN = "text"
BATCH_SIZE = 1000
COMPRESSION = "zstd"
MAX_FILES = None  # Set to an integer to limit number of files processed, None for all
START_FILE_INDEX = 0  # Index of the first file to process (0-based)
# ----------------------------------------


def is_html_content(text: str) -> tuple[bool, str]:
    """
    Detect if the given text contains HTML content.
    
    Args:
        text: Text string to check for HTML content
        
    Returns:
        Tuple of (is_html: bool, html_type: str)
        html_type can be: 'full_html', 'partial_html', 'xml', 'plain_text'
    """
    if not text or not isinstance(text, str):
        return False, "empty"
    
    # Check for common HTML indicators
    text_lower = text[:2000].lower()  # Check first 2000 chars for performance
    
    # Full HTML document
    if "<!doctype html" in text_lower or "<html" in text_lower:
        return True, "full_html"
    
    # XML document
    if text_lower.strip().startswith("<?xml"):
        return True, "xml"
    
    # Partial HTML (contains HTML tags but not full document)
    html_tag_pattern = re.compile(r'<(div|span|p|a|img|table|tr|td|ul|ol|li|h[1-6]|br|hr|script|style|head|body|meta|link|form|input|button|nav|header|footer|section|article)\b', re.IGNORECASE)
    if html_tag_pattern.search(text[:5000]):
        return True, "partial_html"
    
    # Check for generic tags
    generic_tag_pattern = re.compile(r'<[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>.*?</[a-zA-Z][a-zA-Z0-9]*>', re.DOTALL)
    if generic_tag_pattern.search(text[:5000]):
        return True, "generic_tags"
    
    return False, "plain_text"


def process_single_file(
    input_path: str,
    output_path: str,
    text_column: str = "text",
    batch_size: int = 1000,
    compression: str = "zstd",
    enable_langid: bool = True,
) -> dict:
    """
    Process a single parquet file: detect HTML, extract text, and identify language.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        text_column: Column name containing text/HTML content
        batch_size: Number of rows to process at once
        compression: Output compression codec
        enable_langid: Whether to perform language identification
        
    Returns:
        Dictionary with processing statistics for this file
    """
    stats = {
        "total_rows": 0,
        "html_detected": 0,
        "html_extracted": 0,
        "extraction_failed": 0,
        "html_types": defaultdict(int),
        "languages": defaultdict(int),
        "language_scores": defaultdict(list),
    }
    
    # Read the parquet file
    table = pq.read_table(input_path)
    # Convert to Python dict - use to_pylist for better NumPy 2.0 compatibility
    columns = table.column_names
    df_dict = {col: table[col].to_pylist() for col in columns}
    
    if text_column not in df_dict:
        print(f"  Warning: Column '{text_column}' not found in {input_path}")
        return stats
    
    texts = df_dict[text_column]
    stats["total_rows"] = len(texts)
    
    # Process each text entry
    processed_texts = []
    detected_languages = []
    language_scores = []
    
    for text in texts:
        if text is None:
            processed_texts.append(text)
            detected_languages.append(None)
            language_scores.append(0.0)
            stats["html_types"]["none"] += 1
            stats["languages"]["none"] += 1
            continue
        
        # Detect HTML content
        is_html, html_type = is_html_content(text)
        stats["html_types"][html_type] += 1
        
        if is_html:
            stats["html_detected"] += 1
            
            # Convert to bytes and extract text
            if isinstance(text, str):
                html_bytes = text.encode("utf-8")
            else:
                html_bytes = text
            
            extracted = extract_text_from_html_bytes(html_bytes)
            
            if extracted is not None:
                processed_texts.append(extracted)
                stats["html_extracted"] += 1
                text_for_langid = extracted
            else:
                processed_texts.append(text)  # Keep original on failure
                stats["extraction_failed"] += 1
                text_for_langid = text
        else:
            # Not HTML, keep original
            processed_texts.append(text)
            text_for_langid = text
        
        # Language identification
        if enable_langid and text_for_langid:
            lang_code, lang_score = identify_language(text_for_langid)
            detected_languages.append(lang_code)
            language_scores.append(lang_score)
            stats["languages"][lang_code] += 1
            stats["language_scores"][lang_code].append(lang_score)
        else:
            detected_languages.append(None)
            language_scores.append(0.0)
    
    # Update the dictionary with processed texts and language info
    df_dict[text_column] = processed_texts
    if enable_langid:
        df_dict["detected_language"] = detected_languages
        df_dict["language_score"] = language_scores
    
    # Write to output parquet - handle NumPy 2.0 compatibility
    # Convert to PyArrow arrays explicitly to avoid copy issues
    arrays = []
    names = []
    for key, values in df_dict.items():
        names.append(key)
        if key == "language_score":
            # Ensure float array
            arrays.append(pa.array(values, type=pa.float64()))
        elif key == "detected_language":
            # Ensure string array with nulls
            arrays.append(pa.array(values, type=pa.string()))
        else:
            # Let PyArrow infer the type, convert to list if needed
            if hasattr(values, 'tolist'):
                values = values.tolist()
            arrays.append(pa.array(values))
    
    output_table = pa.table(dict(zip(names, arrays)))
    pq.write_table(output_table, output_path, compression=compression)
    
    return stats


def run_pipeline(
    input_glob: str = INPUT_GLOB,
    output_dir: str = OUTPUT_DIR,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    compression: str = COMPRESSION,
    max_files: int | None = MAX_FILES,
    start_index: int = START_FILE_INDEX,
    enable_langid: bool = True,
):
    """
    Run the full HTML extraction and language identification pipeline.
    
    Args:
        input_glob: Glob pattern for input parquet files
        output_dir: Directory to write processed files
        text_column: Column name containing text/HTML content
        batch_size: Batch size for processing
        compression: Output compression codec
        max_files: Maximum number of files to process (None for all)
        start_index: Index of the first file to process (0-based)
        enable_langid: Whether to perform language identification
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all input files
    all_input_files = sorted(glob.glob(input_glob))
    
    if not all_input_files:
        print(f"No files found matching pattern: {input_glob}")
        return
    
    total_available = len(all_input_files)
    
    # Apply start_index and max_files constraints
    if start_index >= total_available:
        print(f"Start index {start_index} is beyond available files ({total_available})")
        return
    
    # Slice the files based on start_index and max_files
    if max_files is not None:
        end_index = min(start_index + max_files, total_available)
        input_files = all_input_files[start_index:end_index]
    else:
        input_files = all_input_files[start_index:]
    
    print("=" * 60)
    print("HTML to Text Extraction Pipeline with Language ID")
    print("=" * 60)
    print(f"Input pattern: {input_glob}")
    print(f"Output directory: {output_dir}")
    print(f"Total files available: {total_available}")
    print(f"Start index: {start_index}")
    print(f"Max files to process: {max_files if max_files else 'ALL'}")
    print(f"Files to process this run: {len(input_files)}")
    print(f"File range: [{start_index} - {start_index + len(input_files) - 1}]")
    print(f"Language identification: {'ENABLED' if enable_langid else 'DISABLED'}")
    print("=" * 60)
    
    # Aggregate statistics
    total_stats = {
        "files_processed": 0,
        "total_rows": 0,
        "html_detected": 0,
        "html_extracted": 0,
        "extraction_failed": 0,
        "html_types": defaultdict(int),
        "languages": defaultdict(int),
    }
    
    start_time = time.time()
    
    # Process each file
    for i, input_path in enumerate(tqdm(input_files, desc="Processing files")):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{len(input_files)}] Processing: {filename}")
        
        try:
            file_stats = process_single_file(
                input_path=input_path,
                output_path=output_path,
                text_column=text_column,
                batch_size=batch_size,
                compression=compression,
                enable_langid=enable_langid,
            )
            
            # Update aggregate statistics
            total_stats["files_processed"] += 1
            total_stats["total_rows"] += file_stats["total_rows"]
            total_stats["html_detected"] += file_stats["html_detected"]
            total_stats["html_extracted"] += file_stats["html_extracted"]
            total_stats["extraction_failed"] += file_stats["extraction_failed"]
            
            for html_type, count in file_stats["html_types"].items():
                total_stats["html_types"][html_type] += count
            
            # Aggregate language stats
            if enable_langid:
                for lang, count in file_stats["languages"].items():
                    total_stats["languages"][lang] += count
            
            # Print file-level stats
            print(f"  Rows: {file_stats['total_rows']:,}")
            print(f"  HTML detected: {file_stats['html_detected']:,}")
            print(f"  Extracted: {file_stats['html_extracted']:,}")
            print(f"  Failed: {file_stats['extraction_failed']:,}")
            
            # Print top languages for this file
            if enable_langid and file_stats["languages"]:
                top_langs = sorted(file_stats["languages"].items(), key=lambda x: x[1], reverse=True)[:3]
                lang_str = ", ".join([f"{get_language_name(l)}:{c}" for l, c in top_langs])
                print(f"  Top languages: {lang_str}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            continue
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print final summary
    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETE - FINAL STATISTICS")
    print("=" * 60)
    print(f"Total files processed:    {total_stats['files_processed']:,}")
    print(f"Total rows processed:     {total_stats['total_rows']:,}")
    print(f"Total HTML detected:      {total_stats['html_detected']:,}")
    print(f"Total text extracted:     {total_stats['html_extracted']:,}")
    print(f"Total extraction failed:  {total_stats['extraction_failed']:,}")
    print(f"Total time:               {elapsed:.2f} seconds")
    if elapsed > 0:
        print(f"Processing rate:          {total_stats['total_rows'] / elapsed:.2f} rows/sec")
    print()
    
    # Print HTML type breakdown
    print("-" * 60)
    print("HTML TYPE BREAKDOWN")
    print("-" * 60)
    print(f"{'Type':<20} {'Count':>15} {'Percentage':>15}")
    print("-" * 60)
    
    total_content = sum(total_stats["html_types"].values())
    for html_type, count in sorted(
        total_stats["html_types"].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        percentage = (count / total_content * 100) if total_content > 0 else 0
        print(f"{html_type:<20} {count:>15,} {percentage:>14.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_content:>15,} {'100.00':>14}%")
    print()
    
    # Print language breakdown
    if enable_langid and total_stats["languages"]:
        print("-" * 60)
        print("LANGUAGE BREAKDOWN")
        print("-" * 60)
        print(f"{'Language':<25} {'Code':<8} {'Count':>12} {'Percentage':>12}")
        print("-" * 60)
        
        total_lang = sum(total_stats["languages"].values())
        for lang_code, count in sorted(
            total_stats["languages"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:30]:  # Show top 30 languages
            lang_name = get_language_name(lang_code)
            percentage = (count / total_lang * 100) if total_lang > 0 else 0
            print(f"{lang_name:<25} {lang_code:<8} {count:>12,} {percentage:>11.2f}%")
        
        if len(total_stats["languages"]) > 30:
            others = sum(v for k, v in sorted(
                total_stats["languages"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[30:])
            print(f"{'... others':<25} {'':<8} {others:>12,} {others/total_lang*100:>11.2f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<25} {'':<8} {total_lang:>12,} {'100.00':>11}%")
        print(f"Unique languages detected: {len(total_stats['languages'])}")
    
    print("=" * 60)
    
    return total_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HTML to Text Extraction Pipeline with Language Identification"
    )
    parser.add_argument(
        "--input-glob", "-i",
        type=str,
        default=INPUT_GLOB,
        help="Glob pattern for input parquet files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to write processed files"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--start-index", "-s",
        type=int,
        default=0,
        help="Index of the first file to process (0-based, default: 0)"
    )
    parser.add_argument(
        "--text-column", "-c",
        type=str,
        default=TEXT_COLUMN,
        help="Column name containing text/HTML content"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=COMPRESSION,
        help="Output compression codec (e.g., zstd, snappy, gzip)"
    )
    parser.add_argument(
        "--no-langid",
        action="store_true",
        help="Disable language identification (faster processing)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        text_column=args.text_column,
        batch_size=args.batch_size,
        compression=args.compression,
        max_files=args.max_files,
        start_index=args.start_index,
        enable_langid=not args.no_langid,
    )
