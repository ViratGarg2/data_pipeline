"""
HTML to Text Processing Pipeline with Language Identification and PII Masking

This script processes parquet files containing text data, detects HTML content,
extracts plain text from HTML using the extract_text_from_html_bytes function,
identifies the language using FastText, masks PII (emails, phones, IPs),
and outputs processed data to new parquet files.

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
    from cs336_data.pii_masking import mask_all_pii
    from cs336_data.toxicity import classify_nsfw, classify_toxic_speech, classify_content
    from cs336_data.quality_filter import gopher_quality_filter
    from cs336_data.quality_classifier import classify_quality, filter_by_quality
    from cs336_data import config
except ModuleNotFoundError:
    # When running directly from the cs336_data directory
    import config
    from extract import extract_text_from_html_bytes
    from langid import identify_language, get_language_name
    from pii_masking import mask_all_pii
    from toxicity import classify_nsfw, classify_toxic_speech, classify_content
    from quality_filter import gopher_quality_filter
    from quality_classifier import classify_quality, filter_by_quality


# ---------------- CONFIG ----------------
INPUT_GLOB = config.INPUT_GLOB
OUTPUT_DIR = config.OUTPUT_DIR
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
    enable_pii_masking: bool = True,
    enable_content_classification: bool = True,
    enable_quality_filter: bool = True,
    enable_quality_classifier: bool = True,
    max_rows: int | None = None,
) -> dict:
    """
    Process a single parquet file: detect HTML, extract text, identify language, mask PII, classify content, and apply quality filter.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        text_column: Column name containing text/HTML content
        batch_size: Number of rows to process at once
        compression: Output compression codec
        enable_langid: Whether to perform language identification
        enable_pii_masking: Whether to mask PII (emails, phones, IPs)
        enable_content_classification: Whether to classify NSFW and toxic content
        enable_quality_filter: Whether to apply Gopher quality filtering
        enable_quality_classifier: Whether to use NVIDIA DeBERTa quality classifier
        max_rows: Maximum number of rows to process per file (None for all)
        
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
        "pii_masked": {
            "emails": 0,
            "phones": 0,
            "ips": 0,
            "total": 0,
        },
        "content_classification": {
            "nsfw": 0,
            "non_nsfw": 0,
            "toxic": 0,
            "non_toxic": 0,
        },
        "filtered": {
            "non_english": 0,
            "nsfw": 0,
            "toxic": 0,
            "low_quality_deberta": 0,
            "total_filtered": 0,
            "kept": 0,
        },
        "quality_filter": {
            "passed": 0,
            "failed": 0,
            "failed_reasons": {
                "word_count": 0,
                "mean_word_length": 0,
                "hash_ratio": 0,
                "ellipsis_ratio": 0,
                "bullet_line_ratio": 0,
                "ellipsis_line_ratio": 0,
                "alphabetic_word_ratio": 0,
                "stop_word_count": 0,
                "empty_or_invalid": 0,
            },
        },
        "quality_classifier": {
            "High": 0,
            "Medium": 0,
            "Low": 0,
        },
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
    
    # Limit rows if max_rows is specified (for debugging/testing)
    if max_rows is not None and max_rows < len(texts):
        print(f"  Limiting to {max_rows} rows (out of {len(texts)})")
        texts = texts[:max_rows]
        # Also limit other columns to match
        for col in df_dict:
            df_dict[col] = df_dict[col][:max_rows]
    
    stats["total_rows"] = len(texts)
    
    # Process and filter each text entry sequentially
    # Order: HTML extraction -> PII masking -> Language filter -> Quality filter -> Content classification
    # Short-circuit: skip expensive checks if document already fails earlier filter
    
    filtered_results = []  # List of dicts with all info for kept documents
    kept_indices = []  # Track original indices of kept documents
    
    for idx, text in enumerate(tqdm(texts, desc="  Processing rows", unit="rows", leave=False)):
        # Track this document's data
        doc_data = {
            "text": None,
            "detected_language": None,
            "language_score": 0.0,
            "nsfw_label": None,
            "nsfw_score": 0.0,
            "toxic_label": None,
            "toxic_score": 0.0,
            "quality_label": None,
            "quality_score": 0.0,
        }
        
        # Handle None texts
        if text is None:
            stats["html_types"]["none"] += 1
            stats["filtered"]["total_filtered"] += 1
            continue
        
        # Step 1: HTML extraction
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
                current_text = extracted
                stats["html_extracted"] += 1
            else:
                current_text = text  # Keep original on failure
                stats["extraction_failed"] += 1
        else:
            current_text = text
        
        # Step 2: PII masking (always apply if enabled, before filtering)
        if enable_pii_masking and current_text:
            current_text, pii_stats = mask_all_pii(current_text)
            stats["pii_masked"]["emails"] += pii_stats["emails"]
            stats["pii_masked"]["phones"] += pii_stats["phones"]
            stats["pii_masked"]["ips"] += pii_stats["ips"]
            stats["pii_masked"]["total"] += pii_stats["total"]
        
        doc_data["text"] = current_text
        
        # Step 3: Language filter (FIRST filter - cheapest)
        if enable_langid and current_text:
            lang_code, lang_score = identify_language(current_text)
            doc_data["detected_language"] = lang_code
            doc_data["language_score"] = lang_score
            stats["languages"][lang_code] += 1
            
            # Short-circuit: if not English, skip remaining filters
            if lang_code != "en":
                stats["filtered"]["non_english"] += 1
                stats["filtered"]["total_filtered"] += 1
                continue  # Skip to next document
        
        # Step 4: Quality filter (SECOND filter - moderate cost, no model)
        if enable_quality_filter and current_text:
            quality_passed, quality_details = gopher_quality_filter(current_text)
            
            if quality_passed:
                stats["quality_filter"]["passed"] += 1
            else:
                stats["quality_filter"]["failed"] += 1
                for check in quality_details.get("failed_checks", []):
                    if check in stats["quality_filter"]["failed_reasons"]:
                        stats["quality_filter"]["failed_reasons"][check] += 1
                # Short-circuit: if quality fails, skip content classification
                stats["filtered"]["total_filtered"] += 1
                continue  # Skip to next document
        
        # Step 5: Quality classifier using DeBERTa (THIRD filter - model-based)
        # Only keep Medium and High quality texts, discard Low quality
        if enable_quality_classifier and current_text:
            keep_text, quality_label, quality_score = filter_by_quality(current_text)
            
            doc_data["quality_label"] = quality_label
            doc_data["quality_score"] = quality_score
            stats["quality_classifier"][quality_label] += 1
            
            if not keep_text:  # Low quality
                stats["filtered"]["low_quality_deberta"] += 1
                stats["filtered"]["total_filtered"] += 1
                continue  # Skip to next document
        
        # Step 6: Content classification (LAST filter - most expensive, uses models)
        if enable_content_classification and current_text:
            nsfw_label, nsfw_score = classify_nsfw(current_text)
            toxic_label, toxic_score = classify_toxic_speech(current_text)
            
            doc_data["nsfw_label"] = nsfw_label
            doc_data["nsfw_score"] = nsfw_score
            doc_data["toxic_label"] = toxic_label
            doc_data["toxic_score"] = toxic_score
            
            # Update stats
            if nsfw_label == "nsfw":
                stats["content_classification"]["nsfw"] += 1
            else:
                stats["content_classification"]["non_nsfw"] += 1
            if toxic_label == "toxic":
                stats["content_classification"]["toxic"] += 1
            else:
                stats["content_classification"]["non_toxic"] += 1
            
            # Short-circuit: if NSFW or toxic, filter out
            if nsfw_label == "nsfw":
                stats["filtered"]["nsfw"] += 1
                stats["filtered"]["total_filtered"] += 1
                continue  # Skip to next document
            
            if toxic_label == "toxic":
                stats["filtered"]["toxic"] += 1
                stats["filtered"]["total_filtered"] += 1
                continue  # Skip to next document
        
        # Document passed all filters - keep it!
        stats["filtered"]["kept"] += 1
        filtered_results.append(doc_data)
        kept_indices.append(idx)
    
    # Build output dataframe from filtered results
    if filtered_results:
        filtered_df_dict = {}
        
        # Copy original columns for kept rows
        for col in df_dict.keys():
            if col == text_column:
                # Use the processed text
                filtered_df_dict[col] = [doc["text"] for doc in filtered_results]
            else:
                # Copy original values for kept indices
                original_values = df_dict[col]
                filtered_df_dict[col] = [original_values[i] for i in kept_indices]
        
        # Add new columns
        if enable_langid:
            filtered_df_dict["detected_language"] = [doc["detected_language"] for doc in filtered_results]
            filtered_df_dict["language_score"] = [doc["language_score"] for doc in filtered_results]
        if enable_quality_classifier:
            filtered_df_dict["quality_label"] = [doc["quality_label"] for doc in filtered_results]
            filtered_df_dict["quality_score"] = [doc["quality_score"] for doc in filtered_results]
        if enable_content_classification:
            filtered_df_dict["nsfw_label"] = [doc["nsfw_label"] for doc in filtered_results]
            filtered_df_dict["nsfw_score"] = [doc["nsfw_score"] for doc in filtered_results]
            filtered_df_dict["toxic_label"] = [doc["toxic_label"] for doc in filtered_results]
            filtered_df_dict["toxic_score"] = [doc["toxic_score"] for doc in filtered_results]
    else:
        filtered_df_dict = {col: [] for col in df_dict.keys()}
        if enable_langid:
            filtered_df_dict["detected_language"] = []
            filtered_df_dict["language_score"] = []
        if enable_quality_classifier:
            filtered_df_dict["quality_label"] = []
            filtered_df_dict["quality_score"] = []
        if enable_content_classification:
            filtered_df_dict["nsfw_label"] = []
            filtered_df_dict["nsfw_score"] = []
            filtered_df_dict["toxic_label"] = []
            filtered_df_dict["toxic_score"] = []
    
    print(f"  Filtered: {stats['filtered']['total_filtered']} rows removed, {stats['filtered']['kept']} rows kept")
    print(f"    - Non-English: {stats['filtered']['non_english']}")
    print(f"    - Gopher quality failed: {stats['quality_filter']['failed']}")
    print(f"    - DeBERTa low quality: {stats['filtered']['low_quality_deberta']}")
    print(f"    - NSFW: {stats['filtered']['nsfw']}")
    print(f"    - Toxic: {stats['filtered']['toxic']}")
    
    # Write to output parquet - handle NumPy 2.0 compatibility
    # Convert to PyArrow arrays explicitly to avoid copy issues
    arrays = []
    names = []
    for key, values in filtered_df_dict.items():
        names.append(key)
        if key in ("language_score", "nsfw_score", "toxic_score", "quality_score"):
            # Ensure float array
            arrays.append(pa.array(values, type=pa.float64()))
        elif key in ("detected_language", "nsfw_label", "toxic_label", "quality_label"):
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
    enable_pii_masking: bool = True,
    enable_content_classification: bool = True,
    enable_quality_filter: bool = True,
    enable_quality_classifier: bool = True,
    max_rows: int | None = 1000,  # DEBUG: Limit rows per file (set to None for all rows)
):
    """
    Run the full HTML extraction, language identification, PII masking, content classification, and quality filtering pipeline.
    
    Args:
        input_glob: Glob pattern for input parquet files
        output_dir: Directory to write processed files
        text_column: Column name containing text/HTML content
        batch_size: Batch size for processing
        compression: Output compression codec
        max_files: Maximum number of files to process (None for all)
        start_index: Index of the first file to process (0-based)
        enable_langid: Whether to perform language identification
        enable_pii_masking: Whether to mask PII (emails, phones, IPs)
        enable_content_classification: Whether to classify NSFW and toxic content
        enable_quality_filter: Whether to apply Gopher quality filtering
        max_rows: Maximum number of rows to process per file (None for all, default: 1000 for debugging)
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
    print("HTML to Text Extraction Pipeline with Language ID & PII Masking")
    print("=" * 60)
    print(f"Input pattern: {input_glob}")
    print(f"Output directory: {output_dir}")
    print(f"Total files available: {total_available}")
    print(f"Start index: {start_index}")
    print(f"Max files to process: {max_files if max_files else 'ALL'}")
    print(f"Max rows per file: {max_rows if max_rows else 'ALL'}")
    print(f"Files to process this run: {len(input_files)}")
    print(f"File range: [{start_index} - {start_index + len(input_files) - 1}]")
    print(f"Language identification: {'ENABLED' if enable_langid else 'DISABLED'}")
    print(f"PII masking: {'ENABLED' if enable_pii_masking else 'DISABLED'}")
    print(f"Content classification: {'ENABLED' if enable_content_classification else 'DISABLED'}")
    print(f"Quality filter (Gopher): {'ENABLED' if enable_quality_filter else 'DISABLED'}")
    print(f"Quality classifier (DeBERTa): {'ENABLED' if enable_quality_classifier else 'DISABLED'}")
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
        "pii_masked": {
            "emails": 0,
            "phones": 0,
            "ips": 0,
            "total": 0,
        },
        "content_classification": {
            "nsfw": 0,
            "non_nsfw": 0,
            "toxic": 0,
            "non_toxic": 0,
        },
        "filtered": {
            "non_english": 0,
            "nsfw": 0,
            "toxic": 0,
            "low_quality_deberta": 0,
            "total_filtered": 0,
            "kept": 0,
        },
        "quality_filter": {
            "passed": 0,
            "failed": 0,
            "failed_reasons": {
                "word_count": 0,
                "mean_word_length": 0,
                "hash_ratio": 0,
                "ellipsis_ratio": 0,
                "bullet_line_ratio": 0,
                "ellipsis_line_ratio": 0,
                "alphabetic_word_ratio": 0,
                "stop_word_count": 0,
                "empty_or_invalid": 0,
            },
        },
        "quality_classifier": {
            "High": 0,
            "Medium": 0,
            "Low": 0,
        },
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
                enable_pii_masking=enable_pii_masking,
                enable_content_classification=enable_content_classification,
                enable_quality_filter=enable_quality_filter,
                enable_quality_classifier=enable_quality_classifier,
                max_rows=max_rows,
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
            
            # Aggregate PII stats
            if enable_pii_masking:
                total_stats["pii_masked"]["emails"] += file_stats["pii_masked"]["emails"]
                total_stats["pii_masked"]["phones"] += file_stats["pii_masked"]["phones"]
                total_stats["pii_masked"]["ips"] += file_stats["pii_masked"]["ips"]
                total_stats["pii_masked"]["total"] += file_stats["pii_masked"]["total"]
            
            # Aggregate content classification stats
            if enable_content_classification:
                total_stats["content_classification"]["nsfw"] += file_stats["content_classification"]["nsfw"]
                total_stats["content_classification"]["non_nsfw"] += file_stats["content_classification"]["non_nsfw"]
                total_stats["content_classification"]["toxic"] += file_stats["content_classification"]["toxic"]
                total_stats["content_classification"]["non_toxic"] += file_stats["content_classification"]["non_toxic"]
            
            # Aggregate filtering stats
            total_stats["filtered"]["non_english"] += file_stats["filtered"]["non_english"]
            total_stats["filtered"]["nsfw"] += file_stats["filtered"]["nsfw"]
            total_stats["filtered"]["toxic"] += file_stats["filtered"]["toxic"]
            total_stats["filtered"]["low_quality_deberta"] += file_stats["filtered"].get("low_quality_deberta", 0)
            total_stats["filtered"]["total_filtered"] += file_stats["filtered"]["total_filtered"]
            total_stats["filtered"]["kept"] += file_stats["filtered"]["kept"]
            
            # Aggregate quality filter stats
            if enable_quality_filter:
                total_stats["quality_filter"]["passed"] += file_stats["quality_filter"]["passed"]
                total_stats["quality_filter"]["failed"] += file_stats["quality_filter"]["failed"]
                for reason, count in file_stats["quality_filter"]["failed_reasons"].items():
                    total_stats["quality_filter"]["failed_reasons"][reason] += count
            
            # Aggregate quality classifier stats
            if enable_quality_classifier:
                for quality_level in ("High", "Medium", "Low"):
                    total_stats["quality_classifier"][quality_level] += file_stats["quality_classifier"].get(quality_level, 0)
            
            # Print file-level stats
            print(f"  Rows: {file_stats['total_rows']:,}")
            print(f"  HTML detected: {file_stats['html_detected']:,}")
            print(f"  Extracted: {file_stats['html_extracted']:,}")
            print(f"  Failed: {file_stats['extraction_failed']:,}")
            
            # Print PII masking stats for this file
            if enable_pii_masking and file_stats["pii_masked"]["total"] > 0:
                pii = file_stats["pii_masked"]
                print(f"  PII masked: {pii['total']} (emails:{pii['emails']}, phones:{pii['phones']}, IPs:{pii['ips']})")
            
            # Print content classification stats for this file
            if enable_content_classification:
                cc = file_stats["content_classification"]
                print(f"  Content: NSFW:{cc['nsfw']}, Non-NSFW:{cc['non_nsfw']}, Toxic:{cc['toxic']}, Non-Toxic:{cc['non_toxic']}")
            
            # Print quality classifier stats for this file
            if enable_quality_classifier:
                qc = file_stats["quality_classifier"]
                print(f"  Quality (DeBERTa): High:{qc['High']}, Medium:{qc['Medium']}, Low:{qc['Low']}")
            
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
    
    # Print PII masking summary
    if enable_pii_masking:
        print("-" * 60)
        print("PII MASKING SUMMARY")
        print("-" * 60)
        pii = total_stats["pii_masked"]
        print(f"Email addresses masked:   {pii['emails']:,}")
        print(f"Phone numbers masked:     {pii['phones']:,}")
        print(f"IP addresses masked:      {pii['ips']:,}")
        print(f"Total PII masked:         {pii['total']:,}")
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
    
    # Print content classification summary
    if enable_content_classification:
        print("-" * 60)
        print("CONTENT CLASSIFICATION SUMMARY")
        print("-" * 60)
        cc = total_stats["content_classification"]
        total_classified = cc["nsfw"] + cc["non_nsfw"]
        print(f"NSFW content:             {cc['nsfw']:,} ({cc['nsfw']/total_classified*100:.2f}%)" if total_classified > 0 else f"NSFW content:             {cc['nsfw']:,}")
        print(f"Non-NSFW content:         {cc['non_nsfw']:,} ({cc['non_nsfw']/total_classified*100:.2f}%)" if total_classified > 0 else f"Non-NSFW content:         {cc['non_nsfw']:,}")
        print(f"Toxic content:            {cc['toxic']:,} ({cc['toxic']/total_classified*100:.2f}%)" if total_classified > 0 else f"Toxic content:            {cc['toxic']:,}")
        print(f"Non-Toxic content:        {cc['non_toxic']:,} ({cc['non_toxic']/total_classified*100:.2f}%)" if total_classified > 0 else f"Non-Toxic content:        {cc['non_toxic']:,}")
        print()
    
    # Print quality classifier summary (DeBERTa)
    if enable_quality_classifier:
        print("-" * 60)
        print("QUALITY CLASSIFIER SUMMARY (NVIDIA DeBERTa)")
        print("-" * 60)
        qc = total_stats["quality_classifier"]
        total_qc = qc["High"] + qc["Medium"] + qc["Low"]
        print(f"High quality:             {qc['High']:,} ({qc['High']/total_qc*100:.2f}%)" if total_qc > 0 else f"High quality:             {qc['High']:,}")
        print(f"Medium quality:           {qc['Medium']:,} ({qc['Medium']/total_qc*100:.2f}%)" if total_qc > 0 else f"Medium quality:           {qc['Medium']:,}")
        print(f"Low quality (filtered):   {qc['Low']:,} ({qc['Low']/total_qc*100:.2f}%)" if total_qc > 0 else f"Low quality (filtered):   {qc['Low']:,}")
        print(f"Kept (High+Medium):       {qc['High']+qc['Medium']:,}")
        print()
    
    # Print filtering summary
    print("-" * 60)
    print("FILTERING SUMMARY (English only, Non-NSFW, Non-Toxic, Quality)")
    print("-" * 60)
    flt = total_stats["filtered"]
    total_processed = flt["kept"] + flt["total_filtered"]
    print(f"Total texts processed:    {total_processed:,}")
    print(f"Texts KEPT:               {flt['kept']:,} ({flt['kept']/total_processed*100:.2f}%)" if total_processed > 0 else f"Texts KEPT:               {flt['kept']:,}")
    print(f"Texts FILTERED OUT:       {flt['total_filtered']:,} ({flt['total_filtered']/total_processed*100:.2f}%)" if total_processed > 0 else f"Texts FILTERED OUT:       {flt['total_filtered']:,}")
    print()
    print("Filtered out due to:")
    print(f"  - Non-English language: {flt['non_english']:,}")
    print(f"  - Gopher quality:       {total_stats['quality_filter']['failed']:,}")
    print(f"  - DeBERTa low quality:  {flt.get('low_quality_deberta', 0):,}")
    print(f"  - NSFW content:         {flt['nsfw']:,}")
    print(f"  - Toxic content:        {flt['toxic']:,}")
    print()
    
    # Print quality filter summary
    if enable_quality_filter:
        print("-" * 60)
        print("GOPHER QUALITY FILTER SUMMARY")
        print("-" * 60)
        qf = total_stats["quality_filter"]
        total_qf = qf["passed"] + qf["failed"]
        print(f"Texts passed quality:     {qf['passed']:,} ({qf['passed']/total_qf*100:.2f}%)" if total_qf > 0 else f"Texts passed quality:     {qf['passed']:,}")
        print(f"Texts failed quality:     {qf['failed']:,} ({qf['failed']/total_qf*100:.2f}%)" if total_qf > 0 else f"Texts failed quality:     {qf['failed']:,}")
        print()
        print("Failed quality checks breakdown:")
        for reason, count in sorted(qf["failed_reasons"].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  - {reason}: {count:,}")
        print()
    
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
    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Disable PII masking (emails, phones, IPs)"
    )
    parser.add_argument(
        "--no-content-classification",
        action="store_true",
        help="Disable NSFW and toxic content classification"
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable Gopher quality filtering"
    )
    parser.add_argument(
        "--no-quality-classifier",
        action="store_true",
        help="Disable NVIDIA DeBERTa quality classifier"
    )
    parser.add_argument(
        "--max-rows", "-r",
        type=int,
        default=-1,
        help="Maximum rows to process per file (default: 1000 for debugging, use -1 for all)"
    )
    
    args = parser.parse_args()
    
    # Handle max_rows: -1 means all rows
    max_rows = None if args.max_rows == -1 else args.max_rows
    
    run_pipeline(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        text_column=args.text_column,
        batch_size=args.batch_size,
        compression=args.compression,
        max_files=args.max_files,
        start_index=args.start_index,
        enable_langid=not args.no_langid,
        enable_pii_masking=not args.no_pii,
        enable_content_classification=not args.no_content_classification,
        enable_quality_filter=not args.no_quality_filter,
        enable_quality_classifier=not args.no_quality_classifier,
        max_rows=max_rows,
    )
