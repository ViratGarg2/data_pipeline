"""
HTML text extraction module using Resiliparse.

This module provides functions to extract plain text from HTML content,
handling various text encodings and processing large datasets.
"""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract plain text from a byte string containing raw HTML.
    
    This function handles encoding detection for non-UTF-8 content
    by first attempting UTF-8 decoding, then falling back to 
    automatic encoding detection using Resiliparse.
    
    Args:
        html_bytes: Raw HTML content as bytes
        
    Returns:
        Extracted plain text as a string, or None if extraction fails
    """
    if not html_bytes:
        return None
    
    # Try UTF-8 first (most common encoding on the web)
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fall back to encoding detection
        detected_encoding = detect_encoding(html_bytes)
        if detected_encoding:
            try:
                html_str = html_bytes.decode(detected_encoding)
            except (UnicodeDecodeError, LookupError):
                # If detection fails or encoding is unknown, try latin-1 as fallback
                # latin-1 can decode any byte sequence
                html_str = html_bytes.decode("latin-1")
        else:
            # No encoding detected, use latin-1 as ultimate fallback
            html_str = html_bytes.decode("latin-1")
    
    # Extract plain text from HTML
    extracted_text = extract_plain_text(html_str)
    
    return extracted_text


def process_parquet_html_to_text(
    input_glob: str,
    output_dir: str,
    text_column: str = "text",
    batch_size: int = 1000,
    compression: str = "zstd",
) -> dict:
    """
    Process parquet files containing HTML content and extract plain text.
    
    This function reads parquet files matching the input glob pattern,
    extracts plain text from HTML content in the specified column,
    and writes the processed data to new parquet files.
    
    Args:
        input_glob: Glob pattern for input parquet files
        output_dir: Directory to write processed parquet files
        text_column: Name of the column containing HTML content
        batch_size: Number of samples to process before writing to disk
        compression: Compression codec for output files (e.g., 'zstd', 'snappy')
        
    Returns:
        Dictionary containing processing statistics
    """
    import os
    import time
    
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset in streaming mode to handle large files
    dataset = load_dataset(
        "parquet",
        data_files=input_glob,
        split="train",
        streaming=True
    )
    
    total_samples = 0
    total_extracted = 0
    total_failed = 0
    buffer = []
    writer = None
    part_id = 0
    
    start_time = time.time()
    
    for sample in tqdm(dataset, desc="Processing HTML to text"):
        total_samples += 1
        
        # Check if the text column exists and contains data
        if text_column in sample and sample[text_column]:
            content = sample[text_column]
            
            # If content is already bytes, use directly; otherwise encode
            if isinstance(content, bytes):
                html_bytes = content
            elif isinstance(content, str):
                # If it's a string, it might already be plain text
                # or it could be HTML. We'll process it anyway.
                html_bytes = content.encode("utf-8")
            else:
                # Skip non-string/bytes content
                buffer.append(sample)
                continue
            
            # Extract text from HTML
            extracted_text = extract_text_from_html_bytes(html_bytes)
            
            if extracted_text is not None:
                sample[text_column] = extracted_text
                total_extracted += 1
            else:
                total_failed += 1
        
        buffer.append(sample)
        
        # Write batch to parquet file
        if len(buffer) >= batch_size:
            table = pa.Table.from_pylist(buffer)
            
            out_path = os.path.join(output_dir, f"part-{part_id:05d}.parquet")
            pq.write_table(table, out_path, compression=compression)
            
            buffer.clear()
            part_id += 1
            
            # Progress update
            elapsed = time.time() - start_time
            print(
                f"Processed {total_samples:,} samples | "
                f"Extracted: {total_extracted:,} | "
                f"Failed: {total_failed:,} | "
                f"Time: {elapsed:.1f}s"
            )
    
    # Write remaining samples
    if buffer:
        table = pa.Table.from_pylist(buffer)
        out_path = os.path.join(output_dir, f"part-{part_id:05d}.parquet")
        pq.write_table(table, out_path, compression=compression)
    
    end_time = time.time()
    
    stats = {
        "total_samples": total_samples,
        "total_extracted": total_extracted,
        "total_failed": total_failed,
        "total_time_seconds": end_time - start_time,
        "output_files": part_id + 1,
    }
    
    print("\n==== FINAL STATS ====")
    print(f"Total samples processed: {stats['total_samples']:,}")
    print(f"Successfully extracted:  {stats['total_extracted']:,}")
    print(f"Failed extractions:      {stats['total_failed']:,}")
    print(f"Total time (seconds):    {stats['total_time_seconds']:.2f}")
    print(f"Output files created:    {stats['output_files']}")
    
    return stats
