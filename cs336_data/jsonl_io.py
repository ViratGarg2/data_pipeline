"""
JSONL File I/O Utilities

This module provides functions for reading and writing compressed JSONL files
(.jsonl, .jsonl.gz, .jsonl.zst) commonly used in NLP data pipelines.
"""

import json
import os

# For reading .jsonl.zst and .jsonl.gz files
try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import gzip as gzip_module
except ImportError:
    gzip_module = None


def read_jsonl_compressed(file_path: str, text_column: str = "text", max_rows: int | None = None) -> dict:
    """
    Read a compressed JSONL file (.jsonl.zst or .jsonl.gz) and return a dict of columns.
    
    Args:
        file_path: Path to the compressed JSONL file
        text_column: Column name to extract (default: "text")
        max_rows: Maximum number of rows to read (None for all)
        
    Returns:
        Dictionary with column names as keys and lists of values
    """
    records = []
    
    # Determine compression type
    if file_path.endswith('.zst') or file_path.endswith('.jsonl.zst'):
        if zstd is None:
            raise ImportError("zstandard package required for .zst files: pip install zstandard")
        with open(file_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                import io
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for i, line in enumerate(text_stream):
                    if max_rows is not None and i >= max_rows:
                        break
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    elif file_path.endswith('.gz') or file_path.endswith('.jsonl.gz'):
        if gzip_module is None:
            raise ImportError("gzip module not available")
        with gzip_module.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    else:
        # Plain jsonl
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    if not records:
        return {text_column: []}
    
    # Convert list of dicts to dict of lists
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    
    result = {key: [] for key in all_keys}
    for r in records:
        for key in all_keys:
            result[key].append(r.get(key))
    
    return result


def write_jsonl_compressed(file_path: str, data_dict: dict, compression: str = "zstd") -> str | None:
    """
    Write data to a compressed JSONL file.
    
    Args:
        file_path: Output file path
        data_dict: Dictionary with column names as keys and lists of values
        compression: Compression type ('zstd', 'gzip', or 'none')
        
    Returns:
        Actual output file path (may differ if extension was adjusted), or None if empty
    """
    if not data_dict:
        return None
    
    # Get number of rows
    first_key = next(iter(data_dict.keys()))
    num_rows = len(data_dict[first_key])
    
    if num_rows == 0:
        return None
    
    # Convert dict of lists to list of dicts
    records = []
    for i in range(num_rows):
        record = {key: data_dict[key][i] for key in data_dict.keys()}
        records.append(record)
    
    # Ensure correct extension
    if compression == "zstd":
        if not file_path.endswith('.jsonl.zst'):
            file_path = file_path.rsplit('.', 1)[0] + '.jsonl.zst'
        if zstd is None:
            raise ImportError("zstandard package required: pip install zstandard")
        with open(file_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=3)
            with cctx.stream_writer(f) as writer:
                for record in records:
                    line = json.dumps(record, ensure_ascii=False) + '\n'
                    writer.write(line.encode('utf-8'))
    elif compression == "gzip":
        if not file_path.endswith('.jsonl.gz'):
            file_path = file_path.rsplit('.', 1)[0] + '.jsonl.gz'
        if gzip_module is None:
            raise ImportError("gzip module not available")
        with gzip_module.open(file_path, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    else:
        # No compression
        if not file_path.endswith('.jsonl'):
            file_path = file_path.rsplit('.', 1)[0] + '.jsonl'
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return file_path


def read_wet_file(input_path: str, max_rows: int | None = None) -> dict:
    """Read a Common Crawl WET file (.warc.wet.gz) into {"text": [...], "url": [...]}.

    WET "conversion" records hold the text Common Crawl extracted from each page.
    """
    from fastwarc.stream_io import FileStream, GZipStream
    from fastwarc.warc import ArchiveIterator, WarcRecordType

    texts, urls = [], []
    stream = GZipStream(FileStream(input_path, "rb"))
    for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
        texts.append(record.reader.read().decode("utf-8", errors="replace"))
        urls.append(record.headers.get("WARC-Target-URI", ""))
        if max_rows is not None and len(texts) >= max_rows:
            break
    return {"text": texts, "url": urls}


def get_output_path(input_path: str, output_dir: str, input_format: str, output_format: str) -> str:
    """
    Generate output file path based on input path and desired output format.
    
    Args:
        input_path: Original input file path
        output_dir: Output directory
        input_format: Input format ('parquet' or 'jsonl')
        output_format: Output format ('parquet' or 'jsonl')
        
    Returns:
        Output file path with correct extension
    """
    filename = os.path.basename(input_path)
    
    # Remove all known extensions
    base = filename
    for ext in ['.warc.wet.gz', '.jsonl.zst', '.jsonl.gz', '.jsonl', '.parquet', '.zst', '.gz']:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    
    # Add new extension based on output format
    if output_format == "parquet":
        return os.path.join(output_dir, base + '.parquet')
    else:
        return os.path.join(output_dir, base + '.jsonl.zst')
