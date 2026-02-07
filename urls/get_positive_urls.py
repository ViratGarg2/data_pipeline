# refrence -> https://github.com/Huhua-Xiao/cs336-assignment4-data/blob/main/cs336_data/quality_classifier/query_positive_samples.py

# Add parent directory to path so cs336_data can be found
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fasttext
import re

# from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.langid import identify_language
# from cs336_data.pii_masking import mask_emails, mask_phone, mask_ip
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.quality_filter import gopher_quality_filter
from tqdm import tqdm
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType

input_path = "./subsampled_positive_urls_new.warc.gz"
output_path = "./subsampled_positive_samples_new.txt"
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes):
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors='replace')

    return extract_plain_text(html_str)

def clean_text_basic(text: str): 
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def query_positive_sample():
    cnt_total = 0
    cnt_written = 0
    cnt_non_english = 0
    cnt_nsfw = 0
    cnt_toxic = 0
    cnt_low_quality = 0
    cnt_empty = 0
    cnt_too_large = 0
    cnt_errors = 0
    
    # Limits to prevent memory issues
    MAX_CONTENT_SIZE = 5 * 1024 * 1024  # 5 MB max for raw HTML
    MAX_TEXT_LENGTH = 500000  # 500K characters max for extracted text
    
    # Open output file for incremental writing (saves memory)
    with open(output_path, "wt", encoding="utf-8") as f_out:
        with gzip.open(input_path, 'rb') as f:
            # Create progress bar with better description
            pbar = tqdm(ArchiveIterator(f), desc="Processing WARC", unit="records")
            
            for record in pbar:
                if record.record_type == WarcRecordType.response and record.content_length > 0:
                    cnt_total += 1
                    
                    try:
                        # Skip if content is too large (prevents memory issues)
                        if record.content_length > MAX_CONTENT_SIZE:
                            cnt_too_large += 1
                            # Still need to read to advance the iterator, but discard
                            _ = record.reader.read()
                            continue
                        
                        html_bytes = record.reader.read()
                        
                        # Double check size after reading
                        if len(html_bytes) > MAX_CONTENT_SIZE:
                            cnt_too_large += 1
                            del html_bytes
                            continue
                        
                        text = extract_text_from_html_bytes(html_bytes)
                        del html_bytes  # Free memory immediately
                        
                        # Skip if extracted text is too long
                        if len(text) > MAX_TEXT_LENGTH:
                            cnt_too_large += 1
                            del text
                            continue
                        
                        # Language check
                        lang, lang_conf = identify_language(text)
                        if lang != "en" or lang_conf < 0.5:
                            cnt_non_english += 1
                            del text
                            continue
                        
                        # NSFW check
                        nsfw_label, nsfw_conf = classify_nsfw(text)
                        if nsfw_label == "nsfw" and nsfw_conf >= 0.7:
                            cnt_nsfw += 1
                            del text
                            continue
                        
                        # Toxic check
                        toxic_label, toxic_conf = classify_toxic_speech(text)
                        if toxic_label == "toxic" and toxic_conf >= 0.7:
                            cnt_toxic += 1
                            del text
                            continue
                        
                        # Quality check
                        if not gopher_quality_filter(text):
                            cnt_low_quality += 1
                            del text
                            continue
                        
                        # Clean text
                        text = clean_text_basic(text)
                        if not text:
                            cnt_empty += 1
                            continue
                        
                        # Write directly to file instead of storing in memory
                        f_out.write(f"__label__positive {text}\n")
                        f_out.flush()  # Flush to disk immediately
                        cnt_written += 1
                        del text  # Free memory
                        
                        # Update progress bar with current stats
                        pbar.set_postfix({
                            "written": cnt_written,
                            "non_en": cnt_non_english,
                            "large": cnt_too_large,
                            "err": cnt_errors
                        })
                        
                    except Exception as e:
                        cnt_errors += 1
                        # Skip problematic records but don't crash
                        continue
            
            pbar.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total response records processed: {cnt_total}")
    print(f"Written to output:                {cnt_written}")
    print(f"Skipped - Non-English/Low conf:   {cnt_non_english}")
    print(f"Skipped - NSFW content:           {cnt_nsfw}")
    print(f"Skipped - Toxic content:          {cnt_toxic}")
    print(f"Skipped - Low quality:            {cnt_low_quality}")
    print(f"Skipped - Empty after cleaning:   {cnt_empty}")
    print(f"Skipped - Too large:              {cnt_too_large}")
    print(f"Skipped - Errors:                 {cnt_errors}")
    print(f"\nOutput file: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    query_positive_sample()
    