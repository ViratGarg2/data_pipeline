# refrence -> https://github.com/Huhua-Xiao/cs336-assignment4-data/blob/main/cs336_data/quality_classifier/get_negative_samples.py

import fasttext
import re

# from cs336_data.extract_text import extract_text_from_html_bytes
# from cs336_data.language_identification import identify_language
# from cs336_data.mask_pii import mask_emails, mask_phone, mask_ip
# from cs336_data.harmful_content import identify_nsfw, identify_toxic_speech
# from cs336_data.gopher_quality_filters import gopher_quality_filter

import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType

# README instructions how to get the negative samples WARC file
input_path = "./CC-MAIN-20251005114239-20251005144239-00000.warc.gz.1"
output_path = "./negative_samples.txt"
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes):
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors='replace')

    return extract_plain_text(html_str)

def get_negative_samples():
    samples = []
    count = 0
    with gzip.open(input_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if not text.strip():
                    continue
                count += 1

                nagative_sample = f"__label__negative {text.replace('\n', ' ').replace('\r', ' ')}\n"
                samples.append(nagative_sample)
                if count == 10000:
                    break

    with open(output_path, "wt", encoding="utf-8") as f_out:
        f_out.writelines(samples)

    print(f"Written {len(samples)} negative samples to {output_path}")

if __name__ == "__main__":
    get_negative_samples()