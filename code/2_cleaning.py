import os
import json
import re
import string
import tempfile

import boto3
import requests  
import spacy

BUCKET = os.environ.get("BUCKET_NAME", "amazonreviewsnlp")
RAW_PREFIX = "amazon_raw/"
CLEAN_PREFIX = "amazon_clean/"

s3 = boto3.client("s3")

# Load NER model
nlp = spacy.load("en_core_web_sm")

# Regexes for cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USERNAME_RE = re.compile(r"@\w+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def basic_clean(text: str) -> str:
    if not text:
        return text

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = URL_RE.sub(" ", text)

    # Remove @usernames
    text = USERNAME_RE.sub(" ", text)

    # Remove non-ASCII (including most emojis)
    text = NON_ASCII_RE.sub(" ", text)

    # Remove punctuation
    text = text.translate(PUNCT_TABLE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def anonymize_ner(text: str) -> str:
    """
    Replace named entities with generic tags like [PERSON], [ORG], etc.
    """
    if not text:
        return text

    doc = nlp(text)
    out = []
    last_end = 0

    for ent in doc.ents:
        # Add text before entity
        out.append(text[last_end:ent.start_char])

        # Map entity label to something simple
        if ent.label_ in ("PERSON",):
            token = "[PERSON]"
        elif ent.label_ in ("ORG",):
            token = "[ORG]"
        elif ent.label_ in ("GPE", "LOC"):
            token = "[LOCATION]"
        else:
            token = "[ENTITY]"

        out.append(token)
        last_end = ent.end_char

    # Add remaining text after last entity
    out.append(text[last_end:])

    return "".join(out)


def clean_text_pipeline(text: str) -> str:
    text = basic_clean(text)
    text = anonymize_ner(text)
    return text


def process_s3_object(key: str):
    """
    Read one raw JSONL file from S3, clean it line-by-line,
    upload cleaned version to CLEAN_PREFIX, same filename.
    """
    print(f"Processing: s3://{BUCKET}/{key}")

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    body = obj["Body"]

    # decide output key
    if key.startswith(RAW_PREFIX):
        clean_key = CLEAN_PREFIX + key[len(RAW_PREFIX) :]
    else:
        clean_key = CLEAN_PREFIX + key

    # Write cleaned data to a temp file on disk
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp_path = tmp.name

        for raw_line in body.iter_lines():
            if not raw_line:
                continue
            # S3 gives bytes; decode to str
            line = raw_line.decode("utf-8")

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # skip bad lines, or log if you care
                continue

            # Clean selected text fields
            for field in ("reviewText", "summary", "title"):
                if field in record and isinstance(record[field], str):
                    record[field] = clean_text_pipeline(record[field])

            tmp.write(json.dumps(record) + "\n")

    # Upload cleaned file back to S3
    print(f"Uploading cleaned file to s3://{BUCKET}/{clean_key}")
    try:
        s3.upload_file(tmp_path, BUCKET, clean_key)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print(f"Done: s3://{BUCKET}/{clean_key}")


def main():
    if not BUCKET:
        raise RuntimeError(
            "Set BUCKET_NAME env var (export BUCKET_NAME=...) before running."
        )

    continuation_token = None
    while True:
        if continuation_token:
            resp = s3.list_objects_v2(
                Bucket=BUCKET, Prefix=RAW_PREFIX, ContinuationToken=continuation_token
            )
        else:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_PREFIX)

        contents = resp.get("Contents", [])
        if not contents:
            print("No objects found under RAW_PREFIX.")
            break

        for obj in contents:
            key = obj["Key"]
            if not key.endswith(".jsonl"):  
                continue
            process_s3_object(key)

        if resp.get("IsTruncated"):
            continuation_token = resp["NextContinuationToken"]
        else:
            break


if __name__ == "__main__":
    main()
