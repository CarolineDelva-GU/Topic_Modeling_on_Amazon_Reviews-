

import os
import requests
from bs4 import BeautifulSoup
import boto3
import gzip
import tempfile
URL = "https://amazon-reviews-2023.github.io/" 
BUCKET = os.environ.get("BUCKET_NAME")
S3_PREFIX = "amazon_reviews"
s3 = boto3.client("s3")
MAX_FILES = None



def upload_gz_stream_to_s3(href: str, key: str) -> None:
    """
    Stream a .gz file from href, decompress to a temp file, upload temp file to S3,
    then remove the temp file.
    """
    print(f"Downloading (gz, streaming to temp): {href}")
    with requests.get(href, stream=True) as resp:
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            with gzip.GzipFile(fileobj=resp.raw) as gz:
                while True:
                    chunk = gz.read(1024 * 1024)  
                    if not chunk:
                        break
                    tmp.write(chunk)

    print(f"Uploading decompressed file to s3://{BUCKET}/{key} from {tmp_path}")
    try:
        s3.upload_file(tmp_path, BUCKET, key)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print(f"Uploaded: s3://{BUCKET}/{key}")


def main():
    print(f"Fetching index page: {URL}")
    resp = requests.get(URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = soup.find_all("tr")

    files_processed = 0

    for row in rows:
        links = row.find_all("a")
        if not links:
            continue

        for link in links:
            href = link.get("href")
            if not href:
                continue

            href = requests.compat.urljoin(URL, href)

            if not href.endswith(".gz"):
                continue

            filename = href.split("/")[-1]
            if not filename:
                continue

            if filename.endswith(".gz"):
                s3_filename = filename[:-3]
            else:
                s3_filename = filename

            key = f"{S3_PREFIX}{s3_filename}"

            upload_gz_stream_to_s3(href, key)

            files_processed += 1
            if MAX_FILES is not None and files_processed >= MAX_FILES:
                print(f"Reached MAX_FILES={MAX_FILES}, stopping early.")
                return


if __name__ == "__main__":
    if not BUCKET:
        raise RuntimeError(
            "Set BUCKET_NAME env var (export BUCKET_NAME=...) before running."
        )
    main()