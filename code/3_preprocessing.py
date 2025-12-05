
import os
import json
import tempfile

import boto3
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)

# NLTK stopwords
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        _ = nltk_stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
except Exception as e:
    print("WARNING: NLTK stopwords unavailable:", e)
    NLTK_STOPWORDS = set()

#   pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("WARNING: sentence-transformers not installed. Embeddings method will fail.")


HTML_ARTIFACTS = {
    "br",
    "brbr",
    "nbsp",
    "amp",
    "lt",
    "gt",
}

DOMAIN_STOPWORDS = {
    # generic praise / filler
    "great", "good", "nice", "love", "really",
    "amazing", "awesome", "excellent", "perfect",
    "best", "better", "super",

    # generic review verbs / meta
    "works", "worked", "using", "use", "used",
    "does", "doesn", "did", "didn", "don",
    "got", "buy", "bought",
    "review", "reviews", "received",
    "purchase", "purchased", "money", "worth",
    "recommend", "recommended", "recommending",
    "highly",  # usually part of "highly recommend"

    # vague filler
    "like", "just", "try", "trying",
    "know", "think", "say", "says", "said",
    "want", "wanted", "seems", "seemed",
    "thing", "things", "stuff",
    "way", "time", "times", "years", "year",
    "little", "bit",
    "look", "looks", "looking",
    "easy", "easily",
    "soft", "pretty",
    "different", "same",

    # meta/about-topic words you probably don't care about
    "product", "products",
}


ALL_STOPWORDS = (
    set(ENGLISH_STOP_WORDS)
    .union(NLTK_STOPWORDS)
    .union(HTML_ARTIFACTS)
    .union(DOMAIN_STOPWORDS)
)
ALL_STOPWORDS_LIST = list(ALL_STOPWORDS)


class AmazonPreprocessor:
    def __init__(
        self,
        bucket_name: str,
        clean_prefix: str = "amazon_clean/",
        vector_prefix: str = "amazon_vectors/",
        text_field: str = "text",  
        region_name: str | None = None,
    ):
        """
        bucket_name: S3 bucket name (e.g. 'amazonreviewsnlp')
        clean_prefix: where cleaned jsonl files live (e.g. 'amazon_clean/')
        vector_prefix: base prefix for vectors (e.g. 'amazon_vectors/')
        text_field: which JSON field to use as text (for you: 'text')
        """
        self.bucket = bucket_name
        self.clean_prefix = clean_prefix
        self.vector_prefix = vector_prefix.rstrip("/") + "/"
        self.text_field = text_field

        self.s3 = boto3.client("s3", region_name=region_name)

    # -----------------------
    # Quick preview
    # -----------------------
    def preview_head(self, n_files: int = 1, n_lines: int = 5):
        """
        Quick preview of the cleaned data in S3.

        - Looks under self.clean_prefix (e.g. 'amazon_clean/')
        - Reads up to n_files files
        - Prints up to n_lines JSON records per file
        """
        resp = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.clean_prefix,
            MaxKeys=n_files,
        )

        contents = resp.get("Contents", [])
        if not contents:
            print("No cleaned files found in", self.clean_prefix)
            return

        print(f"Previewing up to {len(contents)} file(s), {n_lines} line(s) each...")

        for i, obj in enumerate(contents):
            key = obj["Key"]
            print(f"\n=== File {i+1}: s3://{self.bucket}/{key} ===")

            s3_obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = s3_obj["Body"]

            line_count = 0
            for raw_line in body.iter_lines():
                if not raw_line:
                    continue

                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                print("Full record:", record)
                print(f"{self.text_field}:", record.get(self.text_field))
                print("-" * 60)

                line_count += 1
                if line_count >= n_lines:
                    break

    # -----------------------
    # Internal: iterate documents
    # -----------------------
    def _iter_documents(self, max_docs: int | None = None):
        """
        Generator over text_field from all files in amazon_clean/.
        Yields raw text (no further cleaning here).
        """
        continuation_token = None
        doc_count = 0

        while True:
            kwargs = {
                "Bucket": self.bucket,
                "Prefix": self.clean_prefix,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            resp = self.s3.list_objects_v2(**kwargs)
            contents = resp.get("Contents", [])
            if not contents:
                break

            for obj in contents:
                key = obj["Key"]
                if not (key.endswith(".jsonl") or key.endswith(".json")):
                    continue

                s3_obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                body = s3_obj["Body"]

                for raw_line in body.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    text = record.get(self.text_field)
                    if not isinstance(text, str):
                        continue

                    yield text
                    doc_count += 1

                    if max_docs is not None and doc_count >= max_docs:
                        return

            if resp.get("IsTruncated"):
                continuation_token = resp["NextContinuationToken"]
            else:
                break

    # -----------------------
    # CountVectorizer
    # -----------------------
    def build_count_vectors(
        self,
        max_docs: int | None = None,
        max_features: int = 40000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 20,
        max_df: float = 0.5,
    ):
        """
        Create CountVectorizer representations from amazon_clean/
        and upload to S3 under amazon_vectors/countvectorizer/.

        - Uses sklearn + NLTK stopwords + custom stopwords
        - Drops HTML artifacts like 'br', 'brbr'
        - Ignores tokens shorter than 3 characters (only a–z words len >= 3)
        - Uses unigrams + bigrams with frequency filtering
        """
        print("Collecting documents for CountVectorizer...")
        docs = list(self._iter_documents(max_docs=max_docs))
        print(f"Collected {len(docs)} documents.")

        if not docs:
            print("No documents found. Aborting CountVectorizer.")
            return

        print("Fitting CountVectorizer...")
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=ALL_STOPWORDS_LIST,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            min_df=min_df,
            max_df=max_df,
        )
        X = vectorizer.fit_transform(docs)
        print(f"Sparse matrix shape: {X.shape}")

        out_prefix = self.vector_prefix + "countvectorizer/"
        os.makedirs("/tmp/amazon_vectors/countvectorizer", exist_ok=True)

        vec_path = "/tmp/amazon_vectors/countvectorizer/vectors.npz"
        vocab_path = "/tmp/amazon_vectors/countvectorizer/vocab.json"

        print("Saving CountVectorizer outputs locally...")
        sparse.save_npz(vec_path, X)

        # Cast NumPy ints to plain Python ints for JSON
        vocab = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)

        print("Uploading CountVectorizer outputs to S3...")
        self.s3.upload_file(vec_path, self.bucket, out_prefix + "vectors.npz")
        self.s3.upload_file(vocab_path, self.bucket, out_prefix + "vocab.json")

        print(f"Done. Stored under s3://{self.bucket}/{out_prefix}")

    # -----------------------
    # TF-IDF
    # -----------------------
    def build_tfidf_vectors(
        self,
        max_docs: int | None = None,
        max_features: int = 40000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 20,
        max_df: float = 0.5,
    ):
        """
        Create TF-IDF representations from amazon_clean/
        and upload to S3 under amazon_vectors/tfidf/.

        - Uses sklearn + NLTK stopwords + custom stopwords
        - Drops HTML artifacts like 'br', 'brbr'
        - Ignores tokens shorter than 3 characters
        - Uses unigrams + bigrams with frequency filtering
        """
        print("Collecting documents for TfidfVectorizer...")
        docs = list(self._iter_documents(max_docs=max_docs))
        print(f"Collected {len(docs)} documents.")

        if not docs:
            print("No documents found. Aborting TF-IDF.")
            return

        print("Fitting TfidfVectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=ALL_STOPWORDS_LIST,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            min_df=min_df,
            max_df=max_df,
        )
        X = vectorizer.fit_transform(docs)
        print(f"Sparse matrix shape: {X.shape}")

        out_prefix = self.vector_prefix + "tfidf/"
        os.makedirs("/tmp/amazon_vectors/tfidf", exist_ok=True)

        vec_path = "/tmp/amazon_vectors/tfidf/vectors.npz"
        vocab_path = "/tmp/amazon_vectors/tfidf/vocab.json"

        print("Saving TF-IDF outputs locally...")
        sparse.save_npz(vec_path, X)

        vocab = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)

        print("Uploading TF-IDF outputs to S3...")
        self.s3.upload_file(vec_path, self.bucket, out_prefix + "vectors.npz")
        self.s3.upload_file(vocab_path, self.bucket, out_prefix + "vocab.json")

        print(f"Done. Stored under s3://{self.bucket}/{out_prefix}")

    # -----------------------
    # Embeddings
    # -----------------------
    def build_embeddings(
        self,
        max_docs: int | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        """
        Create dense embeddings from amazon_clean/ using SentenceTransformer
        and upload to S3 under amazon_vectors/embeddings/.
        """
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` in your venv."
            )

        print(f"Loading sentence-transformers model: {model_name}")
        model = SentenceTransformer(model_name)

        print("Collecting documents for embeddings...")
        docs = list(self._iter_documents(max_docs=max_docs))
        print(f"Collected {len(docs)} documents.")

        if not docs:
            print("No documents found. Aborting embeddings.")
            return

        print("Encoding documents into embeddings...")
        embeddings = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")

        out_prefix = self.vector_prefix + "embeddings/"
        os.makedirs("/tmp/amazon_vectors/embeddings", exist_ok=True)

        emb_path = "/tmp/amazon_vectors/embeddings/embeddings.npy"

        print("Saving embeddings locally...")
        np.save(emb_path, embeddings)

        print("Uploading embeddings to S3...")
        self.s3.upload_file(emb_path, self.bucket, out_prefix + "embeddings.npy")

        print(f"Done. Stored under s3://{self.bucket}/{out_prefix}")


if __name__ == "__main__":
    bucket = os.environ.get("BUCKET_NAME", "amazonreviewsnlp")

    pre = AmazonPreprocessor(
        bucket_name=bucket,
        clean_prefix="amazon_clean/",
        vector_prefix="amazon_vectors/",
        text_field="text",
    )

    pre.preview_head(n_files=1, n_lines=3)

    pre.build_count_vectors(max_docs=10000)
    pre.build_tfidf_vectors(max_docs=10000)
    pre.build_embeddings(max_docs=5000)
