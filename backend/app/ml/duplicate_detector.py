
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = ROOT_DIR / "data" / "issue_dataset.csv"
INDEX_PATH = ROOT_DIR / "models" / "issue_faiss.index"
METADATA_PATH = ROOT_DIR / "models" / "issue_faiss_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# STEP 1: BUILD FAISS INDEX
# --------------------------------------------------

def build_index():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    # Remove records without usable issue text
    df = df.dropna(subset=["text"]).copy()

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"] != ""].reset_index(drop=True)

    print("Total issues:", len(df))

    print("\nLoading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print("\nCreating FAISS index...")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Save FAISS index
    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # Save corresponding issue metadata
    metadata_columns = [
        "issue_id",
        "repository",
        "title",
        "category",
        "issue_url",
        "text"
    ]

    df[metadata_columns].to_csv(
        METADATA_PATH,
        index=False
    )

    # Clear cached resources so a rebuilt index is loaded next time
    load_duplicate_resources.cache_clear()

    print("\n========== INDEX BUILD COMPLETE ==========")

    print("Total indexed issues:", index.ntotal)
    print("Embedding dimensions:", dimension)

    print("FAISS index saved:", INDEX_PATH)
    print("Metadata saved:", METADATA_PATH)


# --------------------------------------------------
# STEP 2: LOAD AND CACHE RESOURCES
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_duplicate_resources():

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():

        raise FileNotFoundError(
            "FAISS index or metadata not found. Build the index first."
        )

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print("Loading issue metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print("Loading Sentence Transformer model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Duplicate detection resources loaded successfully.")

    return index, metadata, model


# --------------------------------------------------
# STEP 3: BATCH SEMANTIC SEARCH
# --------------------------------------------------


def search_similar_issues_batch(
    queries,
    exclude_urls=None,
    top_k=5,
    similarity_threshold=0.70
):

    from urllib.parse import urlparse

    if not queries:
        return []

    index, metadata, model = load_duplicate_resources()

    if exclude_urls is None:
        exclude_urls = [""] * len(queries)

    if len(exclude_urls) != len(queries):
        raise ValueError(
            "Number of exclude_urls must match number of queries."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1.")

    # Generate query embeddings
    query_embeddings = model.encode(
        queries,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype="float32"
    )

    # Search the complete index so same-repository candidates
    # are not missed behind results from other repositories.
    scores, indices = index.search(
        query_embeddings,
        index.ntotal
    )

    all_results = []

    for row_scores, row_indices, excluded_url in zip(
        scores,
        indices,
        exclude_urls
    ):

        results = []

        # Extract owner/repository from the original GitHub issue URL
        parsed_url = urlparse(excluded_url) if excluded_url else None
        url_parts = parsed_url.path.strip("/").split("/") if parsed_url else []

        query_repository = (
            "/".join(url_parts[:2])
            if len(url_parts) >= 2
            else None
        )

        for score, idx in zip(row_scores, row_indices):

            if idx == -1:
                continue

            similarity = float(score)

            # Apply minimum similarity threshold
            if similarity < similarity_threshold:
                continue

            issue = metadata.iloc[int(idx)]

            issue_url = str(issue["issue_url"])
            issue_repository = str(issue["repository"])

            # Exclude the original issue itself
            if excluded_url and issue_url.rstrip("/") == excluded_url.rstrip("/"):
                continue

            # Keep results from the same repository
            if query_repository and issue_repository != query_repository:
                continue

            results.append({
                "issue_id": int(issue["issue_id"]),
                "repository": issue_repository,
                "title": issue["title"],
                "category": issue["category"],
                "issue_url": issue_url,
                "similarity_score": round(similarity, 4)
            })

            if len(results) >= top_k:
                break

        all_results.append(results)

    return all_results




# --------------------------------------------------
# STEP 4: SINGLE ISSUE SEARCH
# --------------------------------------------------

def search_similar_issues(
    query,
    top_k=5
):

    results = search_similar_issues_batch(
        queries=[query],
        top_k=top_k
    )

    return results[0]


# --------------------------------------------------
# MAIN: BUILD INDEX
# --------------------------------------------------

if __name__ == "__main__":

    build_index()
