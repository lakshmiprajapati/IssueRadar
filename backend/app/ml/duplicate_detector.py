
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

METADATA_PATH = ROOT_DIR / "models" / "issue_faiss_metadata.csv"


# --------------------------------------------------
# LOAD AND CACHE RESOURCES
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_duplicate_resources():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Issue metadata not found: {METADATA_PATH}"
        )

    print("Loading issue metadata...")

    metadata = pd.read_csv(
        METADATA_PATH,
        usecols=[
            "issue_id",
            "repository",
            "title",
            "category",
            "issue_url",
            "text"
        ]
    )

    metadata = metadata.fillna("")

    metadata["text"] = metadata["text"].astype(str).str.strip()

    metadata = metadata[
        metadata["text"] != ""
    ].reset_index(drop=True)

    if metadata.empty:
        raise ValueError("No usable issue records found.")

    print("Building lightweight TF-IDF index...")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
        dtype=float
    )

    issue_matrix = vectorizer.fit_transform(
        metadata["text"]
    )

    print("Duplicate detection resources loaded.")
    print("Indexed issues:", len(metadata))
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    return metadata, vectorizer, issue_matrix


# --------------------------------------------------
# BATCH SIMILARITY SEARCH
# --------------------------------------------------

def search_similar_issues_batch(
    queries,
    exclude_urls=None,
    top_k=5,
    similarity_threshold=0.15
):

    if not queries:
        return []

    if exclude_urls is None:
        exclude_urls = [""] * len(queries)

    if len(exclude_urls) != len(queries):
        raise ValueError(
            "Number of exclude_urls must match number of queries."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "Similarity threshold must be between 0 and 1."
        )

    metadata, vectorizer, issue_matrix = load_duplicate_resources()

    query_texts = [
        str(query) if query else ""
        for query in queries
    ]

    query_matrix = vectorizer.transform(query_texts)

    all_results = []

    for query_vector, excluded_url in zip(
        query_matrix,
        exclude_urls
    ):

        results = []

        parsed_url = urlparse(excluded_url) if excluded_url else None

        url_parts = (
            parsed_url.path.strip("/").split("/")
            if parsed_url else []
        )

        query_repository = (
            "/".join(url_parts[:2])
            if len(url_parts) >= 2
            else None
        )

        # Sparse cosine similarity:
        # TF-IDF vectors are L2 normalized by default.
        similarity_scores = (
            query_vector @ issue_matrix.T
        ).toarray().ravel()

        # Consider highest-scoring candidates first.
        candidate_indices = similarity_scores.argsort()[::-1]

        for idx in candidate_indices:

            similarity = float(similarity_scores[idx])

            if similarity < similarity_threshold:
                break

            issue = metadata.iloc[int(idx)]

            issue_url = str(issue["issue_url"])
            issue_repository = str(issue["repository"])

            # Exclude the original issue.
            if (
                excluded_url
                and issue_url.rstrip("/") == excluded_url.rstrip("/")
            ):
                continue

            # Restrict results to the same repository.
            if (
                query_repository
                and issue_repository != query_repository
            ):
                continue

            try:
                issue_id = int(issue["issue_id"])
            except (ValueError, TypeError):
                issue_id = 0

            results.append({
                "issue_id": issue_id,
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
# SINGLE ISSUE SEARCH
# --------------------------------------------------

def search_similar_issues(query, top_k=5):

    results = search_similar_issues_batch(
        queries=[query],
        top_k=top_k
    )

    return results[0]
