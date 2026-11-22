
from pathlib import Path

import pandas as pd

from app.ml.duplicate_detector import search_similar_issues_batch


# ==========================================
# CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT_DIR / "data" / "duplicate_evaluation.csv"

TOP_K = 5


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate():

    print("Loading duplicate evaluation dataset...")

    if not CSV_PATH.exists():
        print(f"Evaluation file not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH).fillna("")

    required_columns = [
        "query",
        "exclude_url",
        "relevant_urls"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Remove incomplete rows
    df = df[
        (df["query"].str.strip() != "")
        & (df["relevant_urls"].str.strip() != "")
    ].copy()

    if df.empty:
        print("No verified duplicate pairs found.")
        return

    print("Queries available:", len(df))

    # ======================================
    # GENERATE EMBEDDINGS & SEARCH
    # ======================================

    print("\nGenerating embeddings for queries...")

    results = search_similar_issues_batch(
        queries=df["query"].tolist(),
        exclude_urls=df["exclude_url"].tolist(),
        top_k=TOP_K
    )

    print("Search completed successfully!")

    # ======================================
    # METRICS STORAGE
    # ======================================

    precisions = []
    recalls = []
    reciprocal_ranks = []

    # ======================================
    # QUERY-WISE ERROR ANALYSIS
    # ======================================

    for i, (row, retrieved) in enumerate(
        zip(df.to_dict("records"), results),
        start=1
    ):

        relevant = {
            url.strip()
            for url in row["relevant_urls"].split(";")
            if url.strip()
        }

        retrieved_urls = [
            item["issue_url"]
            for item in retrieved
        ]

        found = relevant.intersection(set(retrieved_urls))

        precision = len(found) / TOP_K
        recall = len(found) / len(relevant)

        reciprocal_rank = 0

        for rank, url in enumerate(retrieved_urls, start=1):
            if url in relevant:
                reciprocal_rank = 1 / rank
                break

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        # ==================================
        # PRINT QUERY DETAILS
        # ==================================

        print("\n" + "=" * 65)
        print(f"QUERY {i}")
        print("=" * 65)

        print("Query issue:", row["exclude_url"])
        print("Expected duplicate(s):")

        for url in relevant:
            print("  ", url)

        print("\nSearch Status:")

        if found:
            print("FOUND - Relevant issue retrieved")
        else:
            print("NOT FOUND - Relevant issue missing from Top 5")

        print("\nTop 5 Retrieved Issues:")

        if not retrieved:
            print("No results returned.")

        for rank, item in enumerate(retrieved, start=1):

            is_relevant = item["issue_url"] in relevant

            marker = " <-- RELEVANT" if is_relevant else ""

            print(f"\nRank: {rank}")
            print("Title:", item["title"])
            print(f"Similarity Score: {item['similarity_score']:.4f}")
            print("URL:", item["issue_url"] + marker)

        print("\nQuery Metrics:")
        print(f"Precision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

    # ======================================
    # FINAL EVALUATION
    # ======================================

    print("\n" + "=" * 65)
    print("FINAL DUPLICATE DETECTION EVALUATION")
    print("=" * 65)

    print("Total queries evaluated:", len(df))

    print(
        f"Precision@{TOP_K}: "
        f"{sum(precisions) / len(precisions):.4f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{sum(recalls) / len(recalls):.4f}"
    )

    print(
        f"Mean Reciprocal Rank (MRR): "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
    )

    successful = sum(1 for r in recalls if r > 0)

    print(
        f"\nQueries with at least one relevant result: "
        f"{successful}/{len(df)}"
    )

    print("\nEvaluation completed!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    evaluate()
