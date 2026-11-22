
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = ROOT / "data" / "duplicate_evaluation.csv"
METADATA_PATH = ROOT / "models" / "issue_faiss_metadata.csv"

# Load existing evaluation data and FAISS metadata
df = pd.read_csv(EVALUATION_PATH).fillna("")
metadata = pd.read_csv(METADATA_PATH).fillna("")

# Only relevant issues present in FAISS index can be evaluated
indexed_urls = set(metadata["issue_url"].astype(str))

cleaned = []

for exclude_url, group in df.groupby("exclude_url"):

    relevant_urls = set()

    for urls in group["relevant_urls"]:
        for url in str(urls).split(";"):
            url = url.strip()

            if url in indexed_urls:
                relevant_urls.add(url)

    # Skip queries without any indexed relevant issue
    if not relevant_urls:
        print("Skipped:", exclude_url)
        continue

    cleaned.append({
        "query": group.iloc[0]["query"],
        "exclude_url": exclude_url,
        "relevant_urls": ";".join(sorted(relevant_urls))
    })

result = pd.DataFrame(
    cleaned,
    columns=["query", "exclude_url", "relevant_urls"]
)

result.to_csv(EVALUATION_PATH, index=False)

print("\n========== CLEANED DATASET ==========")
print("Unique queries:", len(result))

for _, row in result.iterrows():
    print("\nQuery:", row["exclude_url"])
    print("Relevant URLs:", row["relevant_urls"])

print("\nUpdated:", EVALUATION_PATH)
