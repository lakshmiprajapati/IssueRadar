
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "issue_dataset.csv"
OUTPUT_PATH = ROOT / "models" / "duplicate_candidates.csv"

df = pd.read_csv(DATA_PATH).fillna("")

candidates = df[
    (df["repository"] == "microsoft/vscode")
    & df["labels"].str.contains(r"\*duplicate", case=False, regex=True)
].copy()

# Dataset URL lookup
url_to_row = {
    str(row["issue_url"]).rstrip("/"): row
    for _, row in df.iterrows()
}

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

token = os.getenv("GITHUB_TOKEN")
if token:
    session.headers["Authorization"] = f"Bearer {token}"

results = []
checked = 0

for _, candidate in candidates.iterrows():
    url = str(candidate["issue_url"]).rstrip("/")
    match = url.rsplit("/", 1)
    issue_number = match[-1]

    api_url = (
        f"https://api.github.com/repos/microsoft/vscode/"
        f"issues/{issue_number}/timeline"
    )

    try:
        response = session.get(
            api_url,
            params={"per_page": 100},
            timeout=20
        )

        if response.status_code in (403, 429):
            print("GitHub API rate limit reached. Stopping scan.")
            break

        if response.status_code != 200:
            print(f"Skipped #{issue_number}: HTTP {response.status_code}")
            continue

        checked += 1

        for event in response.json():
            source = event.get("source", {}).get("issue", {})
            if not source:
                continue

            source_url = str(source.get("html_url", "")).rstrip("/")
            source_labels = [
                label.get("name", "").lower()
                for label in source.get("labels", [])
            ]

            # Source issue must explicitly carry duplicate label
            if not any("duplicate" in label for label in source_labels):
                continue

            # Both issues must exist in our dataset
            if url not in url_to_row or source_url not in url_to_row:
                continue

            duplicate = url_to_row[source_url]
            original = url_to_row[url]

            results.append({
                "query": duplicate["text"],
                "exclude_url": source_url,
                "relevant_urls": url,
                "duplicate_title": duplicate["title"],
                "original_title": original["title"],
                "evidence": "Source issue has duplicate label and references candidate timeline"
            })

        time.sleep(0.3)

    except requests.RequestException as error:
        print(f"Error on issue #{issue_number}: {error}")

result_df = pd.DataFrame(results)

if not result_df.empty:
    result_df.drop_duplicates(
        subset=["exclude_url", "relevant_urls"],
        inplace=True
    )

result_df.to_csv(OUTPUT_PATH, index=False)

print("\n========== SCAN SUMMARY ==========")
print("Duplicate-labelled candidates:", len(candidates))
print("Timeline requests completed:", checked)
print("Dataset-matched pairs:", len(result_df))
print("Saved to:", OUTPUT_PATH)

if not result_df.empty:
    print(result_df[[
        "duplicate_title",
        "original_title",
        "exclude_url",
        "relevant_urls"
    ]].to_string(index=False))
