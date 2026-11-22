
import pandas as pd

# Load dataset and existing evaluation file
dataset = pd.read_csv("../data/issue_dataset.csv").fillna("")
evaluation = pd.read_csv("../data/duplicate_evaluation.csv").fillna("")

# Verified VS Code duplicate relationships
duplicate_pairs = {
    "310364": "308827",
    "310365": "308827",
    "310370": "308827",
    "310371": "308827",
}

# Verify all issue URLs exist in the FAISS metadata
metadata = pd.read_csv("../models/issue_faiss_metadata.csv").fillna("")
indexed_urls = set(metadata["issue_url"])

# Remove old VS Code evaluation entries.
# Preserve other entries, including Scikit-learn.
evaluation = evaluation[
    ~evaluation["exclude_url"].str.rstrip("/").str.split("/").str[-1].isin(
        duplicate_pairs.keys()
    )
]

new_rows = []

for query_id, original_id in duplicate_pairs.items():

    query_url = (
        "https://github.com/microsoft/vscode/issues/" + query_id
    )
    original_url = (
        "https://github.com/microsoft/vscode/issues/" + original_id
    )

    query_issue = dataset[
        dataset["issue_url"].str.rstrip("/") == query_url
    ]

    if query_issue.empty:
        print("Query issue missing:", query_id)
        continue

    if original_url not in indexed_urls:
        print("Original issue missing from FAISS:", original_id)
        continue

    query_text = query_issue.iloc[0]["text"]

    new_rows.append({
        "query": query_text,
        "exclude_url": query_url,
        "relevant_urls": original_url
    })

# Append verified VS Code pairs
evaluation = pd.concat(
    [evaluation, pd.DataFrame(new_rows)],
    ignore_index=True
)

# Save updated evaluation dataset
evaluation.to_csv("../data/duplicate_evaluation.csv", index=False)

print("\nEvaluation dataset updated!")
print("Total queries:", len(evaluation))
print(evaluation[["exclude_url", "relevant_urls"]].to_string(index=False))
