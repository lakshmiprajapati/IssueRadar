import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = ROOT_DIR / "data" / "issue_dataset.csv"


BUG_KEYWORDS = [
    "bug",
    "error",
    "crash",
    "fail",
    "failure",
    "incorrect",
    "wrong",
    "broken",
    "not working",
    "unexpected",
    "regression",
]


def main():
    df = pd.read_csv(DATASET_PATH)

    pandas_df = df[
        df["repository"] == "pandas-dev/pandas"
    ].copy()

    questions = pandas_df[
        pandas_df["category"] == "question"
    ].copy()

    pattern = r"\b(?:" + "|".join(
        re.escape(word) for word in BUG_KEYWORDS
    ) + r")\b"

    questions["bug_keyword_match"] = (
        questions["title"]
        .fillna("")
        .str.lower()
        .str.contains(pattern, regex=True)
    )

    matched = questions[
        questions["bug_keyword_match"]
    ]

    print("Total Pandas questions:", len(questions))
    print("Questions with bug-related keywords:", len(matched))

    print(
        "Percentage:",
        round(len(matched) / len(questions) * 100, 2),
        "%"
    )

    print("\nExamples:")

    for _, row in matched.head(20).iterrows():
        print("\nTitle:", row["title"])
        print("Labels:", row["labels"])
        print("URL:", row["issue_url"])


if __name__ == "__main__":
    main()