from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT_DIR / "data" / "issue_dataset.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    pandas_df = df[
        df["repository"] == "pandas-dev/pandas"
    ].copy()

    print("\n========== RAW LABEL DISTRIBUTION ==========")

    for category in sorted(pandas_df["category"].unique()):

        category_df = pandas_df[
            pandas_df["category"] == category
        ]

        print(f"\nCategory: {category}")
        print("Number of issues:", len(category_df))

        labels = (
            category_df["labels"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
        )

        print("\nOriginal GitHub labels:")
        print(labels.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()