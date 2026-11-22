import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

ERRORS_PATH = ROOT_DIR / "models" / "repository_holdout_errors.csv"
OUTPUT_PATH = ROOT_DIR / "models" / "label_quality_audit.csv"


def main():
    df = pd.read_csv(ERRORS_PATH)

    # Focus on questions predicted as bugs
    sample = df[
        (df["actual"] == "question") &
        (df["predicted"] == "bug")
    ].sample(n=25, random_state=42)

    audit = sample[
        ["title", "labels", "actual", "predicted", "issue_url"]
    ].copy()

    audit["manual_category"] = ""
    audit["audit_notes"] = ""

    audit.to_csv(OUTPUT_PATH, index=False)

    print("Audit file created:", OUTPUT_PATH)
    print("Issues selected:", len(audit))


if __name__ == "__main__":
    main()