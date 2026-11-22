import json
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = ROOT_DIR / "data" / "issue_dataset.csv"
METRICS_PATH = (
    ROOT_DIR / "models" / "repository_holdout_pandas_metrics.json"
)

TEST_REPOSITORY = "pandas-dev/pandas"


def main():
    df = pd.read_csv(DATASET_PATH)

    required_columns = ["repository", "text", "category"]
    df = df.dropna(subset=required_columns).copy()

    df["repository"] = df["repository"].str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].str.strip()

    df = df[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset=["repository", "issue_id"])

    train_df = df[df["repository"] != TEST_REPOSITORY].copy()
    test_df = df[df["repository"] == TEST_REPOSITORY].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing dataset is empty.")

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_test = test_df["text"]
    y_test = test_df["category"]

    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("\nTraining repositories:")
    print(train_df["repository"].value_counts())

    print("\nTesting repository:")
    print(test_df["repository"].value_counts())

    print("\nTraining category distribution:")
    print(y_train.value_counts())

    print("\nTesting category distribution:")
    print(y_test.value_counts())

    missing_classes = set(y_test.unique()) - set(y_train.unique())

    if missing_classes:
        raise ValueError(
            f"Classes present in test but missing in training: {missing_classes}"
        )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000,
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    # Inspect misclassified examples
    results = test_df.copy()

    results["actual"] = y_test.values
    results["predicted"] = predictions

    errors = results[
        results["actual"] != results["predicted"]
    ].copy()

    print("\n========== MISCLASSIFIED EXAMPLES ==========")
    print("Total misclassified:", len(errors))

    # Focus on questions incorrectly predicted as bugs
    question_to_bug = errors[
        (errors["actual"] == "question") &
        (errors["predicted"] == "bug")
    ]

    print("\nQuestions incorrectly classified as bugs:")
    print("Count:", len(question_to_bug))

    for _, row in question_to_bug.head(10).iterrows():
        print("\n--------------------------------")
        print("Title:", row["title"])
        print("Actual:", row["actual"])
        print("Predicted:", row["predicted"])
        print("Text:", row["text"][:500])
        print("URL:", row["issue_url"])

    # Save all errors for further investigation
    errors.to_csv(
        ROOT_DIR / "models" / "repository_holdout_errors.csv",
        index=False
    )

    print("\nAll misclassified examples saved.")

    accuracy = accuracy_score(y_test, predictions)
    macro_precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test, predictions, labels=labels
    )

    print("\n========== REPOSITORY HOLDOUT RESULTS ==========")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )

    print("\nConfusion Matrix (labels:", labels, ")")
    print(matrix)

    metrics = {
        "experiment": "repository_holdout",
        "test_repository": TEST_REPOSITORY,
        "training_samples": len(train_df),
        "testing_samples": len(test_df),
        "training_repositories": train_df["repository"].value_counts().to_dict(),
        "testing_categories": y_test.value_counts().to_dict(),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nMetrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    main()