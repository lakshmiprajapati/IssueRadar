
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT_DIR / "data" / "issue_dataset.csv"

MODEL_DIR = ROOT_DIR / "models"

MODEL_FILE = MODEL_DIR / "issue_classifier.joblib"

METRICS_FILE = MODEL_DIR / "evaluation_metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run collect_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "issue_id",
        "repository",
        "text",
        "category"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove records without essential information
    df = df.dropna(
        subset=["repository", "issue_id", "text", "category"]
    )

    # Remove duplicate issues
    df = df.drop_duplicates(
        subset=["repository", "issue_id"]
    )

    # Normalize text and category values
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["repository"] = df["repository"].astype(str).str.strip()

    # Remove empty or very short issue descriptions
    df = df[df["text"].str.len() >= 20]

    if df.empty:
        raise ValueError(
            "No usable training samples found."
        )

    print("\n" + "=" * 60)
    print("ISSUERADAR - DATASET INFORMATION")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nRepository distribution:")
    print(df["repository"].value_counts())

    # Check identical text records
    duplicate_text_count = df.duplicated(
        subset=["text"]
    ).sum()

    print("\nIdentical text records:")
    print(duplicate_text_count)

    print("\nCategory distribution across repositories:")
    print(
        pd.crosstab(
            df["repository"],
            df["category"]
        ).to_string()
    )

    return df


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def train_model():

    df = load_dataset()

    X = df["text"]
    y = df["category"]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "At least two categories are required."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each category needs at least two samples "
            "for stratified splitting."
        )

    if len(df) < 20:
        raise ValueError(
            "Dataset is too small. Collect more issues."
        )

    # --------------------------------------------------
# GROUP-AWARE STRATIFIED TRAIN / TEST SPLIT
# --------------------------------------------------

    X = df["text"]
    y = df["category"]

    # Identical text records receive the same group ID
    groups = pd.factorize(X)[0]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining category distribution:")
    print(y_train.value_counts())

    print("\nTesting category distribution:")
    print(y_test.value_counts())

    print("\nTraining repository distribution:")
    print(df.loc[X_train.index, "repository"].value_counts())

    print("\nTesting repository distribution:")
    print(df.loc[X_test.index, "repository"].value_counts())

    # Verify there is no text overlap
    overlap = set(X_train).intersection(set(X_test))

    print("\nText overlap between training and testing:", len(overlap))

    if overlap:
        raise ValueError("Data leakage detected: identical text in both splits!")

    # --------------------------------------------------
    # MACHINE LEARNING PIPELINE
    # --------------------------------------------------

    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000,
                min_df=1,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )

    ])

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print("\nTraining TF-IDF + Logistic Regression model...")

    model.fit(X_train, y_train)

    print("\nTraining completed successfully!")

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    predictions = model.predict(X_test)

    classes = model.classes_

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ISSUERADAR - MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print("Class order:", classes.tolist())
    print(matrix)

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    # --------------------------------------------------
    # SAVE EVALUATION METRICS
    # --------------------------------------------------

    metrics = {

        "model_name": "TF-IDF + Logistic Regression",

        "accuracy": float(accuracy),

        "macro_precision": float(macro_precision),

        "macro_recall": float(macro_recall),

        "macro_f1": float(macro_f1),

        "weighted_f1": float(weighted_f1),

        "balanced_accuracy": float(balanced_accuracy),

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),

        "classes": classes.tolist(),

        "dataset": {

            "total_samples": int(len(df)),

            "training_samples": int(len(X_train)),

            "testing_samples": int(len(X_test)),

            "duplicate_text_records": int(
                df.duplicated(subset=["text"]).sum()
            ),

            "category_distribution": {
                str(k): int(v)
                for k, v in class_counts.items()
            },

            "repository_distribution": {
                str(k): int(v)
                for k, v in df["repository"].value_counts().items()
            },

            "training_repository_distribution": {
                str(k): int(v)
                for k, v in df.loc[
                    X_train.index, "repository"
                ].value_counts().items()
            },

            "testing_repository_distribution": {
                str(k): int(v)
                for k, v in df.loc[
                    X_test.index, "repository"
                ].value_counts().items()
            }

        },

        "configuration": {

            "test_size": TEST_SIZE,

            "random_state": RANDOM_STATE,

            "split_strategy": "stratified_random_split",

            "class_weight": "balanced",

            "ngram_range": [1, 2],

            "max_features": 20000,

            "min_df": 1,

            "sublinear_tf": True,

            "max_iter": 1000

        }

    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nModel:   {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    train_model()
