
import re
from functools import lru_cache
from pathlib import Path

import joblib


# Project root: IssueRadar/
ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT_DIR / "models" / "issue_classifier.joblib"


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# TEXT CLEANING
# Keep consistent with collect_dataset.py
# --------------------------------------------------

def clean_text(text):

    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r"https?://\S+", " ", text)

    text = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(r"<[^>]*>", " ", text)

    text = re.sub(r"```[a-zA-Z0-9_+-]*", " ", text)
    text = text.replace("```", " ")
    text = text.replace("`", " ")

    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# PREDICT ISSUE CATEGORY
# --------------------------------------------------

def predict_issue(title: str, body: str = ""):

    cleaned_title = clean_text(title)
    cleaned_body = clean_text(body)

    combined_text = f"{cleaned_title}. {cleaned_body}".strip()

    if len(combined_text) < 20:
        raise ValueError(
            "Issue title and description must contain at least 20 characters."
        )

    model = load_model()

    # Predict category
    predicted_category = model.predict([combined_text])[0]

    # Get probability for each category
    probabilities = model.predict_proba([combined_text])[0]

    categories = model.classes_

    confidence_scores = {
        category: round(float(probability), 4)
        for category, probability in zip(categories, probabilities)
    }

    return {
        "predicted_category": str(predicted_category),
        "confidence": round(
            float(max(probabilities)), 4
        ),
        "probabilities": confidence_scores
    }
