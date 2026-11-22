import csv
import os
import re
import time
from pathlib import Path

import httpx


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"
DATA_FILE = DATA_DIR / "issue_dataset.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# GITHUB CONFIGURATION
# --------------------------------------------------

GITHUB_API = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


REPOSITORIES = [
    "fastapi/fastapi",
    "pallets/flask",
    "django/django",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "microsoft/vscode"
]

PER_PAGE = 100

# Maximum issues fetched per matching label: 500
MAX_PAGES_PER_LABEL = 5

# Maximum pages used to discover repository labels
MAX_LABEL_PAGES = 10

MAX_RETRIES = 3


# --------------------------------------------------
# LABEL MAPPING
# --------------------------------------------------

LABEL_KEYWORDS = {
    "bug": [
        "bug",
        "defect",
        "crash",
        "regression",
        "bug report"
    ],

    "feature_request": [
        "enhancement",
        "feature request",
        "feature-request"
    ],

    "documentation": [
        "documentation",
        "docs",
        "doc"
    ],

    "performance": [
        "performance",
        "perf",
        "optimization"
    ],

    "question": [
        "question",
        "questions"
    ]
}


def normalize_label(label_name):
    """Normalize punctuation and separators in a label."""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        label_name.lower()
    ).strip()


def classify_labels(labels):
    """
    Classify an issue using its labels.

    Returns a category only when exactly one category
    matches. Ambiguous and unknown labels are rejected.
    """

    matched_categories = set()

    for label in labels:

        label_name = label.get("name", "")

        if not label_name:
            continue

        normalized = normalize_label(label_name)

        for category, keywords in LABEL_KEYWORDS.items():

            for keyword in keywords:

                normalized_keyword = normalize_label(keyword)

                pattern = (
                    r"\b"
                    + re.escape(normalized_keyword)
                    + r"\b"
                )

                if re.search(pattern, normalized):
                    matched_categories.add(category)

    if len(matched_categories) == 1:
        return next(iter(matched_categories))

    return None


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------



def clean_text(text):

    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Markdown image syntax
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        r"\1",
        text
    )

    # Preserve link text, remove link destinations
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove HTML tags while preserving surrounding text
    text = re.sub(r"<[^>]*>", " ", text)

    # Preserve fenced code contents, remove fence markers
    text = re.sub(r"```[a-zA-Z0-9_+-]*", " ", text)
    text = text.replace("```", " ")

    # Preserve inline code, remove backtick formatting
    text = text.replace("`", " ")

    # Remove user mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# GITHUB API REQUEST HANDLING
# --------------------------------------------------

def github_get(client, endpoint, params=None):

    for attempt in range(MAX_RETRIES):

        try:

            response = client.get(
                endpoint,
                headers=HEADERS,
                params=params
            )

            # Handle rate limits
            if response.status_code in (403, 429):

                remaining = response.headers.get(
                    "X-RateLimit-Remaining"
                )

                # Primary rate limit exhausted
                if (
                    response.status_code == 403
                    and remaining == "0"
                ):

                    reset_time = response.headers.get(
                        "X-RateLimit-Reset",
                        "unknown"
                    )

                    raise RuntimeError(
                        "GitHub API primary rate limit exhausted. "
                        f"Reset timestamp: {reset_time}. "
                        "Retry later or configure GITHUB_TOKEN."
                    )

                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = 2 ** (attempt + 1)

                print(
                    f"GitHub rate limit encountered. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)
                continue

            # Temporary server errors
            if response.status_code in (500, 502, 503, 504):

                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()

                delay = 2 ** (attempt + 1)

                print(
                    f"GitHub server error {response.status_code}. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)
                continue

            response.raise_for_status()

            return response.json()

        except httpx.RequestError as error:

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"GitHub network request failed: {error}"
                ) from error

            delay = 2 ** (attempt + 1)

            print(
                f"Network error. Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    return []


# --------------------------------------------------
# DISCOVER REPOSITORY LABELS
# --------------------------------------------------

def fetch_repository_labels(client, repository):

    endpoint = f"{GITHUB_API}/repos/{repository}/labels"

    matching_labels = {}

    print(f"\nDiscovering labels: {repository}")

    for page in range(1, MAX_LABEL_PAGES + 1):

        labels = github_get(
            client,
            endpoint,
            params={
                "per_page": PER_PAGE,
                "page": page
            }
        )

        if not labels:
            break

        for label in labels:

            label_name = label.get("name", "")

            if not label_name:
                continue

            category = classify_labels([
                {"name": label_name}
            ])

            if category:

                matching_labels.setdefault(
                    category,
                    []
                ).append(label_name)

        print(
            f"Label discovery page {page}: "
            f"{len(labels)} labels"
        )

        if len(labels) < PER_PAGE:
            break

    for category, labels in matching_labels.items():
        print(f"{category}: {labels}")

    return matching_labels


# --------------------------------------------------
# FETCH LABEL-SPECIFIC ISSUES
# --------------------------------------------------

def fetch_repository_issues(client, repository):

    all_issues = []

    print(f"\nCollecting: {repository}")

    matching_labels = fetch_repository_labels(
        client,
        repository
    )

    if not matching_labels:

        print("No matching repository labels found.")

        return all_issues

    for category, labels in matching_labels.items():

        for label_name in labels:

            print(
                f"\nFetching {category} issues "
                f"using label: {label_name}"
            )

            endpoint = (
                f"{GITHUB_API}/repos/{repository}/issues"
            )

            for page in range(
                1,
                MAX_PAGES_PER_LABEL + 1
            ):

                issues = github_get(
                    client,
                    endpoint,
                    params={
                        "state": "all",
                        "labels": label_name,
                        "per_page": PER_PAGE,
                        "page": page,
                        "sort": "created",
                        "direction": "desc"
                    }
                )

                if not issues:
                    break

                for issue in issues:

                    # Exclude pull requests
                    if "pull_request" in issue:
                        continue

                    # Classify using all issue labels.
                    # Reject ambiguous examples.
                    actual_category = classify_labels(
                        issue.get("labels", [])
                    )

                    if actual_category is None:
                        continue

                    title = clean_text(
                        issue.get("title", "")
                    )

                    body = clean_text(
                        issue.get("body", "")
                    )

                    combined_text = (
                        f"{title}. {body}".strip()
                    )

                    if len(combined_text) < 20:
                        continue

                    all_issues.append({

                        "issue_id": issue["id"],

                        "repository": repository,

                        "title": title,

                        "body": body,

                        "text": combined_text,

                        "category": actual_category,

                        "labels": ", ".join(
                            label.get("name", "")
                            for label in issue.get("labels", [])
                            if label.get("name")
                        ),

                        "issue_url": issue.get("html_url", "")
                    })

                print(
                    f"Label '{label_name}' | "
                    f"Page {page} | "
                    f"Fetched: {len(issues)}"
                )

                if len(issues) < PER_PAGE:
                    break

                time.sleep(0.2)

    # Remove duplicate issues within repository
    unique_issues = {
        issue["issue_id"]: issue
        for issue in all_issues
    }

    final_issues = list(unique_issues.values())

    print(
        f"\nUsable labelled issues from "
        f"{repository}: {len(final_issues)}"
    )

    return final_issues


# --------------------------------------------------
# MAIN DATASET COLLECTION
# --------------------------------------------------

def main():

    collected_data = []

    # Load existing dataset
    existing_data = []

    if DATA_FILE.exists():

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            existing_data = list(
                csv.DictReader(file)
            )

    print(
        f"\nExisting records: {len(existing_data)}"
    )

    # Fetch issues from all repositories
    with httpx.Client(timeout=30.0) as client:

        for repository in REPOSITORIES:

            repository_issues = fetch_repository_issues(
                client,
                repository
            )

            collected_data.extend(repository_issues)

    # Deduplicate existing and newly collected records
    unique_data = {}

    for issue in existing_data:

        issue_id = str(issue["issue_id"])

        unique_data[issue_id] = issue

    previous_count = len(unique_data)

    new_records = 0

    for issue in collected_data:

        issue_id = str(issue["issue_id"])

        if issue_id not in unique_data:

            unique_data[issue_id] = issue

            new_records += 1

    final_data = list(unique_data.values())

    # Save dataset
    fieldnames = [
        "issue_id",
        "repository",
        "title",
        "body",
        "text",
        "category",
        "labels",
        "issue_url"
    ]

    temp_file = DATA_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(final_data)

    # Replace original only after successful writing
    temp_file.replace(DATA_FILE)

    # Calculate statistics
    category_counts = {}
    repository_counts = {}

    for issue in final_data:

        category = issue["category"]
        repository = issue["repository"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

        repository_counts[repository] = (
            repository_counts.get(repository, 0) + 1
        )

    # Display report
    print("\n" + "=" * 55)
    print("ISSUERADAR DATASET COLLECTION REPORT")
    print("=" * 55)

    print(f"Previously stored: {previous_count}")
    print(f"Fetched this run: {len(collected_data)}")
    print(f"New records added: {new_records}")
    print(f"Total unique records: {len(final_data)}")

    print("\nCATEGORY DISTRIBUTION")

    for category in LABEL_KEYWORDS:

        print(
            f"{category}: "
            f"{category_counts.get(category, 0)}"
        )

    print("\nREPOSITORY DISTRIBUTION")

    for repository, count in sorted(
        repository_counts.items()
    ):

        print(f"{repository}: {count}")

    print("\nDataset saved successfully!")
    print(f"Location: {DATA_FILE}")



def test_clean_text():

    sample = """
    This is a bug report.

    ```python
    def calculate():
        raise ValueError("Invalid input")
    ```

    Expected: 200
    Actual: 500

    Read [the documentation](https://example.com/docs).

    Contact @developer.
    """

    cleaned = clean_text(sample)

    assert "ValueError" in cleaned
    assert "Invalid input" in cleaned
    assert "Expected: 200" in cleaned
    assert "Actual: 500" in cleaned
    assert "documentation" in cleaned
    assert "example.com" not in cleaned
    assert "@developer" not in cleaned

    print("All text cleaning tests passed!")

if __name__ == "__main__":
    test_clean_text()

# --------------------------------------------------
# TEST LABEL CLASSIFICATION
# --------------------------------------------------

def test_label_classification():

    test_cases = [

        ([{"name": "bug"}], "bug"),

        ([{"name": "type: bug"}], "bug"),

        ([{"name": "regression"}], "bug"),

        ([{"name": "enhancement"}], "feature_request"),

        ([{"name": "feature request"}], "feature_request"),

        ([{"name": "documentation"}], "documentation"),

        ([{"name": "docs"}], "documentation"),

        ([{"name": "performance"}], "performance"),

        ([{"name": "optimization"}], "performance"),

        ([{"name": "question"}], "question"),

        ([{"name": "questions"}], "question"),

        ([{"name": "docker"}], None),

        ([{"name": "debugging"}], None),

        (
            [
                {"name": "bug"},
                {"name": "enhancement"}
            ],
            None
        ),

        ([], None)
    ]

    for labels, expected in test_cases:

        result = classify_labels(labels)

        assert result == expected, (
            f"Failed: {labels}, "
            f"Expected: {expected}, Got: {result}"
        )

    print("All label classification tests passed!")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    test_label_classification()

    main()