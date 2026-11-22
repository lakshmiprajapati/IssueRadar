
import os
import re
import httpx

from collections import Counter

from fastapi import FastAPI, HTTPException

from app.schemas import (
    RepositoryAnalyzeRequest,
    IssuePredictionRequest,
)

from app.ml.predict import predict_issue

from app.ml.duplicate_detector import search_similar_issues_batch


# --------------------------------------------------
# INITIALIZE APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="IssueRadar API",
    description="AI-powered GitHub Issue Intelligence Platform",
    version="1.0.0"
)


GITHUB_API = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


# --------------------------------------------------
# HELPER: EXTRACT OWNER AND REPOSITORY
# --------------------------------------------------

def extract_repo_details(repository_url: str):

    pattern = r"github\.com/([^/]+)/([^/]+)"

    match = re.search(
        pattern,
        str(repository_url).rstrip("/")
    )

    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub repository URL"
        )

    owner = match.group(1)
    repo = match.group(2).removesuffix(".git")

    return owner, repo


# --------------------------------------------------
# HELPER: GITHUB API REQUEST
# --------------------------------------------------

async def github_request(client, endpoint):

    response = await client.get(
        f"{GITHUB_API}{endpoint}",
        headers=HEADERS
    )

    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="GitHub repository not found"
        )

    if response.status_code == 403:
        raise HTTPException(
            status_code=429,
            detail="GitHub API rate limit exceeded or access denied"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error: {response.text}"
        )

    return response.json()


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
async def home():

    return {
        "message": "Welcome to IssueRadar",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():

    try:
        from app.ml.predict import load_model

        load_model()
        model_status = "loaded"

    except Exception:
        model_status = "unavailable"

    return {
        "status": "healthy",
        "application": "IssueRadar",
        "model_status": model_status
    }


# --------------------------------------------------
# INDIVIDUAL ISSUE PREDICTION
# --------------------------------------------------

@app.post("/api/v1/predict")
async def predict_github_issue(request: IssuePredictionRequest):

    try:

        result = predict_issue(
            title=request.title,
            body=request.description
        )

        return {
            "message": "Issue prediction completed successfully",
            **result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail="Trained ML model is unavailable"
        )


# --------------------------------------------------
# REPOSITORY ANALYSIS ENDPOINT
# --------------------------------------------------

@app.post("/api/v1/analyze")
async def analyze_repository(request: RepositoryAnalyzeRequest):

    owner, repo = extract_repo_details(
        str(request.repository_url)
    )

    async with httpx.AsyncClient(timeout=30.0) as client:

        # STEP 1: FETCH REPOSITORY INFORMATION

        repository = await github_request(
            client,
            f"/repos/{owner}/{repo}"
        )

        # STEP 2: FETCH OPEN ISSUES

        issues_data = await github_request(
            client,
            f"/repos/{owner}/{repo}/issues?state=open&per_page=100&page=1"
        )

        # STEP 3: REMOVE PULL REQUESTS

        issues = []

        for issue in issues_data:

            if "pull_request" in issue:
                continue

            title = issue["title"] or ""
            body = issue["body"] or ""

            # STEP 4: PREDICT ISSUE CATEGORY

            try:

                prediction = predict_issue(
                    title=title,
                    body=body
                )

                prediction_status = "success"

            except ValueError as error:

                prediction = None
                prediction_status = str(error)

            except FileNotFoundError:

                raise HTTPException(
                    status_code=500,
                    detail="Trained ML model is unavailable"
                )

            # STEP 5: PREPARE ISSUE RESPONSE

            issues.append({

                "id": issue["id"],
                "number": issue["number"],
                "title": title,
                "body": body,
                "state": issue["state"],
                "html_url": issue["html_url"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "comments": issue["comments"],

                "labels": [
                    label["name"]
                    for label in issue["labels"]
                ],

                "user": issue["user"]["login"],

                "prediction": prediction,
                "prediction_status": prediction_status

            })


    # --------------------------------------------------
    # STEP 5.5: SEMANTIC DUPLICATE DETECTION
    # --------------------------------------------------

    queries = [
        f"{issue['title']}\n{issue['body']}"
        for issue in issues
    ]

    issue_urls = [
        issue["html_url"]
        for issue in issues
    ]

    try:

        duplicate_results = search_similar_issues_batch(
            queries=queries,
            exclude_urls=issue_urls,
            top_k=5
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail="Duplicate detection index is unavailable"
        )

    for issue, matches in zip(issues, duplicate_results):

        issue["duplicate_issues"] = matches
        issue["duplicate_count"] = len(matches)


    # --------------------------------------------------
    # STEP 6: GENERATE REPOSITORY ANALYTICS
    # --------------------------------------------------

    predicted_categories = [
        issue["prediction"]["predicted_category"]
        for issue in issues
        if issue["prediction"] is not None
    ]

    # Count issues in each predicted category

    category_distribution = dict(
        Counter(predicted_categories)
    )

    # Calculate average prediction confidence

    confidences = [
        issue["prediction"]["confidence"]
        for issue in issues
        if issue["prediction"] is not None
    ]

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    successful_predictions = len(predicted_categories)


    # --------------------------------------------------
    # STEP 7: PREPARE REPOSITORY INFORMATION
    # --------------------------------------------------

    repository_info = {

        "name": repository["name"],
        "full_name": repository["full_name"],
        "description": repository["description"],
        "stars": repository["stargazers_count"],
        "forks": repository["forks_count"],
        "open_issues_count": repository["open_issues_count"],
        "language": repository["language"],
        "html_url": repository["html_url"]

    }


    # --------------------------------------------------
    # STEP 8: FINAL RESPONSE
    # --------------------------------------------------

    return {

        "message": "Repository analyzed successfully",

        "repository": repository_info,

        "issues_fetched": len(issues),

        "issues_classified": successful_predictions,

        "analytics": {

            "total_issues": len(issues),

            "classified_issues": successful_predictions,

            "category_distribution": category_distribution,

            "average_confidence": round(
                average_confidence, 4
            )

        },

        "issues": issues

    }
