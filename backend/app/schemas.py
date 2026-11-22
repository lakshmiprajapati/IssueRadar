
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Literal


# -----------------------------------------
# 1. Health Check Response
# -----------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    application: str = "IssueRadar"


# -----------------------------------------
# 2. Repository Analysis Request
# -----------------------------------------

class RepositoryAnalyzeRequest(BaseModel):
    repository_url: HttpUrl = Field(
        ...,
        description="Public GitHub repository URL",
        examples=["https://github.com/fastapi/fastapi"]
    )

    @field_validator("repository_url")
    @classmethod
    def validate_github_repository(cls, value):
        if value.host != "github.com":
            raise ValueError("Only GitHub repository URLs are supported")

        path_parts = value.path.strip("/").split("/")

        if len(path_parts) != 2:
            raise ValueError(
                "URL must contain a valid owner and repository name"
            )

        return value


# -----------------------------------------
# 3. Repository Analysis Response
# -----------------------------------------

class RepositoryAnalyzeResponse(BaseModel):
    message: str
    repository_url: str
    status: Literal["queued", "processing", "completed", "failed"]


# -----------------------------------------
# 4. Issue Information Schema
# -----------------------------------------

class IssueSchema(BaseModel):
    issue_id: int
    title: str
    description: str = ""
    state: Literal["open", "closed"]
    labels: list[str] = Field(default_factory=list)
    html_url: HttpUrl


# -----------------------------------------
# 5. API Error Response
# -----------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str




# -----------------------------------------
# 6. Issue Prediction Request
# -----------------------------------------

class IssuePredictionRequest(BaseModel):

    title: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="GitHub issue title"
    )

    description: str = Field(
        default="",
        max_length=20000,
        description="GitHub issue description"
    )


# -----------------------------------------
# 7. Issue Prediction Response
# -----------------------------------------

class IssuePredictionResponse(BaseModel):

    predicted_category: str
    confidence: float
    probabilities: dict[str, float]
