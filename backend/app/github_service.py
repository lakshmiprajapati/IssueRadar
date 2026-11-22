
import httpx


class GitHubService:

    BASE_URL = "https://api.github.com"

    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def parse_repository_url(self, repository_url: str):
        parts = repository_url.rstrip("/").split("/")

        if len(parts) < 5:
            raise ValueError("Invalid GitHub repository URL")

        owner = parts[-2]
        repo = parts[-1]

        return owner, repo

    async def get_repository(self, owner: str, repo: str):

        url = f"{self.BASE_URL}/repos/{owner}/{repo}"

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                headers=self.headers
            )

        if response.status_code == 404:
            raise ValueError("GitHub repository not found")

        if response.status_code == 403:
            raise ValueError("GitHub API rate limit exceeded or access denied")

        response.raise_for_status()

        return response.json()

    async def get_open_issues(self, owner: str, repo: str):

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"

        params = {
            "state": "open",
            "per_page": 30,
            "sort": "created",
            "direction": "desc"
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                headers=self.headers,
                params=params
            )

        if response.status_code == 404:
            raise ValueError("GitHub repository not found")

        if response.status_code == 403:
            raise ValueError("GitHub API rate limit exceeded or access denied")

        response.raise_for_status()

        # Pull requests are also returned by GitHub's issues endpoint.
        # Exclude them because IssueRadar analyzes issues only.
        return [
            issue for issue in response.json()
            if "pull_request" not in issue
        ]
