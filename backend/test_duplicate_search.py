from app.ml.duplicate_detector import search_similar_issues

query = """
My FastAPI application fails to start when the OpenTelemetry
exporter endpoint is configured but the required package
is not installed.
"""

results = search_similar_issues(query, top_k=5)

print("\n========== TOP 5 SIMILAR ISSUES ==========")

for i, issue in enumerate(results, start=1):
    print(f"\n{i}. {issue['title']}")
    print("Repository:", issue["repository"])
    print("Category:", issue["category"])
    print("Similarity:", issue["similarity_score"])
    print("URL:", issue["issue_url"])