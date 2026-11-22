
from app.ml.predict import predict_issue


result = predict_issue(
    title="Application crashes when loading a CSV file",
    body="When I try to load a large CSV file, the application "
         "crashes with an unexpected error."
)

print("\n========== ISSUERADAR PREDICTION ==========")

print("Predicted category:", result["predicted_category"])

print("Confidence:", result["confidence"])

print("\nCategory probabilities:")

for category, probability in result["probabilities"].items():
    print(f"{category}: {probability}")
