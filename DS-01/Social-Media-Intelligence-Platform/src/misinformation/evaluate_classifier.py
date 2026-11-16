
import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "misinformation_classifier.joblib"
)

TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "misinformation_test.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR, "reports", "evaluation"
)

os.makedirs(REPORT_DIR, exist_ok=True)

# Load saved model and test dataset
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

test_df = pd.read_csv(TEST_PATH)

X_test = test_df["statement"].fillna("")
y_test = test_df["binary_label"]

# Generate predictions
print("Evaluating on unseen test data...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("\nFINAL TEST RESULTS")
print("=" * 45)

print("Test records:", len(test_df))
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Potentially misleading",
        "Likely reliable"
    ],
    output_dict=True
)

print(classification_report(
    y_test,
    predictions,
    target_names=[
        "Potentially misleading",
        "Likely reliable"
    ]
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Save individual predictions
results = test_df[["id", "label", "statement"]].copy()

results["actual_label"] = y_test
results["predicted_label"] = predictions
results["reliable_probability"] = probabilities

results["prediction"] = results[
    "predicted_label"
].map({
    0: "Potentially misleading",
    1: "Likely reliable"
})

output_path = os.path.join(
    REPORT_DIR,
    "misinformation_test_predictions.csv"
)

results.to_csv(output_path, index=False)

print("\nTest predictions saved:")
print(output_path)

print("\nEvaluation completed!")
