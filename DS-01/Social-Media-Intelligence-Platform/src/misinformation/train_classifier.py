
import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load training and validation datasets
train_df = pd.read_csv(
    os.path.join(DATA_DIR, "misinformation_train.csv")
)

valid_df = pd.read_csv(
    os.path.join(DATA_DIR, "misinformation_valid.csv")
)

X_train = train_df["statement"].fillna("")
y_train = train_df["binary_label"]

X_valid = valid_df["statement"].fillna("")
y_valid = valid_df["binary_label"]

print("Training records:", len(X_train))
print("Validation records:", len(X_valid))

# Build TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train model
print("\nTraining misinformation classifier...")

model.fit(X_train, y_train)

# Validate model
predictions = model.predict(X_valid)
probabilities = model.predict_proba(X_valid)[:, 1]

print("\nVALIDATION RESULTS")
print("------------------------------")

print(
    "Accuracy:",
    round(accuracy_score(y_valid, predictions), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_valid, probabilities), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        predictions,
        target_names=[
            "Potentially misleading",
            "Likely reliable"
        ]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_valid, predictions))

# Save model
model_path = os.path.join(
    MODEL_DIR,
    "misinformation_classifier.joblib"
)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(model_path)