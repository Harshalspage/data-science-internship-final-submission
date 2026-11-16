
import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_posts.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "misinformation_classifier.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "misinformation_social_posts.json"
)

print("Loading social media posts...")

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    posts = json.load(file)

print("Total posts loaded:", len(posts))

print("\nLoading misinformation model...")

model = joblib.load(MODEL_PATH)

results = []

for post in posts:

    text = post.get("cleaned_text") or post.get("text") or ""

    if not text.strip():
        prediction = None
        confidence = None
        result = "Not analyzed"
    else:
        prediction = int(model.predict([text])[0])

        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))

        result = (
            "Potentially misleading"
            if prediction == 0
            else "Likely reliable"
        )

    analyzed_post = post.copy()

    analyzed_post["misinformation_prediction"] = result
    analyzed_post["misinformation_class"] = prediction
    analyzed_post["misinformation_confidence"] = (
        round(confidence, 4)
        if confidence is not None
        else None
    )

    analyzed_post["fact_check_status"] = "Unverified"

    results.append(analyzed_post)

# Save results
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

# Summary
df = pd.DataFrame(results)

print("\nMISINFORMATION ANALYSIS SUMMARY")
print("=" * 45)

print("\nPlatform distribution:")
print(df["platform"].value_counts())

print("\nPrediction distribution:")
print(df["misinformation_prediction"].value_counts())

print("\nAverage model confidence by platform:")
print(
    df.groupby("platform")[
        "misinformation_confidence"
    ].mean().round(4)
)

print("\nAnalyzed posts:", len(results))
print("Saved results:", OUTPUT_PATH)

print("\nMisinformation analysis completed!")
