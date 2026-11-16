
import json
from pathlib import Path
from collections import Counter

from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_posts.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sentiment_results_unified.json"
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def run_sentiment_analysis():

    print("Loading unified dataset...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        posts = json.load(file)

    if not posts:
        raise ValueError("No records found in the cleaned dataset.")

    print("Total records:", len(posts))

    print(f"\nLoading sentiment model: {MODEL_NAME}")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )

    texts = [
        post["cleaned_text"]
        for post in posts
    ]

    print("\nRunning sentiment predictions...")

    predictions = classifier(
        texts,
        batch_size=4,
        truncation=True,
        max_length=512
    )

    results = []

    for post, prediction in zip(posts, predictions):

        result = post.copy()

        result["sentiment_label"] = prediction["label"].lower()

        result["sentiment_score"] = round(
            float(prediction["score"]), 4
        )

        result["sentiment_model"] = MODEL_NAME

        results.append(result)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Overall sentiment distribution
    sentiment_counts = Counter(
        result["sentiment_label"]
        for result in results
    )

    # Platform-wise sentiment distribution
    platform_sentiments = {}

    for result in results:

        platform = result["platform"]
        sentiment = result["sentiment_label"]

        if platform not in platform_sentiments:
            platform_sentiments[platform] = Counter()

        platform_sentiments[platform][sentiment] += 1

    print("\nSentiment analysis completed!")

    print("Total analyzed:", len(results))

    print("\nOverall sentiment distribution:")

    for label, count in sentiment_counts.items():
        print(f"{label}: {count}")

    print("\nPlatform-wise sentiment:")

    for platform, counts in platform_sentiments.items():
        print(f"\n{platform}:")

        for label, count in counts.items():
            print(f"  {label}: {count}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    run_sentiment_analysis()
