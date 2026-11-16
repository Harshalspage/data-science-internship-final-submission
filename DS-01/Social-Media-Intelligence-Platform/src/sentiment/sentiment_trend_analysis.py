
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "sentiment_results_unified.json"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def analyze_sentiment_trends():

    print("Loading sentiment results...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("No sentiment records found.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall = Counter()
    platform_data = defaultdict(Counter)
    daily_data = defaultdict(Counter)

    for record in records:

        sentiment = record["sentiment_label"]
        platform = record["platform"]

        overall[sentiment] += 1
        platform_data[platform][sentiment] += 1

        timestamp = record.get("created_at")

        if timestamp:
            try:
                date = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                ).date().isoformat()

                daily_data[date][sentiment] += 1

            except ValueError:
                print("Invalid timestamp:", timestamp)

    # 1. Overall sentiment report
    overall_file = REPORT_DIR / "overall_sentiment.csv"

    with overall_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "sentiment",
            "record_count",
            "percentage"
        ])

        total = len(records)

        for sentiment, count in sorted(overall.items()):
            writer.writerow([
                sentiment,
                count,
                round(count / total * 100, 2)
            ])

    # 2. Platform-wise sentiment report
    platform_file = REPORT_DIR / "platform_sentiment.csv"

    with platform_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "platform",
            "sentiment",
            "record_count",
            "percentage"
        ])

        for platform, counts in sorted(platform_data.items()):

            platform_total = sum(counts.values())

            for sentiment, count in sorted(counts.items()):
                writer.writerow([
                    platform,
                    sentiment,
                    count,
                    round(count / platform_total * 100, 2)
                ])

    # 3. Daily sentiment trend report
    daily_file = REPORT_DIR / "daily_sentiment_trends.csv"

    with daily_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "date",
            "positive",
            "neutral",
            "negative",
            "total"
        ])

        for date, counts in sorted(daily_data.items()):

            positive = counts["positive"]
            neutral = counts["neutral"]
            negative = counts["negative"]

            writer.writerow([
                date,
                positive,
                neutral,
                negative,
                positive + neutral + negative
            ])

    # Console summary
    print("\nSentiment trend analysis completed!")

    print("\nOverall sentiment:")
    for sentiment, count in overall.items():
        print(f"{sentiment}: {count}")

    print("\nPlatform-wise sentiment:")
    for platform, counts in platform_data.items():
        print(platform, dict(counts))

    print("\nUnique publication dates:", len(daily_data))

    print("\nReports generated:")
    print(overall_file)
    print(platform_file)
    print(daily_file)


if __name__ == "__main__":
    analyze_sentiment_trends()
