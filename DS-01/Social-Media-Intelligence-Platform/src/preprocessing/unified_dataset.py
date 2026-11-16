
import json
from pathlib import Path
from collections import Counter

YOUTUBE_FILE = Path("data/raw/youtube_posts.json")
HACKERNEWS_FILE = Path("data/raw/hackernews_posts.json")
REDDIT_SAMPLE_FILE = Path("data/raw/ingested_posts.json")

OUTPUT_FILE = Path("data/processed/unified_posts.json")


def load_json(file_path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_record(record):
    """Convert records into a common structure."""

    return {
        "post_id": record.get("post_id"),
        "platform": record.get("platform", "unknown"),
        "text": record.get("text", ""),
        "created_at": record.get("created_at"),
        "author": record.get("author", "unknown"),
        "community": record.get("community", "unknown"),
        "item_type": record.get("item_type", "post"),
        "engagement": {
            "views": record.get("statistics", {}).get("view_count", 0),
            "likes": record.get("statistics", {}).get("like_count", 0),
            "comments": record.get(
                "statistics", {}
            ).get(
                "comment_count",
                record.get("comment_count", 0)
            ),
            "score": record.get("score", 0)
        },
        "source_url": record.get("url", ""),
        "data_origin": "sample" if record.get("platform") == "reddit"
                       else "api"
    }


def build_dataset():
    print("Loading platform datasets...")

    youtube_data = load_json(YOUTUBE_FILE)
    hackernews_data = load_json(HACKERNEWS_FILE)
    reddit_data = load_json(REDDIT_SAMPLE_FILE)

    # Keep only synthetic Reddit records from the sample file.
    reddit_samples = [
        record for record in reddit_data
        if record.get("platform") == "reddit"
    ]

    all_records = (
        [normalize_record(r) for r in youtube_data]
        + [normalize_record(r) for r in hackernews_data]
        + [normalize_record(r) for r in reddit_samples]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_records, file, indent=4, ensure_ascii=False)

    print("\nUnified dataset created successfully!")
    print("Total records:", len(all_records))
    print("Platform distribution:")

    for platform, count in Counter(
        r["platform"] for r in all_records
    ).items():
        print(f"{platform}: {count}")

    print("\nData origin distribution:")

    for origin, count in Counter(
        r["data_origin"] for r in all_records
    ).items():
        print(f"{origin}: {count}")

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()
