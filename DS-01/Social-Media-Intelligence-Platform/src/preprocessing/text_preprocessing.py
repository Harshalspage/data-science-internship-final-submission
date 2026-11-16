
import json
import re
import html
from pathlib import Path

INPUT_FILE = Path("data/processed/unified_posts.json")
OUTPUT_FILE = Path("data/processed/cleaned_posts.json")


def clean_text(text):
    """Clean text while preserving its original meaning."""

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataset():
    print("Starting text preprocessing...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    cleaned_records = []
    seen_text = set()

    removed_empty = 0
    removed_duplicates = 0

    for record in records:

        original_text = record.get("text", "")
        cleaned = clean_text(original_text)

        if not cleaned:
            removed_empty += 1
            continue

        # Detect duplicate text within the same platform
        duplicate_key = (record.get("platform"), cleaned)

        if duplicate_key in seen_text:
            removed_duplicates += 1
            continue

        seen_text.add(duplicate_key)

        cleaned_record = record.copy()

        cleaned_record["original_text"] = original_text
        cleaned_record["cleaned_text"] = cleaned

        cleaned_records.append(cleaned_record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            cleaned_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nPreprocessing completed!")
    print("Input records:", len(records))
    print("Output records:", len(cleaned_records))
    print("Empty records removed:", removed_empty)
    print("Duplicate records removed:", removed_duplicates)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_dataset()
