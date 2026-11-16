
import json
import re
import time
import html
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://hacker-news.firebaseio.com/v0"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "hackernews_posts.json"
)

MAX_STORIES = 30
MAX_COMMENTS_PER_STORY = 2

session = requests.Session()
session.headers.update({
    "User-Agent": "SocialMediaIntelligencePlatform/1.0"
})


def fetch_item(item_id):
    """Fetch a Hacker News story or comment."""

    url = f"{BASE_URL}/item/{item_id}.json"

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        print(f"Could not fetch item {item_id}: {error}")
        return None


def clean_text(raw_text):
    """Remove HTML tags and decode HTML entities."""

    if not raw_text:
        return ""

    text = html.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def convert_timestamp(timestamp):
    """Convert Unix timestamp into ISO format."""

    if not timestamp:
        return None

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).isoformat()


def collect_hackernews_data():

    print("Starting Hacker News data collection...")

    try:
        response = session.get(
            f"{BASE_URL}/topstories.json",
            timeout=15
        )

        response.raise_for_status()
        story_ids = response.json()

    except requests.RequestException as error:
        print("Unable to retrieve top stories:", error)
        return

    story_ids = story_ids[:MAX_STORIES]

    records = []

    collected_at = datetime.now(timezone.utc).isoformat()

    for index, story_id in enumerate(story_ids, start=1):

        print(f"Processing story {index}/{len(story_ids)}")

        story = fetch_item(story_id)

        if not story or story.get("deleted") or story.get("dead"):
            continue

        story_author = story.get("by", "unknown")

        story_title = clean_text(story.get("title", ""))
        story_text = clean_text(story.get("text", ""))

        # Store story record
        if story_title or story_text:

            records.append({
                "post_id": f"hn_{story_id}",
                "platform": "hacker_news",
                "text": f"{story_title}. {story_text}".strip(),
                "created_at": convert_timestamp(story.get("time")),
                "author": story_author,
                "community": "hacker_news",
                "item_type": "story",
                "story_id": story_id,
                "parent_id": story.get("parent"),
                "story_author": story_author,
                "score": story.get("score", 0),
                "comment_count": story.get("descendants", 0),
                "url": story.get("url", ""),
                "collected_at": collected_at
            })

        # Fetch comments belonging to this story
        comment_ids = story.get("kids", [])[
            :MAX_COMMENTS_PER_STORY
        ]

        for comment_id in comment_ids:

            comment = fetch_item(comment_id)

            if (
                not comment
                or comment.get("deleted")
                or comment.get("dead")
            ):
                continue

            comment_text = clean_text(comment.get("text", ""))

            if not comment_text:
                continue

            records.append({
                "post_id": f"hn_{comment_id}",
                "platform": "hacker_news",
                "text": comment_text,
                "created_at": convert_timestamp(comment.get("time")),
                "author": comment.get("by", "unknown"),
                "community": "hacker_news",
                "item_type": "comment",
                "story_id": story_id,
                "parent_id": comment.get("parent"),
                "story_author": story_author,
                "score": comment.get("score", 0),
                "collected_at": collected_at
            })

            time.sleep(0.1)

        time.sleep(0.1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nHacker News collection completed!")
    print("Total records collected:", len(records))
    print("Stories requested:", len(story_ids))
    print("Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    collect_hackernews_data()
