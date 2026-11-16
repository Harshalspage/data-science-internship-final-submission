
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "youtube_posts.json"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")


def create_youtube_client():
    """Initialize YouTube API client."""

    if not API_KEY:
        raise ValueError("YouTube API key not found in .env")

    return build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )


def search_youtube_videos(youtube, keyword, max_results=10):
    """Search YouTube videos using a keyword."""

    response = youtube.search().list(
        part="snippet",
        q=keyword,
        type="video",
        maxResults=max_results,
        order="relevance"
    ).execute()

    videos = []

    for item in response.get("items", []):

        snippet = item["snippet"]

        videos.append({
            "video_id": item["id"]["videoId"],
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_id": snippet.get("channelId"),
            "channel_title": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt")
        })

    return videos


def get_video_statistics(youtube, video_ids):
    """Retrieve statistics for collected videos."""

    statistics = {}

    for start in range(0, len(video_ids), 50):

        batch = video_ids[start:start + 50]

        response = youtube.videos().list(
            part="statistics",
            id=",".join(batch)
        ).execute()

        for item in response.get("items", []):

            stats = item.get("statistics", {})

            statistics[item["id"]] = {
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0))
            }

    return statistics


def run_youtube_ingestion():

    print("Starting YouTube data collection...")

    youtube = create_youtube_client()

    keywords = [
        "artificial intelligence",
        "climate change",
        "education technology"
    ]

    all_videos = []

    for keyword in keywords:

        print(f"Searching: {keyword}")

        videos = search_youtube_videos(
            youtube,
            keyword,
            max_results=10
        )

        for video in videos:
            video["search_keyword"] = keyword

        all_videos.extend(videos)

    # Remove duplicate videos
    unique_videos = {
        video["video_id"]: video
        for video in all_videos
    }

    videos = list(unique_videos.values())

    # Retrieve video statistics
    video_ids = [video["video_id"] for video in videos]

    statistics = get_video_statistics(
        youtube,
        video_ids
    )

    # Combine video metadata and statistics
    records = []

    for video in videos:

        video_id = video["video_id"]

        record = {
            "post_id": f"youtube_{video_id}",
            "platform": "youtube",
            "text": (
                f"{video.get('title', '')} "
                f"{video.get('description', '')}"
            ).strip(),
            "created_at": video.get("published_at"),
            "author": video.get("channel_title"),
            "community": video.get("channel_id"),
            "video_id": video_id,
            "search_keyword": video.get("search_keyword"),
            "statistics": statistics.get(video_id, {}),
            "collected_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        records.append(record)

    # Save raw data
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nYouTube ingestion completed.")
    print(f"Videos collected: {len(records)}")
    print(f"Output saved: {OUTPUT_FILE}")


if __name__ == "__main__":

    try:
        run_youtube_ingestion()

    except HttpError as error:
        print("YouTube API error:", error)

    except Exception as error:
        print("Ingestion failed:", error)
