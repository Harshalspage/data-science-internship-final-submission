
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv(".env")

api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    raise ValueError("YouTube API key is missing.")

try:
    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    response = youtube.search().list(
        part="snippet",
        q="artificial intelligence",
        type="video",
        maxResults=3
    ).execute()

    print("YouTube API connection successful!")
    print("Videos returned:", len(response.get("items", [])))

    for item in response.get("items", []):
        print("Title:", item["snippet"]["title"])

except HttpError as error:
    print("YouTube API request failed.")
    print("HTTP status:", error.resp.status)
    print("Check your credential type, API restrictions and project settings.")
