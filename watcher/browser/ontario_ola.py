from datetime import datetime
from typing import Generator, Dict
import requests


def videos_since(since: datetime) -> Generator[Dict, None, None]:
    url = "https://www.ola.org/en/legislative-business/video/search"
    params = {"dateFrom": since.strftime("%Y-%m-%d")}
    r = requests.get(url, params=params)
    data = r.json()

    for item in data.get("results", []):
        yield {
            "title_en": item["title"],
            "date": item["date"][:10],
            "video_url": item["videoUrl"],  # direct MP4
            "page_en": f"https://www.ola.org/en/legislative-business/video/{item['id']}",
        }
