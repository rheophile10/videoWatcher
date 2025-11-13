from datetime import datetime
from typing import Generator, Dict
import requests, time


def videos_since(since: datetime) -> Generator[Dict, None, None]:
    session = requests.Session()
    page = 1

    while True:
        r = session.get(
            "https://senparlvu.parl.gc.ca/api/v1/videos",
            params={
                "page": page,
                "pageSize": 100,
                "from": since.strftime("%Y-%m-%d"),
            },
            timeout=30,
        )
        data = r.json()

        for v in data.get("items", []):
            yield {
                "video_id": v["id"],
                "title_en": v["title"],
                "date": v["date"][:10],
                "video_url": v["hlsUrl"],
                "page_en": f"https://senparlvu.parl.gc.ca/Video/{v['id']}",
            }

        if page >= data.get("totalPages", 1):
            break
        page += 1
        time.sleep(0.1)
