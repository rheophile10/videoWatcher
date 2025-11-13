from datetime import datetime
from typing import Generator, Dict, Any
import requests, time

BASE = "https://cpac.ca/api/1/site/search/search.json"
session = requests.Session()
session.headers.update(
    {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
)


def videos_since(since: datetime) -> Generator[Dict[str, Any], None, None]:
    filters = [
        "state_s:LIVE_PUBLISHED",
        f'event_date_time_dt:[{since.strftime("%Y-%m-%dT%H:%M:%SZ")} TO *]',
    ]
    start = 0
    rows = 100
    total = None

    while True:
        payload = {
            "query": 'content-type:"/page/episode"',
            "sort": "event_date_time_dt desc",
            "start": start,
            "rows": rows,
            "fl": "*,score",
            "fq": filters,
        }
        data = session.post(BASE, json=payload, timeout=30).json()
        if total is None:
            total = data["response"]["numFound"]
            print(f"CPAC: {total} videos found")

        for doc in data["response"]["docs"]:
            vid_url = None
            if "videos_o" in doc:
                for item in doc["videos_o"].get("item", []):
                    url = item.get("component", {}).get("url_s")
                    if url and url.endswith(".m3u8"):
                        vid_url = url
                        break

            yield {
                "episode_id": doc.get("episode_id_s"),
                "title_en": doc.get("title_en_t", ""),
                "title_fr": doc.get("title_fr_t", ""),
                "date": doc.get("event_date_time_dt", "")[:10],
                "video_url": vid_url,
                "page_en": f"https://cpac.ca/episode?id={doc.get('episode_id_s')}",
            }

        start += rows
        if start >= total:
            break
        time.sleep(0.15)
