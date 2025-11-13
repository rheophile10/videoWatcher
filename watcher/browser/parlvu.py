from datetime import datetime
from typing import Generator, Dict
import requests, time

BASE = "https://parlvu.parl.gc.ca/Harmony/en-CA/PowerBrowser/PowerBrowserV2"


def videos_since(since: datetime) -> Generator[Dict, None, None]:
    session = requests.Session()
    start = 0
    page_size = 100

    while True:
        r = session.get(
            f"{BASE}/GetEventGridData",
            params={
                "start": start,
                "length": page_size,
                "dateFrom": since.strftime("%Y-%m-%d"),
                "eventTypeFilter": "2",  # Committee meetings
            },
            timeout=30,
        )
        data = r.json()

        for rec in data["data"]:
            event_id = rec["EventID"]
            yield {
                "event_id": event_id,
                "title_en": rec["TitleEn"],
                "title_fr": rec["TitleFr"],
                "date": rec["Date"][:10],
                "video_url": f"https://parlvu.parl.gc.ca/Harmony/en-CA/PowerBrowser/Video/{event_id}/High",
                "page_en": f"https://parlvu.parl.gc.ca/Harmony/en-CA/PowerBrowser/PowerBrowserV2/{event_id}",
            }

        if len(data["data"]) < page_size:
            break
        start += page_size
        time.sleep(0.1)
