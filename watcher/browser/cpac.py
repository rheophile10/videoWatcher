"""basic cpac scraper"""

import asyncio
import re
from datetime import datetime, timedelta
import requests
from typing import Tuple
from pathlib import Path

from playwright.async_api import async_playwright
from watcher.browser.utils import download_video


SCRAPER_NAME = "cpac"


async def get_recent_episode_ids(days_back: int = 2, max_pages: int = 5) -> set[str]:
    """Scrape CPAC for recent episode IDs within the given days_back."""
    seen_ids = set()
    cutoff_date = (datetime.now() - timedelta(days=days_back)).date()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(
            "https://www.cpac.ca/search?programId=6&sort=desc", wait_until="networkidle"
        )
        await page.wait_for_selector(".list-main__item", timeout=15000)

        page_num = 1
        while page_num <= max_pages:

            for card in await page.query_selector_all(".list-main__item"):
                link = await card.query_selector("a[href*='?id=']")
                if not link:
                    continue
                href = await link.get_attribute("href")
                m = re.search(r"id=([a-f0-9-]{36})", href)
                if not m:
                    continue
                date_match = re.search(r"--([a-z]+-\d{1,2}-\d{4})\?", href)
                date_str = (
                    date_match.group(1).replace("-", " ").title()
                    if date_match
                    else None
                )
                episode_date = (
                    datetime.strptime(date_str, "%B %d %Y").date() if date_str else None
                )
                if episode_date and episode_date < cutoff_date:
                    await browser.close()
                    return seen_ids

                eid = m.group(1)
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)

            next_btn = await page.query_selector(
                "#pagination__control_next:not([disabled])"
            )
            if not next_btn or page_num >= max_pages:
                break

            await next_btn.click()
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(1.5)
            page_num += 1

        await browser.close()

    return seen_ids


def get_episode_details(episode_id: str, source_id: int) -> dict:
    url = "https://www.cpac.ca/api/1/services/contentModel.json"
    params = {
        "url": "/site/website/episode/index.xml",
        "crafterSite": "cpactv",
        "id": episode_id,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    db_record = (
        source_id,
        data["page"]["details"]["videoUrl"],
        data["page"]["details"]["title_en_t"],
        data["page"]["details"]["description_en_t"],
        datetime.fromisoformat(data["page"]["details"]["liveDateTime"][:10]),
    )
    return db_record


def videos_since(since: datetime, source_id: int) -> list[tuple[dict, str]]:
    """Get videos since a given datetime."""
    episode_ids = asyncio.run(
        get_recent_episode_ids(days_back=(datetime.now() - since).days + 1)
    )
    videos = []
    for eid in episode_ids:
        db_record = get_episode_details(eid, source_id)
        videos.append((db_record, eid))
    return videos


def video_download(video_url: str, video_id: int) -> Tuple[Path, float]:
    """Download video from m3u8 URL."""
    filename = str(video_id) + ".mp4"
    return download_video(video_url, filename)
