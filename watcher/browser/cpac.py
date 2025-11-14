"""basic cpac scraper"""

import asyncio
import re
from datetime import datetime, timedelta
import requests

from playwright.async_api import async_playwright


async def get_recent_episode_ids(days_back: int = 2, max_pages: int = 5) -> set[str]:
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
            print(f"Scraping page {page_num}...")

            for card in await page.query_selector_all(".list-main__item"):
                link = await card.query_selector("a[href*='?id=']")
                if not link:
                    continue
                href = await link.get_attribute("href")
                m = re.search(r"id=([a-f0-9-]{36})", href)
                if not m:
                    continue

                eid = m.group(1)
                if eid in seen_ids:
                    continue

                # Optional: stop early if too old
                time_el = await card.query_selector("time")
                if time_el:
                    dt = await time_el.get_attribute("datetime")
                    if dt and datetime.fromisoformat(dt[:10]).date() < cutoff_date:
                        print(f"Date cutoff reached, stopping.")
                        await browser.close()
                        return seen_ids

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

    print(f"Found {len(seen_ids)} unique episodes")
    return seen_ids


def get_episode_details(episode_id: str) -> dict:
    url = "https://www.cpac.ca/api/1/services/contentModel.json"
    params = {
        "url": "/site/website/episode/index.xml",
        "crafterSite": "cpactv",
        "id": episode_id,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()
