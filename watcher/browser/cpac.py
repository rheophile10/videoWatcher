"""basic cpac scraper"""

import asyncio
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
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


def download_video(m3u8_url: str, output_path: Path):
    print(f"Downloading → {output_path.name}")
    print(f"   Source: {m3u8_url}\n")

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            m3u8_url,
            "-c",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            str(output_path),
        ],
        check=False,
    )

    if result.returncode == 0:
        print("Download completed successfully!")
    else:
        print("ffmpeg failed (but probably still worked partially)")


async def main():
    ids = await get_recent_episode_ids(days_back=2, max_pages=5)
    if not ids:
        print("No episodes found.")
        return

    newest_id = next(iter(ids))
    print(f"\nTesting download with newest episode: {newest_id}")

    data = get_episode_details(newest_id)
    details = data["page"]["details"]

    title = re.sub(r'[<>:"/\\|?*]', "_", details["title_en_t"])[:120]
    date = details["liveDateTime"][:10]
    m3u8 = details["videoUrl"]

    out_dir = Path(__file__).parent / "downloads"
    out_dir.mkdir(exist_ok=True)
    output_file = out_dir / f"{newest_id}.mp4"
