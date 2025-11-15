"""Coordinate source scraping activities."""

import sqlite3
from datetime import datetime, timedelta
from typing import Callable, Dict, List
from importlib import import_module
from watcher.db import p_query, s_proc
from watcher.db.log import insert_log
from watcher.db.videos import (
    insert_videos,
    get_videos_to_download,
    update_video_downloaded,
)


def _toggle_source_activation_status(
    conn: sqlite3.Connection, name: str, activation_status: int = 0
) -> None:
    """Deactivate a source by its scraper name."""
    if activation_status not in (0, 1):
        raise ValueError("Activation status must be 0 (deactivate) or 1 (activate)")
    s_proc(
        conn, "sources", "update_active_by_scraper_name", [(activation_status, name)]
    )


def _import_scraper(conn: sqlite3.Connection, name: str, active: int) -> Callable:
    """Dynamically import a scraper module by name. If import fails, deactivate the source."""
    try:
        mod = import_module(f".{name}", package=__package__)
    except ImportError:
        if active == 1:
            _toggle_source_activation_status(
                conn,
                name=name,
                activation_status=0,
            )
        return None, None
    if active == 0:
        _toggle_source_activation_status(conn, name=name, activation_status=1)
    return mod.videos_since, mod.video_download


def get_scrapers(conn: sqlite3.Connection) -> List[Dict]:
    """Retrieve scrapers listed in database and attempt to import them"""
    rows = p_query(conn, "sources", "all", ())
    scrapers = {}
    for row in rows:
        scraper_name = row["scraper_name"]
        videos_since, video_download = _import_scraper(
            conn, scraper_name, row["active"]
        )
        if videos_since is not None and video_download is not None:
            scrapers[scraper_name] = {
                "source_id": row["id"],
                "videos_since_func": videos_since,
                "download_videos_func": video_download,
            }
    return scrapers


def get_video_urls(conn: sqlite3.Connection, scrapers: Dict, since: datetime) -> None:
    """Extract video URLs from the scraper's videos_since function output."""
    try:
        for scraper_name, scraper_info in scrapers.items():
            source_id = scraper_info["source_id"]
            insert_log(
                conn,
                "source_check_started",
                source_id=source_id,
                source_name=scraper_name,
            )
            videos_since_func = scraper_info["videos_since_func"]
            videos = videos_since_func(since, source_id)
            num_videos = len(videos)
            video_db_records = [video_db_record for video_db_record, _ in videos]
            insert_videos(conn, video_db_records)
            insert_log(
                conn,
                "source_check_successful",
                source_id=source_id,
                num_videos=num_videos,
                source_name=scraper_name,
            )
    except Exception as e:
        insert_log(
            conn,
            "source_check_failed",
            source_id=source_id,
            source_name=scraper_name,
            error_msg=str(e),
        )


def download_videos(conn: sqlite3.Connection, scrapers: Dict) -> None:
    """Download videos using the scraper's download_videos function."""
    videos_to_download = get_videos_to_download(conn)
    for row in videos_to_download:
        scraper_name = row["scraper_name"]
        source_id = row["source_id"]
        video_id = row["video_id"]
        video_url = row["video_url"]
        scraper_info = scrapers.get(scraper_name)
        download_videos_func = scraper_info["download_videos_func"]
        try:
            output_path, duration = download_videos_func(video_url, video_id)
            update_video_downloaded(conn, video_id, str(output_path), duration)
            insert_log(
                conn,
                "video_downloaded",
                source_id=source_id,
                video_id=video_id,
                video_url=video_url,
            )
        except Exception as e:
            insert_log(
                conn,
                "video_download_failed",
                source_id=source_id,
                video_id=video_id,
                video_url=video_url,
                error_msg=str(e),
            )


def get_videos_in_last_n_days(conn: sqlite3.Connection, n: int = 1) -> None:
    """Main function to get videos from all scrapers in the last n days."""
    since = datetime.now() - timedelta(days=n)
    scrapers = get_scrapers(conn)
    get_video_urls(conn, scrapers, since)
    download_videos(conn, scrapers)


def get_source_by_scraper_name(
    conn: sqlite3.Connection, scraper_name: str
) -> sqlite3.Row:
    """Get source details by scraper name."""
    rows = p_query(
        conn,
        "sources",
        "get_by_scraper_name",
        (scraper_name,),
    )
    if rows:
        return rows[0]
    return None


__all__ = [
    "get_videos_in_last_n_days",
    "get_scrapers",
    "get_video_urls",
    "download_videos",
    "get_source_by_scraper_name",
]
