"""Coordinate source scraping activities.

could add more logging of scraper errors here later
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Callable, Dict, List
from importlib import import_module
from ..db import p_query, s_proc, log


def _get_video_id_by_url(conn: sqlite3.Connection, video_url: str) -> sqlite3.Row:
    """Retrieve video ID by its URL."""
    p_query(conn, "videos", "get_id_by_url", (video_url,))


def _toggle_source_activation_status(
    conn: sqlite3.Connection, name: str, activation_status: int = 0
) -> None:
    """Deactivate a source by its scraper name."""
    if activation_status not in (0, 1):
        raise ValueError("Activation status must be 0 (deactivate) or 1 (activate)")
    s_proc(
        conn, "sources", "update_active_by_scraper_name", [(activation_status, name)]
    )


def _register_scraper_run_attempt(conn: sqlite3.Connection, name: str) -> datetime:
    """update last run time for a given scraper."""
    s_proc(conn, "sources", "update_last_run_by_scraper_name", (name,))


def _register_scraper_successful_run(conn: sqlite3.Connection, name: str) -> datetime:
    """update last successful run time for a given scraper."""
    s_proc(conn, "sources", "update_last_success_by_scraper_name", (name,))


def _register_video_download_attempt(conn: sqlite3.Connection, video: Dict) -> int:
    """Register a video download attempt in the database."""
    s_proc(
        conn,
        "videos",
        "update_download_attempt",
        [
            (
                video["source_id"],
                video["video_url"],
                video["title"],
                video["published_at"],
                video.get("video_file_path"),
            )
        ],
    )
    return _get_video_id_by_url(conn, video["video_url"])


def _register_succesful_download(conn: sqlite3.Connection, video: Dict) -> None:
    """Register a successful video download in the database."""
    s_proc(
        conn,
        "videos",
        "mark_downloaded",
        [
            (
                video.get("video_file_path"),
                video.get("duration_seconds"),
                video["video_url"],
            )
        ],
    )


def _import_scraper(conn: sqlite3.Connection, name: str, active: int) -> Callable:
    """Dynamically import a scraper module by name. If import fails, deactivate the source."""
    try:
        mod = import_module(f".{name}_scraper", package=__package__)
    except ImportError:
        if active == 1:
            _toggle_source_activation_status(
                conn,
                name=name,
                activation_status=0,
            )
        return None, None
    if active == 0:
        _toggle_source_activation_status(conn, activation_status=1, name=name)
    return mod.videos_since, mod.download_videos


def get_scrapers(conn: sqlite3.Connection) -> List[Dict]:
    """Retrieve scrapers listed in database and attempt to import them"""
    rows = p_query(conn, "sources", "all", ())
    scrapers = {}
    for row in rows:
        scraper_name = row.get("scraper_name")
        videos_since, video_download = _import_scraper(
            conn, scraper_name, row["active"]
        )
        if videos_since is not None:
            scrapers[scraper_name] = {
                "scraper_id": row["id"],
                "videos_since_func": videos_since,
                "download_videos_func": video_download,
            }
    return scrapers


def get_videos_in_last_n_days(conn: sqlite3.Connection, n: int = 1) -> None:
    """Main function to get videos from all scrapers in the last n days."""
    since = datetime.now() - timedelta(days=n)
    scrapers = get_scrapers(conn)
    for scraper_id, scraper_name, funcs in scrapers.items():
        _register_scraper_run_attempt(conn, scraper_name)
        try:
            videos = funcs["videos_since_func"](scraper_id, since)
            for video_db_record, download_args in videos:
                video_id = _register_video_download_attempt(conn, video_db_record)
                try:
                    updated_video_db_record = funcs["download_videos_func"](
                        video_db_record, **download_args
                    )
                    _register_succesful_download(conn, updated_video_db_record)
                except Exception as e:
                    log(
                        conn,
                        log_type="error",
                        log_message=e,
                        source_id=video_db_record.get("source_id"),
                        video_id=video_id,
                    )
            _register_scraper_successful_run(conn, scraper_name)
        except Exception as e:
            log(
                conn,
                log_type="error",
                log_message=e,
                source_id=scraper_id,
                video_id=None,
            )


__all__ = ["get_videos_in_last_n_days", "get_scrapers"]
