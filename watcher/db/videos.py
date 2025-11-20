"""
Video-related database operations.
"""

from typing import List, Tuple, Any
import sqlite3
from datetime import datetime, time

from watcher.db import s_proc, p_query, rows_to_csv


def insert_videos(conn: sqlite3.Connection, videos: List[Tuple[Any, ...]]) -> None:
    """Bulk insert videos into the database."""
    s_proc(conn, "videos", "upsert_video", videos)


def get_videos_to_download(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Get videos that need downloading."""
    return p_query(conn, "videos", "get_videos_to_download", ())


def get_videos_fetched_today(
    conn,
    since: datetime = datetime.combine(datetime.today(), time.min),
    keywords_fts_query: str = "gun OR firearm OR rifle",
) -> List[sqlite3.Row]:
    """
    Returns all videos seen today with rich metadata:
    - downloaded status
    - has transcript?
    - has embeddings?
    - how many keyword hits?
    - analyzed/briefed flags (hardcoded 0 for now)
    """
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")

    rows = p_query(
        conn=conn,
        table="videos",
        stmt="videos_fetched_today",
        params=(keywords_fts_query, since_str),
    )
    rows_to_csv(rows, "videos_fetched_today")
    return rows


def update_video_downloaded(
    conn: sqlite3.Connection, video_id: int, file_path: str, duration: float, title: str
) -> None:
    """Update a video with download details."""
    s_proc(
        conn,
        "videos",
        "update_video_downloaded",
        [(file_path, duration, title, video_id)],
    )


__all__ = [
    "insert_videos",
    "get_videos_to_download",
    "update_video_downloaded",
    "get_videos_fetched_today",
]
