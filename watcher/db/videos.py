"""
Video-related database operations.
"""

from typing import List, Tuple, Any
import sqlite3

from watcher.db import s_proc, p_query


def insert_videos(conn: sqlite3.Connection, videos: List[Tuple[Any, ...]]) -> None:
    """Bulk insert videos into the database."""
    s_proc(conn, "videos", "upsert_video", videos)


def get_videos_to_download(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Get videos that need downloading."""
    return p_query(conn, "videos", "get_videos_to_download", ())


def update_video_downloaded(
    conn: sqlite3.Connection, video_id: int, file_path: str, duration: float
) -> None:
    """Update a video with download details."""
    s_proc(
        conn,
        "videos",
        "update_video_downloaded",
        [(file_path, duration, video_id)],
    )


__all__ = [
    "insert_videos",
    "get_videos_to_download",
    "update_video_downloaded",
]
