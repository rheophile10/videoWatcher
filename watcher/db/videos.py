"""
Video-related database operations.
"""

from typing import List, Tuple, Any
import sqlite3
from pathlib import Path
from watcher.db.chunks import insert_chunks_and_embeddings
from watcher.db.log import insert_log
from watcher.db import s_proc, p_query


def insert_videos(conn: sqlite3.Connection, videos: List[Tuple[Any, ...]]) -> None:
    """Bulk insert videos into the database."""
    s_proc(conn, "videos", "upsert_video", videos)


def get_videos_to_download(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Get videos that need downloading."""
    return p_query(conn, "videos", "get_videos_to_download", ())


def update_video_downloaded(
    conn: sqlite3.Connection,
    video_id: int,
    file_path: str,
    duration: float,
    transcript_path: str = None,
) -> None:
    """Update a video with download details."""
    s_proc(
        conn,
        "videos",
        "update_video_downloaded",
        [(file_path, duration, transcript_path, video_id)],
    )


def get_videos_to_transcribe(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Get videos that need transcription."""
    return p_query(conn, "videos", "get_videos_to_transcribe", ())


def transcribe_video_to_db(
    conn: sqlite3.Connection, video_path: Path, video_id: int, source_id: int
) -> None:
    from watcher.transcriber import orchestrate_transcription_and_embedding

    try:
        insert_chunks_and_embeddings(
            conn,
            orchestrate_transcription_and_embedding(video_path, video_id),
            batch_size=10,
        )
        insert_log(
            conn,
            "video_transcription_completed",
            source_id=source_id,
            video_id=video_id,
            video_url=str(video_path),
        )
    except Exception as e:
        insert_log(
            conn,
            "video_transcription_failed",
            source_id=source_id,
            video_id=video_id,
            video_url=str(video_path),
            error_msg=str(e),
        )


def transcribe_videos_to_db(conn: sqlite3.Connection) -> None:
    videos_to_transcribe = get_videos_to_transcribe(conn)
    for video in videos_to_transcribe:
        video_id = video["id"]
        video_path = Path(video["file_path"])
        source_id = video["source_id"]
        transcribe_video_to_db(conn, video_path, video_id, source_id)


__all__ = [
    "insert_videos",
    "get_videos_to_download",
    "get_videos_to_transcribe",
    "update_video_downloaded",
    "transcribe_videos_to_db",
    "transcribe_video_to_db",
]
