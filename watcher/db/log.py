"""
Logging utilities for the database.
"""

from datetime import datetime, timedelta
from typing import Optional
import sqlite3
import traceback
from watcher.db import rows_to_csv

from . import s_proc, p_query

LOG_MESSAGES = {
    "source_check_started": {
        "event_type": "info",
        "template": "Source check started for {source_name}",
    },
    "source_check_successful": {
        "event_type": "info",
        "template": "Source check successful: {num_videos} videos found for {source_name}",
    },
    "source_check_failed": {
        "event_type": "error",
        "template": "Source check failed for {source_name}: {error_msg}",
    },
    "video_download_failed": {
        "event_type": "error",
        "template": "Video download failed for {video_url}: {error_msg}",
    },
    "video_downloaded": {
        "event_type": "info",
        "template": "Video downloaded: {video_url}",
    },
    "video_transcription_completed": {
        "event_type": "info",
        "template": "Video transcription completed: {video_url}",
    },
    "video_transcription_failed": {
        "event_type": "error",
        "template": "Video transcription failed for {video_url}: {error_msg}",
    },
    "video_embedding_started": {
        "event_type": "info",
        "template": "Video embedding started: {video_url}",
    },
    "video_embedding_completed": {
        "event_type": "info",
        "template": "Video embedding completed: {video_url}",
    },
    "video_embedding_failed": {
        "event_type": "error",
        "template": "Video embedding failed for {video_url}: {error_msg}",
    },
}


def insert_log(
    conn: sqlite3.Connection,
    key: str,
    source_id: Optional[int] = None,
    video_id: Optional[int] = None,
    exception: Optional[Exception] = None,
    **kwargs,
) -> None:
    """Insert a log entry based on the key and provided arguments."""
    if key not in LOG_MESSAGES:
        raise ValueError(f"Unknown log key: {key}")

    log_info = LOG_MESSAGES[key]
    event_type = log_info["event_type"]
    template = log_info["template"]

    # If exception is provided, use full traceback for error_msg
    if exception is not None:
        kwargs["error_msg"] = traceback.format_exc()

    # Format the message with kwargs
    try:
        event_message = template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required argument for log key '{key}': {e}")

    # Print error messages to console for debugging
    if event_type == "error":
        print(f"[ERROR] {event_message}")
        if exception is not None:
            print("Full traceback:")
            print(traceback.format_exc())

    # Ensure source_id is provided for source-related logs
    if source_id is None:
        raise ValueError(f"source_id required for log key '{key}'")

    # Ensure video_id is provided for video-related logs
    if video_id is None and key.startswith("video_"):
        raise ValueError(f"video_id required for log key '{key}'")

    s_proc(conn, "log", "make_log", [(source_id, video_id, event_type, event_message)])


def cleanup_old_logs(conn: sqlite3.Connection, days: int = 30) -> int:
    """Delete logs older than the specified number of days. Returns the number of deleted rows."""
    cutoff_date = datetime.now() - timedelta(days=days)
    cursor = conn.execute("DELETE FROM logs WHERE occurred_at < ?", (cutoff_date,))
    return cursor.rowcount


def print_today_logs(conn: sqlite3.Connection) -> None:
    """Pretty print all logs from today."""
    rows = p_query(conn, "log", "get_today_logs", ())
    if not rows:
        print("No logs found for today.")
        return
    for row in rows:
        timestamp = row["occurred_at"]
        event_type = row["event_type"]
        message = row["event_message"]
        print(f"{timestamp} [{event_type.upper()}] {message}")
    rows_to_csv(rows, "logs")


__all__ = ["LOG_MESSAGES", "insert_log", "cleanup_old_logs", "print_today_logs"]
