"""download utilities for videos."""

from pathlib import Path
import subprocess
import json
from typing import Tuple
import yt_dlp

DOWNLOADS_DIR = Path(__file__).parent.parent.parent / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def list_formats(url: str):
    """Prints available formats for debugging."""
    ydl_opts = {
        "listformats": True,  # ← This is the key flag
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            print(f"Title: {info.get('title', 'N/A')}")
            print(f"Duration: {info.get('duration', 'N/A')}s")
            print("\nAvailable Formats:")
            for f in info.get("formats", []):
                print(
                    f"ID: {f.get('format_id')}, Ext: {f.get('ext')}, Res: {f.get('height')}p, Filesize: {f.get('filesize_approx', 'N/A')} bytes"
                )
        except Exception as e:
            print(f"Error: {e}")


def download_video(url: str, output_filename: str) -> Tuple[Path, float, str]:
    path = DOWNLOADS_DIR / f"{Path(output_filename).stem}.mp4"
    list_formats(url)
    ydl_opts = {
        "format": "worstvideo+bestaudio/best",
        "outtmpl": str(path),
        "concurrent_fragment_downloads": 16,
        "retries": 30,
        "fragment_retries": 30,
        "continuedl": True,
        "merge_output_format": "mp4",
        "get_duration": True,
        "progress": True,
        "quiet": True,
        "console_title": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        duration = info.get("duration")
        title = info.get("title")

    if duration is None:
        raise ValueError("Could not determine video duration")

    return path, float(duration), title


__all__ = [
    "download_video",
    "DOWNLOADS_DIR",
]
