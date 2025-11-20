"""download utilities for videos."""

from pathlib import Path
from typing import Tuple
import yt_dlp
from datetime import datetime

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


def download_video(
    url: str, output_filename: str, show_formats: bool = False
) -> Tuple[Path, float, str]:
    stem = Path(f"{output_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path_template = DOWNLOADS_DIR / f"{stem}.%(ext)s"

    ydl_opts = {
        "format": "worstvideo+bestaudio/best",
        "outtmpl": str(path_template),
        "concurrent_fragment_downloads": 16,
        "retries": 30,
        "fragment_retries": 30,
        "continuedl": True,
        "merge_output_format": "mp4",
        "get_duration": True,
        "writesubtitles": True,
        "writeautomaticsubtitles": True,
        "subtitleslangs": ["en", "en-US", "en-CA"],
        "subtitlesformat": "srt",
        "skip_download": False,
        "progress": True,
        "quiet": True,
        "console_title": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_path = Path(ydl.prepare_filename(info))

    possible_subs = [
        video_path.with_suffix(".en.srt"),
        video_path.with_suffix(".en-US.srt"),
        video_path.with_suffix(".en-CA.srt"),
        video_path.with_suffix(".srt"),  # fallback
        video_path.with_suffix(".vtt"),
    ]
    subtitle_path = next((p for p in possible_subs if p.exists()), None)
    duration = float(info.get("duration") or 0)
    title = info.get("title", "Unknown Title")
    upload_date = info.get("upload_date")
    publish_date = (
        datetime.strptime(upload_date, "%Y%m%d").strftime("%Y-%m-%d")
        if upload_date and len(upload_date) == 8
        else "Unknown"
    )
    if show_formats:
        print(
            f"video info: {title} ({publish_date}), duration: {duration}s, subtitles: {subtitle_path}"
        )
        list_formats(url)

    return (
        video_path,
        float(duration),
        title,
        # subtitle_path,
    )


__all__ = [
    "download_video",
    "DOWNLOADS_DIR",
]
