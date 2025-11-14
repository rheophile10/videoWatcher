"""download utilities for videos."""

from pathlib import Path
import subprocess
import json

DOWNLOADS_DIR = Path(__file__).parent.parent.parent / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


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


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video file in seconds using ffprobe."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    duration_str = data.get("format", {}).get("duration")
    if duration_str is None:
        raise ValueError("Duration not found in ffprobe output")

    return float(duration_str)
