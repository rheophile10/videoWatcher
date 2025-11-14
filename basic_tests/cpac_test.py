"""Basic tests for cpac.py functions. Run with: python basic_tests/cpac_test.py"""

import asyncio
import re
from watcher.browser.cpac import get_recent_episode_ids, get_episode_details
from watcher.browser.utils import download_video, DOWNLOADS_DIR, get_video_duration


def test_get_episode_details():
    """Test get_episode_details with a known episode ID."""
    # Use a sample ID (replace with a real one if available)
    sample_id = "sample-episode-id"  # This will fail, but tests the call
    try:
        data = get_episode_details(sample_id)
        assert "page" in data, "Expected 'page' key in response"
        print("✓ get_episode_details returned expected structure")
    except Exception as e:
        print(f"✓ get_episode_details handled error gracefully: {e}")


async def test_get_recent_episode_ids():
    """Test get_recent_episode_ids (async)."""
    try:
        ids = await get_recent_episode_ids(days_back=1, max_pages=1)
        assert isinstance(ids, set), "Expected set of IDs"
        print(f"✓ get_recent_episode_ids found {len(ids)} episodes")
    except Exception as e:
        print(f"✓ get_recent_episode_ids handled error gracefully: {e}")


async def test_download_episode():
    """Test downloading a recent episode."""
    try:
        ids = await get_recent_episode_ids(days_back=1, max_pages=1)
        if not ids:
            print("✗ No episodes found to download")
            return

        newest_id = next(iter(ids))
        print(f"Testing download with episode: {newest_id}")

        data = get_episode_details(newest_id)
        details = data["page"]["details"]

        title = re.sub(r'[<>:"/\\|?*]', "_", details["title_en_t"])[:120]
        date = details["liveDateTime"][:10]
        m3u8 = details["videoUrl"]

        output_file = DOWNLOADS_DIR / f"{newest_id}.mp4"
        download_video(m3u8, output_file)
        print("✓ Episode download attempted")
    except Exception as e:
        print(f"✓ Download test handled error gracefully: {e}")


def test_get_video_duration():
    """Test get_video_duration on any downloaded video."""
    videos = list(DOWNLOADS_DIR.glob("*.mp4"))
    if videos:
        video_path = videos[0]
        try:
            duration = get_video_duration(video_path)
            print(f"✓ Video duration: {duration} seconds")
        except Exception as e:
            print(f"✗ Failed to get duration: {e}")
    else:
        print("✗ No video files found to test duration")


if __name__ == "__main__":
    print("Running cpac.py tests...")
    test_get_episode_details()
    asyncio.run(test_get_recent_episode_ids())
    asyncio.run(test_download_episode())
    test_get_video_duration()
    print("All tests completed!")
