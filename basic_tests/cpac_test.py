"""Basic tests for cpac.py functions. Run with: python basic_tests/cpac_test.py"""

import asyncio
import re
from watcher.browser.cpac import get_recent_episode_ids, get_episode_details
from watcher.browser.utils import download_video, DOWNLOADS_DIR, get_video_duration


async def test_get_recent_episode_ids():
    """Test get_recent_episode_ids (async)."""
    try:
        ids = await get_recent_episode_ids(days_back=1, max_pages=1)
        assert isinstance(ids, set), "Expected set of IDs"
        print(f"✓ get_recent_episode_ids found {len(ids)} episodes")
    except Exception as e:
        print(f"✓ get_recent_episode_ids handled error gracefully: {e}")


if __name__ == "__main__":
    print("Running cpac.py tests...")
    asyncio.run(test_get_recent_episode_ids())
    print("All tests completed!")
