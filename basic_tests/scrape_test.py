"""
Test scraping functionality.
"""

from db_tests import create_test_db, delete_test_db
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from watcher.browser import get_scrapers, get_videos_in_last_n_days
from watcher.db import p_query
from watcher.db.log import print_today_logs


def test_get_scrapers_toggles_active(conn):
    """Test that get_scrapers toggles active flags appropriately."""
    # Get initial active status
    initial_sources = p_query(conn, "sources", "all", ())
    initial_active = {row["scraper_name"]: row["active"] for row in initial_sources}
    print(f"Initial active status: {initial_active}")

    # Get scrapers (this should toggle active flags)
    scrapers = get_scrapers(conn)
    print(f"Retrieved scrapers: {list(scrapers.keys())}")

    # Check updated active status
    updated_sources = p_query(conn, "sources", "all", ())
    updated_active = {row["scraper_name"]: row["active"] for row in updated_sources}
    print(f"Updated active status: {updated_active}")

    # Verify that active flags were toggled appropriately
    for name, was_active in initial_active.items():
        now_active = updated_active.get(name, 0)
        if was_active == 0 and now_active == 1:
            print(f"✓ {name} was activated")
        elif was_active == 1 and now_active == 0:
            print(f"✗ {name} was deactivated unexpectedly")
        else:
            print(f"- {name} active status unchanged")


def test_get_videos_in_last_n_days(conn):
    """Test get_videos_in_last_n_days (mock test, since no real scrapers)."""
    print("Testing get_videos_in_last_n_days...")
    # This will attempt to run scrapers, but since they may not exist, it should log errors
    get_videos_in_last_n_days(conn, n=1, vid_count=1)


def main():
    # Create test db
    with create_test_db() as conn:
        test_get_scrapers_toggles_active(conn)
        test_get_videos_in_last_n_days(conn)
        print_today_logs(conn)
    delete_test_db()
    print("Test database deleted")


if __name__ == "__main__":
    main()
