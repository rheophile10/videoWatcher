"""
Test script to transcribe the first .mp4 in downloads and insert chunks/embeddings into DB.
"""

from watcher.db.videos import transcribe_videos_to_db
from db_tests import create_test_db, delete_test_db
from scrape_test import test_get_videos_in_last_n_days
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from watcher.db.log import print_today_logs


def main():

    with create_test_db() as conn:
        test_get_videos_in_last_n_days(conn)
        transcribe_videos_to_db(conn)
        print_today_logs(conn)
    delete_test_db()


if __name__ == "__main__":
    main()
