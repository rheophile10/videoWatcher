"""
Test script to transcribe the first .mp4 in downloads and insert chunks/embeddings into DB.
"""

from watcher.db import p_query, db
from watcher.transcriber import transcribe_videos_to_db, embed_chunks_in_db
from db_tests import create_test_db, delete_test_db, describe_db
from scrape_test import test_get_videos_in_last_n_days
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from watcher.db.log import print_today_logs


def main():

    with create_test_db(reset=False) as conn:
        describe_db(conn)
        videos_to_transcribe = p_query(conn, "videos", "get_videos_to_transcribe", ())
        if len(videos_to_transcribe) == 0:
            print("No videos to transcribe found in DB.")
            print("Running scrape test to populate videos...")
            test_get_videos_in_last_n_days(conn)
        transcribe_videos_to_db(conn)
        describe_db(conn)
        embed_chunks_in_db(conn)
        describe_db(conn)
        print_today_logs(conn)
        describe_db(conn)
    # delete_test_db()


if __name__ == "__main__":
    main()
