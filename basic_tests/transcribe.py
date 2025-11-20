"""
scrape, download and transcribe videos and report out.
"""

from watcher.transcriber import transcribe_videos_to_db, embed_chunks_in_db
from watcher.browser import get_videos_in_last_n_days
from watcher.db.videos import get_videos_fetched_today
from watcher.db.chunks import export_today_chunk_hits
from db_tests import create_test_db, delete_test_db, describe_db
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from watcher.db.log import print_today_logs


def main():

    with create_test_db(reset=False) as conn:
        describe_db(conn)
        print_today_logs(conn)
        get_videos_in_last_n_days(conn, n=2, vid_count=20)
        transcribe_videos_to_db(conn)
        describe_db(conn)
        embed_chunks_in_db(conn)
        describe_db(conn)
        print_today_logs(conn)
        _ = get_videos_fetched_today(conn)
        _ = export_today_chunk_hits(conn)
        describe_db(conn)
    # delete_test_db()


if __name__ == "__main__":
    main()
