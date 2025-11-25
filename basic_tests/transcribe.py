"""
scrape, download and transcribe videos and report out.
"""

from watcher.transcriber import (
    transcribe_videos_to_db,
    prepare_transcription_model,
    prepare_diarization_pipeline,
)
from watcher.browser import get_videos_in_last_n_days, download_videos
from watcher.db.videos import (
    get_videos_fetched_today,
    get_videos_to_download,
)
from watcher.db.chunks import export_today_chunk_hits
from db_tests import create_test_db, describe_db
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from watcher.db.log import print_today_logs


def main():

    with create_test_db(reset=False) as conn:
        # get all the webscrapers for each source (there's only cpac right now)
        # and then with each of them get the videos from last n days (writes it to sqlite db as it goes)
        scrapers = get_videos_in_last_n_days(conn, n=2, vid_count=5)
        # gets the videos from sqlite
        videos_to_download = get_videos_to_download(conn)
        # a report for what videos were downloaded
        _ = get_videos_fetched_today(conn)
        # loading a model for transcription locally takes a minute
        whisper_model = prepare_transcription_model()
        # diarization is figuring out who was saying what but its buggy right now. This also loads a model
        diarization_pipeline = prepare_diarization_pipeline()
        for video in videos_to_download:
            try:
                # each scraper can have a different way to download a video but I'm not sure that's necessary yet
                download_videos(conn, scrapers, [video])
                # now that its downloaded, transcribe it and put the chunks into the db
                transcribe_videos_to_db(
                    conn,
                    whisper_model=whisper_model,
                    diarization_pipeline=diarization_pipeline,
                    export_chunks_after_each_video=True,
                )
                # this is just me making sure the db is updating
                describe_db(conn)
            except Exception as e:
                print(f"Error processing video {video['video_id']}: {e}")
                continue
        # a report for what videos were downloaded
        _ = get_videos_fetched_today(conn)
        # a report for what transcript chunks had hits on keywords of interest
        _ = export_today_chunk_hits(conn)
        print_today_logs(conn)
        # embed_chunks_in_db(conn)
        # describe_db(conn)

    # delete_test_db()


if __name__ == "__main__":
    main()
