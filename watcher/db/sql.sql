-- table: sources
-- create
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT NOT NULL,
    scraper_name TEXT NOT NULL CHECK(scraper_name REGEXP '^[a-z_]+$'),
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_run_at DATETIME,
    last_success_at DATETIME,
    error_log TEXT,
    error_datetime DATETIME,
    UNIQUE(entity),
    UNIQUE(scraper_name)
);
CREATE INDEX IF NOT EXISTS idx_scraper_active ON sources(scraper_name) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_active ON sources(active);
-- upsert
INSERT INTO sources (entity, url, description, scraper_name, active)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(scraper_name) DO UPDATE SET
    entity = excluded.entity,
    url = excluded.url,
    description = excluded.description,
    active = excluded.active;
-- all
SELECT * FROM sources;
-- scraper_name
SELECT * FROM sources WHERE scraper_name = ?;
-- update_last_run_by_scraper_name
UPDATE sources SET last_run_at = CURRENT_TIMESTAMP WHERE scraper_name = ?;
-- update_last_success_by_scraper_name
UPDATE sources SET last_success_at = CURRENT_TIMESTAMP WHERE scraper_name = ?;
-- update_active_by_scraper_name
UPDATE sources SET active = ? WHERE scraper_name = ?;
-- seed: sources.csv

-- table: log
-- create
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    video_id INTEGER,
    event_type TEXT DEFAULT 'error',
    event_message TEXT NOT NULL,
    occurred_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);
-- make_log
INSERT INTO logs (source_id, video_id, event_type, event_message)
VALUES (?, ?, ?, ?);

-- table: videos
-- create
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    video_url TEXT NOT NULL PRIMARY KEY,
    title TEXT,
    published_at TEXT,
    first_download_attempted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    download_attempted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    downloaded_at DATETIME,
    video_file_path TEXT,
    duration_seconds INTEGER,
    transcript_file_path TEXT,
    transcribed_at DATETIME,
    embedding_started_at DATETIME,
    embedding_completed_at DATETIME,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
    UNIQUE(source_id, video_url)
);
CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published_at);
-- update_download_attempt
INSERT INTO videos(source_id, video_url, title, published_at, video_file_path)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(source_id, video_url) DO UPDATE SET
    download_attempted_at = CURRENT_TIMESTAMP,
    video_file_path = excluded.video_file_path;
-- get_video_by_url
SELECT * FROM videos WHERE video_url = ?;
-- mark_downloaded
UPDATE videos SET downloaded_at = CURRENT_TIMESTAMP, video_file_path = ?, duration_seconds = ?
WHERE video_url = ?;
-- mark_transcribed
UPDATE videos SET transcribed_at = CURRENT_TIMESTAMP, transcript_file_path = ?
WHERE video_url = ?;
-- mark_embedding_started
UPDATE videos SET embedding_started_at = CURRENT_TIMESTAMP
WHERE video_url = ?;
-- mark_embedding_completed
UPDATE videos SET embedding_completed_at = CURRENT_TIMESTAMP
WHERE video_url = ?;
-- get_videos_to_download
SELECT * FROM videos
WHERE downloaded_at IS NULL
-- get_videos_to_transcribe
SELECT * FROM videos
WHERE downloaded_at IS NOT NULL
    AND transcribed_at IS NULL;
-- get_videos_to_embed
SELECT * FROM videos
WHERE transcribed_at IS NOT NULL
    AND embedding_completed_at IS NULL;


-- table: snippets
-- create
CREATE TABLE IF NOT EXISTS snippets (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    snippet_url TEXT,
    start_sec REAL,
    end_sec REAL,
    text TEXT,
    embedding BLOB,
    metadata JSON,
    suggested_copy TEXT,
    relevance_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- table: items
-- create
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    text TEXT
);

-- table: vec_items
-- create
CREATE VIRTUAL TABLE vec_items USING vec0(
    embedding float[384]
);