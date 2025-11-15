-- table: sources
-- create
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT NOT NULL,
    scraper_name TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source),
    UNIQUE(scraper_name)
);
CREATE INDEX IF NOT EXISTS idx_scraper_active ON sources(scraper_name) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_active ON sources(active);
-- upsert
INSERT INTO sources (source, url, description, scraper_name, active)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(scraper_name) DO UPDATE SET
    source = excluded.source,
    url = excluded.url,
    description = excluded.description,
    active = excluded.active;
-- all
SELECT * FROM sources;
-- scraper_name
SELECT * FROM sources WHERE scraper_name = ?;
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
-- get_today_logs
SELECT * FROM logs WHERE DATE(occurred_at) = DATE('now') ORDER BY occurred_at DESC;

-- table: videos
-- create
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    video_url TEXT NOT NULL,
    title TEXT,
    published_at DATETIME,
    seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    video_file_path TEXT,
    duration_seconds INTEGER,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
    UNIQUE(video_url)
);
CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published_at);
-- upsert_video
INSERT INTO videos(source_id, video_url, title, published_at)
VALUES (?, ?, ?, ?)
ON CONFLICT(source_id, video_url) DO UPDATE SET
    title = excluded.title,
    published_at = excluded.published_at;
-- update_video_downloaded
UPDATE videos SET video_file_path = ?, duration_seconds = ?, transcript_file_path = ? WHERE id = ?;
-- get_videos_to_download
SELECT s.scraper_name, s.source_id, v.id as video_id, v.video_url FROM videos v JOIN sources s ON v.source_id = s.id and v.video_file_path IS NULL
-- get_videos_to_transcribe
SELECT v.source_id, v.id as video_id, v.video_file_path FROM videos v LEFT JOIN chunks c on v.id = c.video_id AND c.video_id IS NULL

-- table: chunks
-- All text segments (sentences, paragraphs, summaries, topics, sliding windows, etc.)
-- create
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      INTEGER NOT NULL,
    speaker_id    INTEGER,                            
    start_sec     REAL NOT NULL,
    end_sec       REAL NOT NULL,
    text          TEXT NOT NULL,
    layer         TEXT NOT NULL DEFAULT 'transcript',   
    chunk_type    TEXT NOT NULL,                        
    metadata      JSON,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(video_id)      REFERENCES videos(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_video    ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_time     ON chunks(video_id, start_sec, end_sec);
CREATE INDEX IF NOT EXISTS idx_chunks_layer    ON chunks(layer);
CREATE INDEX IF NOT EXISTS idx_chunks_type     ON chunks(chunk_type);

-- insert_chunk
INSERT INTO chunks (video_id, speaker_id, start_sec, end_sec, layer, chunk_type, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

-- table: chunk_vectors
-- create
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id      INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding     float[768], 
);

-- insert_embedding
INSERT INTO chunk_vectors (chunk_id, embedding)
VALUES (?, ?);