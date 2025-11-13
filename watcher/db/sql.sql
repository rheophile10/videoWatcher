-- table: sources
-- create
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT NOT NULL,
    UNIQUE(entity)
);
-- upsert
INSERT INTO sources(entity, url, description)
VALUES (?, ?, ?)
ON CONFLICT(entity) DO UPDATE SET
    url = excluded.url,
    description = excluded.description;
-- seed: sources.csv

-- table: videos
-- create
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    video_url TEXT NOT NULL,
    title TEXT,
    published_at TEXT,
    downloaded_at TEXT,
    file_path TEXT,
    duration_seconds INTEGER,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published_at);
-- upsert
INSERT INTO videos(source_id, video_url, title, published_at, status)
VALUES (?, ?, ?, ?, ?)

-- table: snippets
-- create
CREATE TABLE IF NOT EXISTS snippets (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    video_url TEXT,
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