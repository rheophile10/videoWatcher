"""
Chunk-related database operations.
"""

from typing import Tuple, Any, Generator, List
import sqlite3
from watcher.db import p_query, batched_insert, rows_to_csv
from datetime import datetime


def get_chunks_without_embeddings(
    conn: sqlite3.Connection,
    limit: int = 100,
) -> List[sqlite3.Row]:
    """Retrieve chunks that do not have embeddings yet."""
    return p_query(
        conn,
        "chunks",
        "get_chunks_without_embeddings",
        (limit,),
    )


def export_today_chunk_hits(
    conn: sqlite3.Connection,
    since: datetime = datetime.combine(datetime.today(), datetime.min.time()),
    contest_chunks: int = 5,
    keywords_fts_query: str = "gun OR firearm OR rifle",
) -> List[sqlite3.Row]:
    """Export chunks with keyword hits from videos seen today."""
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    rows = p_query(
        conn,
        "chunks",
        "export_today_chunks_with_video_info",
        params=(keywords_fts_query, since_str, contest_chunks, since_str),
    )
    rows_to_csv(rows, "today_chunk_hits")
    return rows


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: Generator[Tuple[Tuple[Any, ...], Tuple[Any, ...]], None, None],
    batch_size: int = 100,
) -> None:
    """Insert batches of chunks and their embeddings into the DB from a generator of (chunk_tuple, embedding_tuple)."""
    batched_insert(
        conn,
        "chunks",
        "insert_chunk_with_embedding",
        chunks,
        batch_size=batch_size,
    )


__all__ = ["insert_chunks", "get_chunks_without_embeddings"]
