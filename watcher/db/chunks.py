"""
Chunk-related database operations.
"""

from typing import Tuple, Any, Generator
import sqlite3
from watcher.db import s_proc, s_proc_with_ids


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: Generator[Tuple[Any, ...], None, None],
    batch_size: int = 100,
) -> Generator[int, None, None]:
    """Insert chunks from a generator in batches and yield their IDs."""
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            ids = s_proc_with_ids(conn, "chunks", "insert_chunk", batch)
            yield from ids
            batch.clear()
    if batch:
        ids = s_proc_with_ids(conn, "chunks", "insert_chunk", batch)
        yield from ids


def insert_chunk_vectors(
    conn: sqlite3.Connection,
    chunk_embeddings: Generator[Tuple[int, Tuple[Any, ...]], None, None],
    batch_size: int = 100,
) -> None:
    """Insert chunk embeddings from a generator in batches."""
    batch = []
    for chunk_id, emb in chunk_embeddings:
        batch.append((chunk_id,) + emb)
        if len(batch) >= batch_size:
            s_proc(conn, "chunk_vectors", "insert_embedding", batch)
            batch.clear()
    if batch:
        s_proc(conn, "chunk_vectors", "insert_embedding", batch)


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    chunk_embeddings: Generator[Tuple[Tuple[Any, ...], Tuple[Any, ...]], None, None],
    batch_size: int = 100,
) -> None:
    """Insert batches of chunks and their embeddings into the DB from a generator of (chunk_tuple, embedding_tuple)."""
    chunk_batch = []
    embedding_batch = []
    for chunk_tuple, embedding_tuple in chunk_embeddings:
        chunk_batch.append(chunk_tuple)
        embedding_batch.append(embedding_tuple)
        if len(chunk_batch) >= batch_size:
            chunk_ids = s_proc_with_ids(conn, "chunks", "insert_chunk", chunk_batch)
            emb_data = [(cid,) + emb for cid, emb in zip(chunk_ids, embedding_batch)]
            s_proc(conn, "chunk_vectors", "insert_embedding", emb_data)
            chunk_batch.clear()
            embedding_batch.clear()
    if chunk_batch:
        chunk_ids = s_proc_with_ids(conn, "chunks", "insert_chunk", chunk_batch)
        emb_data = [(cid,) + emb for cid, emb in zip(chunk_ids, embedding_batch)]
        s_proc(conn, "chunk_vectors", "insert_embedding", emb_data)


__all__ = ["insert_chunks_and_embeddings"]
