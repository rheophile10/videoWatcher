from pathlib import Path
import pytest
import numpy as np

from watcher.db import db, upsert

TEST_DB_PATH = Path(__file__).parent / "test_watcher.db"


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    yield
    # Force-close any leaked connections
    import gc, time

    gc.collect()
    time.sleep(0.1)  # let Windows release file lock
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink(missing_ok=True)
        except PermissionError:
            pass  # best effort


@pytest.fixture
def conn():
    with db(TEST_DB_PATH) as c:
        yield c


def test_connect_creates_db(conn):
    assert TEST_DB_PATH.exists()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    names = {r[0] for r in tables}
    assert "sources" in names
    assert "items" in names
    assert "vec_items" in names


def test_upsert_sources(conn):
    upsert(conn, "sources", [("alice", "https://a.com", "Alice Co")])
    row = conn.execute("SELECT * FROM sources WHERE entity=?", ("alice",)).fetchone()
    assert row["entity"] == "alice"

    upsert(conn, "sources", [("alice", "https://b.com", "Updated")])
    row = conn.execute(
        "SELECT description FROM sources WHERE entity=?", ("alice",)
    ).fetchone()
    assert row["description"] == "Updated"


def test_vector_search(conn):
    # Insert text
    cur = conn.cursor()
    cur.execute("INSERT INTO items(text) VALUES ('hello world')")
    item_id = cur.lastrowid

    # Insert zero vector
    zero_vec = bytes(np.zeros(384, dtype="float32"))
    cur.execute(
        "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", (item_id, zero_vec)
    )
    conn.commit()

    results = conn.execute(
        """
        SELECT 
            i.text,
            distance
        FROM vec_items v
        JOIN items i ON i.id = v.rowid
        WHERE v.embedding MATCH ?
            AND k = 1
        ORDER BY distance
        """,
        (zero_vec,),
    ).fetchall()

    assert len(results) == 1
    assert results[0]["text"] == "hello world"
    assert results[0]["distance"] == 0.0
