from pathlib import Path
import pytest
import numpy as np

from db import connect, upsert

DB_PATH = Path(__file__).parent / "test_watcher.db"


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    yield
    # Force-close any leaked connections
    import gc, time

    gc.collect()
    time.sleep(0.1)  # let Windows release file lock
    if DB_PATH.exists():
        try:
            DB_PATH.unlink(missing_ok=True)
        except PermissionError:
            pass  # best effort


@pytest.fixture
def conn():
    # Patch DB_PATH
    from db import DB_PATH as orig
    import db

    db.DB_PATH = DB_PATH

    with connect() as c:
        yield c

    # Restore
    db.DB_PATH = orig


def test_connect_creates_db(conn):
    assert DB_PATH.exists()
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

    query_vec = zero_vec
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
