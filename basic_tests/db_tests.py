"""Basic tests for db functions. Run with: python basic_tests/db_tests.py"""

from pathlib import Path
import numpy as np

from watcher.db import db, upsert

TEST_DB_PATH = Path(__file__).parent / "test_watcher.db"


def cleanup():
    import gc, time

    gc.collect()
    time.sleep(0.1)
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_connect_creates_db():
    try:
        with db(TEST_DB_PATH) as conn:
            assert TEST_DB_PATH.exists()
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            names = {r[0] for r in tables}
            assert "sources" in names
            assert "items" in names
            assert "vec_items" in names
        print("✓ test_connect_creates_db passed")
    except Exception as e:
        print(f"✗ test_connect_creates_db failed: {e}")


def test_upsert_sources():
    try:
        with db(TEST_DB_PATH) as conn:
            upsert(conn, "sources", [("alice", "https://a.com", "Alice Co")])
            row = conn.execute(
                "SELECT * FROM sources WHERE entity=?", ("alice",)
            ).fetchone()
            assert row["entity"] == "alice"

            upsert(conn, "sources", [("alice", "https://b.com", "Updated")])
            row = conn.execute(
                "SELECT description FROM sources WHERE entity=?", ("alice",)
            ).fetchone()
            assert row["description"] == "Updated"
        print("✓ test_upsert_sources passed")
    except Exception as e:
        print(f"✗ test_upsert_sources failed: {e}")


if __name__ == "__main__":
    print("Running db tests...")
    test_connect_creates_db()
    test_upsert_sources()
    cleanup()
    print("All db tests completed!")
