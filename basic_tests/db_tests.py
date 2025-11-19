"""
Database testing utilities.
"""

from pathlib import Path
import os
import sys
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))

from watcher.db import db

TEST_DB_PATH = Path(__file__).parent / "test.db"


def create_test_db(reset: bool = True) -> sqlite3.Connection:
    """Create a test database in basic_tests folder and seed it."""
    if reset and TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)

    return db(TEST_DB_PATH)


def describe_db(conn: sqlite3.Connection) -> None:
    """Print all tables and their row counts."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    print("Database table row counts:")
    print("-" * 30)

    for table_row in tables:
        table_name = table_row[0]
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = count_cursor.fetchone()[0]
        print(f"{table_name}: {count} rows")


def delete_test_db() -> None:
    """Delete the test database."""
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)
    shm_path = TEST_DB_PATH.with_suffix(".db-shm")
    wal_path = TEST_DB_PATH.with_suffix(".db-wal")
    if shm_path.exists():
        os.remove(shm_path)
    if wal_path.exists():
        os.remove(wal_path)
    print("Test database deleted")


if __name__ == "__main__":
    with create_test_db() as conn:
        describe_db(conn)
    # delete_test_db()


__all__ = [
    "create_test_db",
    "delete_test_db",
    "describe_db",
]
