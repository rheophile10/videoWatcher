"""
Database testing utilities.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from watcher.db import db

TEST_DB_PATH = Path(__file__).parent / "test.db"


def create_test_db():
    """Create a test database in basic_tests folder and seed it."""
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)

    return db(TEST_DB_PATH)


def delete_test_db() -> None:
    """Delete the test database."""
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)


if __name__ == "__main__":
    # Create test db
    with create_test_db() as conn:
        # Prove sources table has length
        cursor = conn.execute("SELECT COUNT(*) FROM sources")
        count = cursor.fetchone()[0]
        print(f"Sources table has {count} rows")

    # Close and delete
    delete_test_db()
    print("Test database deleted")

__all__ = [
    "create_test_db",
    "delete_test_db",
]
