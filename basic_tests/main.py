"""
Main test.
"""

from db_tests import create_test_db, delete_test_db
from watcher.db.log import cleanup_old_logs


def main():
    print("Creating test database...")
    conn = create_test_db()
    print("Test database created and seeded.")

    cleanup_old_logs(conn)

    conn.close()
    delete_test_db()
    print("Test database deleted.")


if __name__ == "__main__":
    main()
