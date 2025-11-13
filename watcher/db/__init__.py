"""
# SQLite Schema + Vector DB Bootstrapper

## Objective
- **All SQL in one file**: `sql.sql`
- **No ORM**: Avoids abstraction overhead, enables direct use of SQLite **vector extensions**
(e.g. `sqlite-vec`, `MATCH`, `embedding BLOB`)
- **Structured parsing**: `-- table:`, `-- create`, `-- upsert`, `-- seed:`
→ auto-generate schema, upserts, seeds
- **Seeds live in `seeds/*.csv`**, referenced via `-- seed: file.csv`

> **Warning: This is a crude hack with lots of room for errors for the unwary.**
> - Regex parsing is fragile to whitespace/comments
> - No SQL validation
> - Seed assumes CSV column order = upsert order
> - No migrations
> - No concurrency handling (beyond SQLite WAL)
>
> Use for prototyping or embedded apps. For production, prefer `alembic` + per-table `.sql` files.
"""

import csv
import re
import sqlite3
import sqlite_vec
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Any, Generator
import threading

# ----------------------------------------------------------------------
# Paths & Global Cache
# ----------------------------------------------------------------------
SQL_FILE = Path(__file__).with_name("sql.sql")
SEEDS_DIR = Path(__file__).parent / "seeds"
DB_PATH = Path(__file__).parent / "watcher.db"

sql_cache: Dict[str, Dict[str, Any]] = {}


# ----------------------------------------------------------------------
# 1. SQL Parsing
# ----------------------------------------------------------------------
def init_sql_cache() -> Dict[str, Dict[str, Any]]:
    """Parse `sql.sql` → cache of table metadata."""
    if sql_cache:
        return sql_cache

    content = SQL_FILE.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in content.splitlines()]

    tables: Dict[str, Dict[str, Any]] = {}
    current_table: str | None = None
    current_block: str | None = None
    buffer: List[str] = []

    block_pat = re.compile(r"--\s*(create|upsert|delete)\s*$", re.I)
    attr_pat = re.compile(r"--\s*(\w+):\s*(.+)", re.I)

    def flush() -> None:
        nonlocal buffer
        if buffer and current_block and current_table:
            sql = "\n".join(buffer).strip()
            tables[current_table][current_block] = sql
        buffer = []

    for line in lines:
        if not line.strip() and not buffer:
            continue

        # -- key: value
        if m := attr_pat.match(line):
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "table":
                flush()
                current_table = val
                tables[current_table] = {"create": None, "upsert": None, "delete": None}
                current_block = None
            elif current_table:
                tables[current_table][key] = val
            continue

        # -- create / upsert / delete
        if m := block_pat.match(line):
            flush()
            current_block = m.group(1).lower()
            continue

        # SQL body
        if current_block and current_table:
            buffer.append(line)

    flush()
    sql_cache.update(tables)
    return tables


# ----------------------------------------------------------------------
# 2. DB Initialization
# ----------------------------------------------------------------------
def _init_db(conn: sqlite3.Connection, seed: bool = True) -> None:
    """Create tables and optionally seed from CSV."""
    schema = init_sql_cache()
    for table, info in schema.items():
        if info.get("create"):
            conn.executescript(info["create"])
        if seed and info.get("seed"):
            _seed(conn, table, info["seed"])


def _seed(conn: sqlite3.Connection, table: str, seed_file: str) -> None:
    path = SEEDS_DIR / seed_file
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [tuple(row[col] for col in reader.fieldnames) for row in reader]
        if rows:
            _execute_statement(conn, table, "upsert", rows)


# ----------------------------------------------------------------------
# 3. CRUD Helpers
# ----------------------------------------------------------------------
def _execute_statement(
    conn: sqlite3.Connection, table: str, stmt_type: str, rows: List[Tuple]
) -> None:
    stmt = sql_cache.get(table, {}).get(stmt_type)
    if not stmt:
        raise ValueError(f"No {stmt_type} statement for table '{table}'")
    conn.executemany(stmt, rows)


def upsert(conn: sqlite3.Connection, table: str, rows: List[Tuple]) -> None:
    """Insert or update rows."""
    _execute_statement(conn, table, "upsert", rows)


def delete(conn: sqlite3.Connection, table: str, rows: List[Tuple]) -> None:
    """Delete rows by ID."""
    _execute_statement(conn, table, "delete", rows)


# ----------------------------------------------------------------------
# 4. DB Lifecycle & Context Manager (sqlite-vec)
# ----------------------------------------------------------------------


@contextmanager
def connect() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for DB connection.
    - Auto-creates DB on first use
    - Enables WAL, foreign_keys, row_factory
    - Guarantees commit/rollback/close
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_db_exists() -> None:
    """Create DB file + schema + vec + seeds if missing."""
    if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        _init_db(conn, seed=True)


_ensure_db_exists()

__all__ = ["connect", "upsert", "delete"]
