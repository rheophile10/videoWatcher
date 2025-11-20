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
from typing import Dict, List, Tuple, Any, Generator, Iterable, Optional
from datetime import datetime


SQL_FILE = Path(__file__).with_name("sql.sql")
SEEDS_DIR = Path(__file__).parent / "seeds"
DB_PATH = Path(__file__).parent.parent.parent / "watcher.db"
EXPORTS_PATH = Path(__file__).parent.parent.parent / "exports"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_PATH.mkdir(parents=True, exist_ok=True)

sql_cache: Dict[str, Dict[str, Any]] = {}


def _init_sql_cache() -> Dict[str, Dict[str, Any]]:
    """Parse `sql.sql` → cache of table metadata."""
    if sql_cache:
        return sql_cache

    content = SQL_FILE.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in content.splitlines()]

    tables: Dict[str, Dict[str, Any]] = {}
    current_table: str | None = None
    current_block: str | None = None
    buffer: List[str] = []

    block_pat = re.compile(r"--\s*([a-z_]+)\s*$", re.I)
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
                tables[current_table] = {}
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


def _init_db(conn: sqlite3.Connection, seed: bool = True) -> None:
    """Create tables and optionally seed from CSV."""
    schema = _init_sql_cache()
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
            upsert(conn, table, rows)


def _get_statement(table: str, stmt_name: str) -> str:
    stmt = sql_cache.get(table, {}).get(stmt_name)
    if not stmt:
        raise ValueError(f"No {stmt_name} statement for table '{table}'")
    return stmt


def s_proc(
    conn: sqlite3.Connection, table: str, stmt_type: str, rows: List[Tuple]
) -> None:
    """Execute a statement for multiple rows."""
    stmt = _get_statement(table, stmt_type)
    conn.executemany(stmt, rows)
    conn.commit()


def _query_statement(
    conn: sqlite3.Connection, stmt: str, params: Optional[Tuple] = None
) -> List[sqlite3.Row]:
    """Execute a query and return all results."""
    if params is None:
        cur = conn.execute(stmt)
    else:
        cur = conn.execute(stmt, params)
    return cur.fetchall()


def p_query(
    conn: sqlite3.Connection, table: str, stmt: str, params: Optional[Tuple] = None
) -> List[sqlite3.Row]:
    """Parameterized query."""
    stmt = sql_cache.get(table, {}).get(stmt)
    if not stmt:
        raise ValueError(f"No {stmt} statement for table '{table}'")
    return _query_statement(conn, stmt, params)


def batched_insert(
    conn: sqlite3.Connection,
    table_name: str,
    stmt_name: str,
    chunk_embeddings: Iterable,
    batch_size: int = 100,
) -> None:
    """Insert chunk embeddings from a generator in batches."""
    batch = []
    for chunk in chunk_embeddings:
        batch.append(chunk)
        if len(batch) == batch_size:
            s_proc(conn, table_name, stmt_name, batch)
            batch.clear()
    if batch:
        s_proc(conn, table_name, stmt_name, batch)


def upsert(conn: sqlite3.Connection, table: str, rows: List[Tuple]) -> None:
    """Insert or update rows."""
    s_proc(conn, table, "upsert", rows)


def delete(conn: sqlite3.Connection, table: str, rows: List[Tuple]) -> None:
    """Delete rows by ID."""
    s_proc(conn, table, "delete", rows)


def rows_to_csv(rows: List[sqlite3.Row], report_name: str) -> Path:
    if not rows:
        raise ValueError("No rows to export")

    path = (
        EXPORTS_PATH
        / f"export_{report_name}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    rows = [dict(row) for row in rows]
    keys = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows → {path.absolute()}")
    return path


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for DB connection.
    - Auto-creates DB on first use
    - Enables WAL, foreign_keys, row_factory
    - Guarantees commit/rollback/close
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
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


def _ensure_db_exists(db_path: Path = DB_PATH) -> None:
    """Create DB file + schema + vec + seeds if missing."""
    _init_sql_cache()  # Always populate the SQL cache
    if db_path.exists() and db_path.stat().st_size > 0:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        _init_db(conn, seed=True)


@contextmanager
def db(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Get a DB connection, ensuring DB exists."""
    _ensure_db_exists(db_path)
    with _connect(db_path) as conn:
        yield conn


__all__ = [
    "db",
    "upsert",
    "delete",
    "s_proc",
    "p_query",
    "log",
    "DB_PATH",
]
