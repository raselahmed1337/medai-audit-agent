"""Durable execution layer for long-lived deployments.

When DATABASE_URL is set (e.g. by docker-compose), the app uses:
  * a shared PostgresSaver so graph checkpoints — including paused approval
    interrupts — survive process restarts;
  * a `runs` registry table so the console can rebuild the run list;
  * per-run audit JSONL files under AUDIT_DIR so audit trails persist.

Without DATABASE_URL everything stays in-memory (tests, quick demos) and the
behaviour is unchanged.
"""
from __future__ import annotations

import contextlib
import os
import zlib

_SAVER = None
_STACK = contextlib.ExitStack()
_REGISTRY_CONN = None


def _db_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url and not url.startswith("postgresql"):
        url = url.replace("postgres://", "postgresql://", 1)  # Render-style URLs
    return url


def audit_path(run_id: str) -> str | None:
    """Per-run audit file path when AUDIT_DIR is configured, else None."""
    audit_dir = os.environ.get("AUDIT_DIR")
    return os.path.join(audit_dir, f"audit-{run_id}.jsonl") if audit_dir else None


def run_seed(run_id: str) -> int:
    """Stable fault-injection seed so rebuilt runs replay the same fault model."""
    return zlib.crc32(run_id.encode())


def get_checkpointer():
    """Shared checkpointer: PostgresSaver when DATABASE_URL is set, else in-memory.

    The saver is process-wide; thread_id (= run_id) isolates runs.
    """
    global _SAVER
    if _SAVER is not None:
        return _SAVER
    db = _db_url()
    if not db:
        from langgraph.checkpoint.memory import InMemorySaver
        _SAVER = InMemorySaver()
        return _SAVER
    from langgraph.checkpoint.postgres import PostgresSaver
    # from_conn_string is a @contextmanager in current versions: enter it once
    # process-wide and keep the saver alive for the server's lifetime.
    saver = _STACK.enter_context(PostgresSaver.from_conn_string(db))
    saver.setup()  # idempotent DDL
    _SAVER = saver
    return _SAVER


# ---------------- runs registry ----------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    query        TEXT NOT NULL,
    live         BOOLEAN NOT NULL DEFAULT FALSE,
    failure_rate REAL NOT NULL DEFAULT 0,
    created      DOUBLE PRECISION NOT NULL
)"""


def _registry():
    global _REGISTRY_CONN
    if _REGISTRY_CONN is not None:
        return _REGISTRY_CONN
    db = _db_url()
    if not db:
        return None
    import psycopg
    _REGISTRY_CONN = psycopg.connect(db, autocommit=True)
    with _REGISTRY_CONN.cursor() as cur:
        cur.execute(_DDL)
    return _REGISTRY_CONN


def registry_insert(run_id: str, query: str, live: bool, failure_rate: float, created: float) -> None:
    conn = _registry()
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute("INSERT INTO runs VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (run_id, query, live, failure_rate, created))


def registry_delete(run_id: str) -> None:
    conn = _registry()
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))


def registry_list() -> list[dict]:
    """All registered runs (used to rebuild the console after a restart)."""
    conn = _registry()
    if conn is None:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT run_id, query, live, failure_rate, created FROM runs ORDER BY created DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
