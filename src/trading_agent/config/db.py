"""SQLite layer for the cockpit backend: connection helpers + schema bootstrap.

Reuses the framework's existing WAL + autocommit pattern (see
:class:`trading_agent.db.DatabaseManager`) but is dedicated to the per-user
``config.db`` defined in ``design/handoff/CONTRACTS.md §Per-user model``.

Two ways in:
- :func:`connect` / :func:`bootstrap` — the free functions WS-0's contract names.
- :class:`Database` — a thin wrapper holding a path with thread-local
  connections, used by the FastAPI app and the stores so each worker thread gets
  its own connection (FastAPI runs sync handlers in a threadpool).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

# Default location for the per-user config DB. Override with TRADING_AGENT_DB.
# data/*.db is gitignored, so this never leaks into the repo.
DEFAULT_DB_PATH = os.environ.get("TRADING_AGENT_DB", "data/config.db")

# DDL straight from CONTRACTS.md §Per-user model. CREATE ... IF NOT EXISTS keeps
# bootstrap idempotent.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    pw_hash    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,                 -- JSON-encoded
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS endpoints (
    id       TEXT PRIMARY KEY,
    user_id  TEXT NOT NULL,
    type     TEXT NOT NULL,                -- openrouter | openai | anthropic | local
    name     TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key  TEXT NOT NULL DEFAULT '',
    enabled  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,             -- reddit | rss | stocktwits | browser
    name        TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT,
    started_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    scope      TEXT NOT NULL,              -- trader | ticker
    ref        TEXT NOT NULL,
    text       TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_requests (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    trader_id  TEXT,
    symbol     TEXT NOT NULL,
    reason     TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user      ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_endpoints_user     ON endpoints(user_id);
CREATE INDEX IF NOT EXISTS idx_sources_user       ON sources(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_notes_user_scope   ON notes(user_id, scope, ref);
CREATE INDEX IF NOT EXISTS idx_requests_user      ON stock_requests(user_id, status);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a configured SQLite connection (WAL, autocommit, busy_timeout).

    ``check_same_thread=False`` so a connection can be handed to FastAPI's
    threadpool worker; callers must still confine a connection to one thread
    (the :class:`Database` wrapper does this via thread-locals).
    """
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def bootstrap(conn: sqlite3.Connection) -> None:
    """Create every table/index in :data:`SCHEMA`. Idempotent."""
    conn.executescript(SCHEMA)


class Database:
    """Path-bound SQLite handle with thread-local connections.

    Bootstraps the schema on construction. Pass a ``tmp_path`` DB in tests for
    full isolation; the default points at ``data/config.db``.
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        self._local = threading.local()
        bootstrap(self.connect())

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, opening one on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def query(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connect().execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()
