"""Durable persistence layer for PaperBroker fill history and idempotency keys.

A single SQLite file can host many logical books (identified by ``book_id``).
On broker init, :meth:`PaperBrokerStore.load_fills` replays the stored fills so
the in-memory book is restored to its last state.  Every fill is written via
:meth:`append_fill` before the method returns, so a crash can lose at most the
in-progress fill.

Idempotency
-----------
Direct (non-approval) trades pass a caller-supplied ``idem_key`` to
:meth:`place_order_idempotent`.  The key is inserted atomically (UNIQUE
constraint) before execution; a duplicate key returns the cached result without
re-executing.  Because the keys are stored in the DB they survive restarts and
eliminate double-fire on ``systemd Restart=on-failure``.

Thread safety
-------------
A single ``sqlite3`` connection is opened per ``PaperBrokerStore`` instance with
``isolation_level=None`` (autocommit).  The PaperBroker already holds an
external lock around all mutations; the store relies on that lock rather than
adding its own.  Do NOT share a store across multiple PaperBroker instances.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def _utcnow() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


# ---------------------------------------------------------------------------
# Fill record (lightweight; no dataclass overhead needed for replay)
# ---------------------------------------------------------------------------

class FillRecord:
    """Minimal value object returned by :meth:`PaperBrokerStore.load_fills`."""

    __slots__ = ("fill_seq", "order_id", "symbol", "side", "quantity", "fill_price", "commission")

    def __init__(
        self,
        fill_seq: int,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float,
    ) -> None:
        self.fill_seq = fill_seq
        self.order_id = order_id
        self.symbol = symbol
        self.side = side          # "BUY" | "SELL"
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PaperBrokerStore:
    """SQLite-backed persistence for one or more paper-broker books.

    Pass a ``book_id`` when constructing :class:`~trading_agent.paper_broker.PaperBroker`
    to activate durability::

        store = PaperBrokerStore("data/paper.db")
        broker = PaperBroker(initial_balance=100_000.0, store=store, book_id="test")

    Schema is created idempotently on first connection, so no explicit migration
    step is required when the file is new.
    """

    def __init__(self, db_path: PathLike = "data/paper.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_fills (
                fill_seq    INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id     TEXT    NOT NULL,
                order_id    TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                side        TEXT    NOT NULL,   -- BUY | SELL
                quantity    REAL    NOT NULL,
                fill_price  REAL    NOT NULL,
                commission  REAL    NOT NULL DEFAULT 0.0,
                ts          TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_paper_fills_book
                ON paper_fills(book_id, fill_seq);

            CREATE TABLE IF NOT EXISTS paper_idem_keys (
                book_id     TEXT NOT NULL,
                idem_key    TEXT NOT NULL,
                result_json TEXT,           -- JSON-encoded broker result (may be NULL)
                ts          TEXT NOT NULL,
                PRIMARY KEY (book_id, idem_key)
            );
            """
        )

    # ------------------------------------------------------------------
    # Fill persistence
    # ------------------------------------------------------------------

    def append_fill(
        self,
        book_id: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float,
    ) -> None:
        """Write a fill record.  Called by PaperBroker immediately after _execute_trade."""
        self._conn().execute(
            """
            INSERT INTO paper_fills
                (book_id, order_id, symbol, side, quantity, fill_price, commission, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (book_id, order_id, symbol, side, quantity, fill_price, commission, _utcnow()),
        )

    def load_fills(self, book_id: str) -> list[FillRecord]:
        """Return all fills for *book_id* ordered by fill_seq (ascending)."""
        rows = self._conn().execute(
            """
            SELECT fill_seq, order_id, symbol, side, quantity, fill_price, commission
            FROM paper_fills
            WHERE book_id = ?
            ORDER BY fill_seq ASC
            """,
            (book_id,),
        ).fetchall()
        return [
            FillRecord(
                fill_seq=row["fill_seq"],
                order_id=row["order_id"],
                symbol=row["symbol"],
                side=row["side"],
                quantity=row["quantity"],
                fill_price=row["fill_price"],
                commission=row["commission"],
            )
            for row in rows
        ]

    def fill_count(self, book_id: str) -> int:
        """Return the number of fills stored for *book_id*."""
        row = self._conn().execute(
            "SELECT COUNT(*) AS n FROM paper_fills WHERE book_id = ?", (book_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Idempotency keys
    # ------------------------------------------------------------------

    def has_idem_key(self, book_id: str, idem_key: str) -> dict[str, Any] | None:
        """Return the stored result if *idem_key* already exists, else None."""
        row = self._conn().execute(
            "SELECT result_json FROM paper_idem_keys WHERE book_id=? AND idem_key=?",
            (book_id, idem_key),
        ).fetchone()
        if row is None:
            return None
        raw = row["result_json"]
        return json.loads(raw) if raw else {}

    def add_idem_key(
        self,
        book_id: str,
        idem_key: str,
        result: dict[str, Any] | None,
    ) -> bool:
        """Insert *idem_key*.

        Returns True on success, False if the key was already present (race).
        Raises ``sqlite3.IntegrityError`` only on genuine unexpected DB errors.
        """
        try:
            self._conn().execute(
                """
                INSERT INTO paper_idem_keys (book_id, idem_key, result_json, ts)
                VALUES (?, ?, ?, ?)
                """,
                (
                    book_id,
                    idem_key,
                    json.dumps(result, default=str) if result is not None else None,
                    _utcnow(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate — race condition, key already committed

    def close(self) -> None:
        """Close the thread-local connection (if open)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
