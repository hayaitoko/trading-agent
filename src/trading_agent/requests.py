"""Stock-requests + per-trader universe (WS-H).

A trader may only trade the symbols in its **universe** (the tradable-symbols set
``LLMTrader.symbols`` reads). When a trader wants a symbol outside that set it
emits a :class:`StockRequest`; the request surfaces in the notification center
and the operator **allows** (→ symbol joins that trader's universe, request
marked fulfilled) or **declines** (→ universe unchanged, request marked declined).

Two persistent pieces, both keyed per user:

* :class:`UniverseStore` — the per-``(user_id, trader_id)`` tradable-symbols set.
  Its own table (``trader_universe``) is created idempotently here rather than in
  WS-0's bootstrap, so this stays inside the WS-H file set.
* :class:`RequestStore` — CRUD over the ``stock_requests`` table (WS-0 bootstrap).

:class:`RequestService` ties them together and is what the routers + the
bench/trader call. A live bench can pass ``universe_listener`` to be told when a
symbol is allowed, so an in-memory trader's ``symbols`` can be updated too — the
store remains the source of truth.

See ``CONTRACTS.md §Per-user model`` (``stock_requests``) and
``design/handoff/workstreams/H-requests-notes.md``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .config.db import Database

# Request lifecycle. A request starts pending; allow→fulfilled, decline→declined.
STATUS_PENDING = "pending"
STATUS_FULFILLED = "fulfilled"
STATUS_DECLINED = "declined"

# Listener fired on allow with (user_id, trader_id, symbol) for live coordination.
UniverseListener = Callable[[str, str, str], None]


class RequestError(Exception):
    """Bad transition (e.g. allowing a non-pending request)."""


@dataclass
class StockRequest:
    id: str
    user_id: str
    trader_id: str
    symbol: str
    reason: str
    status: str
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


# --- universe ---------------------------------------------------------------


class UniverseStore:
    """Per-``(user_id, trader_id)`` tradable-symbols set.

    Owns its table; ``_ensure_schema`` makes construction idempotent so the store
    works against any :class:`Database` (the cockpit's ``config.db`` or a test's
    tmp db) without touching WS-0's bootstrap.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS trader_universe (
                user_id   TEXT NOT NULL,
                trader_id TEXT NOT NULL,
                symbol    TEXT NOT NULL,
                added_at  REAL NOT NULL,
                PRIMARY KEY (user_id, trader_id, symbol)
            )
            """
        )

    def get(self, user_id: str, trader_id: str) -> list[str]:
        """The trader's tradable symbols, alphabetically."""
        rows = self._db.query(
            "SELECT symbol FROM trader_universe WHERE user_id = ? AND trader_id = ? "
            "ORDER BY symbol",
            (user_id, trader_id),
        )
        return [r["symbol"] for r in rows]

    def contains(self, user_id: str, trader_id: str, symbol: str) -> bool:
        row = self._db.query_one(
            "SELECT 1 FROM trader_universe WHERE user_id = ? AND trader_id = ? AND symbol = ?",
            (user_id, trader_id, _norm_symbol(symbol)),
        )
        return row is not None

    def add(self, user_id: str, trader_id: str, symbol: str) -> bool:
        """Add ``symbol`` to the trader's universe. True if newly added."""
        sym = _norm_symbol(symbol)
        if self.contains(user_id, trader_id, sym):
            return False
        self._db.execute(
            "INSERT OR IGNORE INTO trader_universe (user_id, trader_id, symbol, added_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, trader_id, sym, time.time()),
        )
        return True

    def remove(self, user_id: str, trader_id: str, symbol: str) -> bool:
        cur = self._db.execute(
            "DELETE FROM trader_universe WHERE user_id = ? AND trader_id = ? AND symbol = ?",
            (user_id, trader_id, _norm_symbol(symbol)),
        )
        return (cur.rowcount or 0) > 0

    def set(self, user_id: str, trader_id: str, symbols: list[str]) -> list[str]:
        """Replace the trader's universe with ``symbols`` (used to seed it)."""
        self._db.execute(
            "DELETE FROM trader_universe WHERE user_id = ? AND trader_id = ?",
            (user_id, trader_id),
        )
        for sym in symbols:
            self.add(user_id, trader_id, sym)
        return self.get(user_id, trader_id)


# --- requests ---------------------------------------------------------------


class RequestStore:
    """CRUD over the ``stock_requests`` table (created by WS-0 bootstrap)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, user_id: str, trader_id: str, symbol: str, reason: str = "") -> StockRequest:
        req = StockRequest(
            id=uuid.uuid4().hex,
            user_id=user_id,
            trader_id=trader_id,
            symbol=_norm_symbol(symbol),
            reason=reason,
            status=STATUS_PENDING,
            created_at=time.time(),
        )
        self._db.execute(
            "INSERT INTO stock_requests (id, user_id, trader_id, symbol, reason, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (req.id, user_id, trader_id, req.symbol, reason, req.status, req.created_at),
        )
        return req

    def get(self, user_id: str, request_id: str) -> StockRequest | None:
        row = self._db.query_one(
            "SELECT * FROM stock_requests WHERE id = ? AND user_id = ?",
            (request_id, user_id),
        )
        return self._row(row) if row else None

    def list(self, user_id: str, status: str | None = None) -> list[StockRequest]:
        """A user's requests, newest first; optionally filtered by status."""
        if status is None:
            rows = self._db.query(
                "SELECT * FROM stock_requests WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM stock_requests WHERE user_id = ? AND status = ? "
                "ORDER BY created_at DESC",
                (user_id, status),
            )
        return [self._row(r) for r in rows]

    def set_status(self, user_id: str, request_id: str, status: str) -> None:
        self._db.execute(
            "UPDATE stock_requests SET status = ? WHERE id = ? AND user_id = ?",
            (status, request_id, user_id),
        )

    @staticmethod
    def _row(row: Any) -> StockRequest:
        return StockRequest(
            id=row["id"],
            user_id=row["user_id"],
            trader_id=row["trader_id"] or "",
            symbol=row["symbol"],
            reason=row["reason"] or "",
            status=row["status"],
            created_at=row["created_at"],
        )


class RequestService:
    """The stock-request flow: submit → (notify) → allow/decline.

    ``universe_listener`` is an optional hook a live bench registers so that when
    a symbol is allowed, an in-memory trader's tradable set can be updated too.
    """

    def __init__(self, db: Database, *, universe_listener: UniverseListener | None = None) -> None:
        self.requests = RequestStore(db)
        self.universe = UniverseStore(db)
        self._listener = universe_listener

    def submit(self, user_id: str, trader_id: str, symbol: str, reason: str = "") -> StockRequest:
        """A trader asks to trade ``symbol`` (called by the bench/trader)."""
        if not trader_id:
            raise RequestError("trader_id is required")
        return self.requests.create(user_id, trader_id, symbol, reason)

    def pending(self, user_id: str) -> list[StockRequest]:
        return self.requests.list(user_id, status=STATUS_PENDING)

    def allow(self, user_id: str, request_id: str) -> StockRequest:
        """Allow a pending request: add the symbol to the trader's universe."""
        req = self._require_pending(user_id, request_id)
        self.universe.add(user_id, req.trader_id, req.symbol)
        self.requests.set_status(user_id, request_id, STATUS_FULFILLED)
        if self._listener is not None:
            self._listener(user_id, req.trader_id, req.symbol)
        req.status = STATUS_FULFILLED
        return req

    def decline(self, user_id: str, request_id: str) -> StockRequest:
        """Decline a pending request: universe is left unchanged."""
        req = self._require_pending(user_id, request_id)
        self.requests.set_status(user_id, request_id, STATUS_DECLINED)
        req.status = STATUS_DECLINED
        return req

    def _require_pending(self, user_id: str, request_id: str) -> StockRequest:
        req = self.requests.get(user_id, request_id)
        if req is None:
            raise KeyError(request_id)
        if req.status != STATUS_PENDING:
            raise RequestError(f"request {request_id} is {req.status!r}, not pending")
        return req
