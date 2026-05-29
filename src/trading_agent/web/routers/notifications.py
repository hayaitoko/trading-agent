"""Notification center (WS-H): one feed merging requests, alerts, fills, blocks.

Sources, all optional except requests:
* **request** — pending :class:`StockRequest`s (a trader asking to trade a new
  symbol); actionable via the requests router. Always read from ``config.db``.
* **alert** — market moves from a :class:`MarketMoveWatcher` on
  ``app.state.market_watch`` (if wired).
* **fill / block** — recent bench decisions from a :class:`Bench` on
  ``app.state.bench`` (if wired): ``filled`` → fill, ``blocked``/``rejected`` →
  block.

Items match the cockpit ``NOTIFS`` shape (``{id, type, unread, who, t, m, ts}``,
types ``request|alert|fill|block``). Read-state is persisted per user in a small
``notification_reads`` table (created here, additive to WS-0's schema).

``app.state.market_watch`` / ``app.state.bench`` are wiring WS-0/serve attach for
live data; absent them, the feed is just the (real, per-user) stock requests.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...config.db import Database
from ...config.users import current_user, get_db
from ...requests import STATUS_PENDING, RequestStore

if TYPE_CHECKING:
    from ...bench.bench import Bench
    from ..market_watch import MarketMove, MarketMoveWatcher

router = APIRouter(tags=["notifications"])


# --- read-state -------------------------------------------------------------


class NotificationReadStore:
    """Per-user set of notification ids the operator has marked read."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_reads (
                user_id  TEXT NOT NULL,
                notif_id TEXT NOT NULL,
                read_at  REAL NOT NULL,
                PRIMARY KEY (user_id, notif_id)
            )
            """
        )

    def read_ids(self, user_id: str) -> set[str]:
        rows = self._db.query(
            "SELECT notif_id FROM notification_reads WHERE user_id = ?", (user_id,)
        )
        return {r["notif_id"] for r in rows}

    def mark(self, user_id: str, ids: Iterable[str]) -> int:
        now = time.time()
        n = 0
        for notif_id in ids:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO notification_reads (user_id, notif_id, read_at) "
                "VALUES (?, ?, ?)",
                (user_id, notif_id, now),
            )
            n += cur.rowcount or 0
        return n


# --- time helpers -----------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _parse_iso(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return _utcnow()
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _relative(then: datetime, now: datetime) -> str:
    """Compact relative-time label matching the cockpit (``1m``, ``2h``, ``3d``)."""
    secs = (now - then).total_seconds()
    if secs < 0:
        secs = 0
    if secs < 45:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h"
    return f"{int(secs // 86400)}d"


def _verb(action: str) -> str:
    a = action.upper()
    if a == "BUY":
        return "bought"
    if a == "SELL":
        return "sold"
    return action.lower()


# --- feed builder -----------------------------------------------------------


def build_items(
    user_id: str,
    *,
    request_store: RequestStore,
    moves: Sequence[MarketMove] | None = None,
    decisions: list[dict[str, Any]] | None = None,
    read_ids: set[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Merge sources into one cockpit-shaped, newest-first notification list."""
    now = now or _utcnow()
    read_ids = read_ids or set()
    rows: list[tuple[datetime, dict[str, Any]]] = []

    # requests (pending only — allow/decline drops them, matching the mock)
    for req in request_store.list(user_id, status=STATUS_PENDING):
        when = datetime.fromtimestamp(req.created_at, UTC).replace(tzinfo=None)
        rows.append(
            (
                when,
                {
                    "id": req.id,  # == /api/requests/{id} so the UI can allow/decline
                    "type": "request",
                    "who": req.trader_id,
                    "t": f"Wants to start trading {req.symbol}",
                    "m": req.reason or "Asking for your OK before trading a new stock.",
                    "data": {"request_id": req.id, "symbol": req.symbol},
                },
            )
        )

    # market-move alerts
    for mv in moves or []:
        when = _parse_iso(mv.timestamp)
        pct = mv.pct_change * 100.0
        word = "up" if mv.direction == "up" else "down"
        rows.append(
            (
                when,
                {
                    "id": f"alert:{mv.symbol}:{mv.timestamp}",
                    "type": "alert",
                    "who": None,
                    "t": f"{mv.symbol} is {word} {abs(pct):.1f}% today",
                    "m": f"Now {mv.current_price:,.2f}, from session open {mv.reference_price:,.2f}.",
                    "data": {"symbol": mv.symbol, "pct_change": mv.pct_change},
                },
            )
        )

    # fills + blocks from bench decisions
    for dec in decisions or []:
        status = str(dec.get("status", "")).lower()
        when = _parse_iso(str(dec.get("timestamp", "")))
        who = str(dec.get("competitor", ""))
        symbol = str(dec.get("symbol", ""))
        qty = dec.get("quantity", 0)
        if status == "filled":
            rows.append(
                (
                    when,
                    {
                        "id": f"fill:{who}:{symbol}:{dec.get('timestamp')}",
                        "type": "fill",
                        "who": who,
                        "t": f"{who} {_verb(str(dec.get('action', '')))} {qty:g} shares of {symbol}",
                        "m": str(dec.get("reason") or dec.get("detail") or ""),
                        "data": dec,
                    },
                )
            )
        elif status in ("blocked", "rejected"):
            rows.append(
                (
                    when,
                    {
                        "id": f"block:{who}:{symbol}:{dec.get('timestamp')}",
                        "type": "block",
                        "who": who,
                        "t": f"{who}: trade in {symbol} didn't go through",
                        "m": str(dec.get("detail") or dec.get("reason") or "Blocked by your limits."),
                        "data": dec,
                    },
                )
            )

    rows.sort(key=lambda r: r[0], reverse=True)
    items: list[dict[str, Any]] = []
    for when, item in rows:
        item["unread"] = item["id"] not in read_ids
        item["ts"] = _relative(when, now)
        item["timestamp"] = when.isoformat()
        items.append(item)
    return items


# --- routes -----------------------------------------------------------------


def _sources(request: Request) -> tuple[MarketMoveWatcher | None, Bench | None]:
    state = request.app.state
    return getattr(state, "market_watch", None), getattr(state, "bench", None)


def _current_items(request: Request, user_id: str) -> list[dict[str, Any]]:
    db = get_db(request)
    watch, bench = _sources(request)
    moves = list(watch.recent()) if watch is not None else None
    # Gap B (WS-LOOKTOOL-WIRING): the fill/block notifications draw from the agent
    # turn store (where ACT-tool trades land) when one is wired, falling back to the
    # legacy bench decision log. Same decision-row shape either way, so build_items
    # is unchanged.
    decisions = _decisions_for_feed(request, bench)
    return build_items(
        user_id,
        request_store=RequestStore(db),
        moves=moves,
        decisions=decisions,
        read_ids=NotificationReadStore(db).read_ids(user_id),
    )


def _decisions_for_feed(request: Request, bench: Bench | None) -> list[dict[str, Any]] | None:
    """Recent decision rows: agent turn store first, then the bench decision log."""
    store = getattr(request.app.state, "turn_store", None)
    if store is not None:
        try:
            turns = store.recent_all(limit=30)
        except Exception:
            turns = []
        if turns:
            from .bench import turns_to_decision_rows

            return turns_to_decision_rows(turns)
    return bench.recent_decisions() if bench is not None else None


@router.get("/api/notifications")
def notifications(request: Request, user_id: str = Depends(current_user)) -> dict[str, Any]:
    items = _current_items(request, user_id)
    return {
        "generated_at": _utcnow().isoformat(),
        "items": items,
        "unread": sum(1 for it in items if it["unread"]),
    }


class ReadBody(BaseModel):
    ids: list[str] | None = None


@router.post("/api/notifications/read")
def mark_read(
    request: Request,
    body: ReadBody | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Mark the given ids read, or all currently-visible items if none given."""
    db = get_db(request)
    store = NotificationReadStore(db)
    ids = body.ids if body and body.ids is not None else [it["id"] for it in _current_items(request, user_id)]
    marked = store.mark(user_id, ids)
    remaining = sum(1 for it in _current_items(request, user_id) if it["unread"])
    return {"status": "ok", "marked": marked, "unread": remaining}
