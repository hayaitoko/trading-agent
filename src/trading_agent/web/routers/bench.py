"""Bench router (WS-I): the cockpit's core trading surfaces over the live engine.

Reads a live :class:`~trading_agent.bench.bench.Bench` off ``app.state.bench``
(``leaderboard()`` / ``snapshot()`` / ``recent_decisions()``) and creates new
competitors through ``app.state.bench_controller`` (``BenchController.add_model``).
Everything degrades gracefully: when no engine is attached (e.g. plain
``create_cockpit_app`` in unit tests, or a serve process that hasn't wired the
bench yet) the read routes return ``[]`` so the cockpit keeps its mock fallback,
and the create route answers 503.

Response shapes match the cockpit's render functions (``design/cockpit.html`` —
the ``ACCOUNTS`` / ``POSITIONS`` arrays and the leaderboard / activity-log rows):
- accounts / leaderboard rows: ``{name, prov, value, ret, status, cash, pos,
  trades, win, dec, act, sym, rank}``
- positions: ``{sym, name, price, chg, pct, seed, trend, holders[], notes[]}``
- activity: ``{lv, text, ts}`` tri

The engine has no single configurable per-trader starting cash or win-rate
ledger, so the wizard's ``cash``/``style`` are accepted but not honored, and
``win`` is reported as 0 (no realized-P&L tracking yet). See WS-I handoff.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user

if TYPE_CHECKING:
    from ...bench.bench import Bench
    from ...bench.controller import BenchController

router = APIRouter(tags=["bench"])


# --- request bodies ----------------------------------------------------------


class CreateTrader(BaseModel):
    """Add-trader wizard payload. ``cash``/``style`` are accepted for forward
    compatibility but not yet honored by the bench engine (fixed initial balance,
    no per-trader style)."""

    model: str
    name: str | None = None
    cash: float | None = None
    style: str | None = None


# --- app.state plumbing ------------------------------------------------------


def _bench(request: Request) -> Bench | None:
    return getattr(request.app.state, "bench", None)


def _controller(request: Request) -> BenchController | None:
    return getattr(request.app.state, "bench_controller", None)


# --- shaping helpers ---------------------------------------------------------


def _provider_of(model: str) -> str:
    """Provider chip from a model slug (``anthropic/claude-opus-4.7`` -> ``anthropic``)."""
    return model.split("/", 1)[0] if "/" in model else model


def _last_actions(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Newest decision per competitor (``recent_decisions`` is already newest-first)."""
    latest: dict[str, dict[str, Any]] = {}
    for entry in decisions:
        name = str(entry.get("competitor", ""))
        if name and name not in latest:
            latest[name] = entry
    return latest


def _account_row(row: dict[str, Any], latest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map a ``Bench.leaderboard()`` row onto the cockpit's ACCOUNTS card shape."""
    model = str(row.get("model") or row.get("name") or "")
    last = latest.get(str(row.get("name", "")))
    action = str((last or {}).get("action", "")).upper()
    if action in ("BUY", "SELL"):
        act, sym = action, str(last.get("symbol") or "—")  # type: ignore[union-attr]
    else:
        act, sym = "HOLD", "—"
    decisions = int(row.get("decisions", 0) or 0)
    return {
        "name": row.get("name"),
        "prov": _provider_of(model),
        "model": model,
        "value": row.get("account_value", 0.0),
        "ret": row.get("return_pct", 0.0),
        "pnl": row.get("pnl", 0.0),
        "status": "error" if row.get("error") else ("trading" if decisions else "idle"),
        "cash": row.get("cash", 0.0),
        "pos": len(row.get("positions") or []),
        "trades": row.get("trades", 0),
        "win": 0,  # no realized-P&L ledger in the engine yet
        "dec": decisions,
        "act": act,
        "sym": sym,
        "rank": row.get("rank"),
    }


def _accounts(bench: Bench) -> list[dict[str, Any]]:
    latest = _last_actions(bench.recent_decisions())
    return [_account_row(row, latest) for row in bench.leaderboard()]


def _seed(symbol: str) -> int:
    """Stable per-symbol sparkline seed so a position's curve doesn't jump on reload."""
    return (sum(ord(c) for c in symbol) % 90) + 5


def _positions(bench: Bench) -> list[dict[str, Any]]:
    """Aggregate every competitor's holdings into one card per symbol."""
    snap = bench.snapshot()
    prices: dict[str, float] = snap.get("last_prices", {}) or {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in snap.get("leaderboard", []):
        owner = str(row.get("name", "?"))
        for pos in row.get("positions") or []:
            symbol = str(pos.get("symbol"))
            grouped.setdefault(symbol, []).append(
                {
                    "acct": owner,
                    "qty": pos.get("quantity", 0),
                    "avg": pos.get("avg_price", 0.0),
                }
            )
    cards = []
    for symbol, holders in grouped.items():
        price = float(prices.get(symbol, holders[0]["avg"]))
        cards.append(
            {
                "sym": symbol,
                "name": symbol,  # engine has no company-name lookup
                "price": price,
                "chg": 0.0,  # no prior-close reference available
                "pct": 0.0,
                "seed": _seed(symbol),
                "trend": 0,
                "holders": holders,
                "notes": [],
            }
        )
    cards.sort(key=lambda c: str(c["sym"]))
    return cards


_ACTIVITY_LEVEL = {
    "filled": "trade",
    "blocked": "warn",
    "rejected": "warn",
    "error": "warn",
    "hold": "info",
}


def _clock(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ts


def _activity_row(entry: dict[str, Any]) -> dict[str, Any]:
    status = str(entry.get("status", "info"))
    competitor = str(entry.get("competitor", "?"))
    action = str(entry.get("action", "")).strip()
    symbol = str(entry.get("symbol", "")).strip()
    qty = entry.get("quantity", 0)
    qty_str = f"{qty:g}" if isinstance(qty, (int, float)) else str(qty)

    parts = [competitor]
    if action and action != "-":
        parts.append(action)
    if qty:
        parts.append(qty_str)
    if symbol and symbol != "-":
        parts.append(symbol)
    text = " ".join(parts)
    if status != "filled":
        text += f" [{status}]"
    extra = entry.get("detail") or entry.get("reason")
    if extra:
        text += f" — {extra}"
    return {"lv": _ACTIVITY_LEVEL.get(status, "info"), "text": text, "ts": _clock(str(entry.get("timestamp", "")))}


# --- routes ------------------------------------------------------------------


@router.get("/api/accounts")
def accounts(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """Every trader's practice account (the Accounts grid). Empty -> cockpit mock."""
    bench = _bench(request)
    return _accounts(bench) if bench is not None else []


@router.post("/api/accounts")
def create_account(
    body: CreateTrader, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    """Add-trader wizard: register a new competitor on the live bench."""
    controller = _controller(request)
    if controller is None:
        raise HTTPException(status_code=503, detail="bench engine not running")
    model = body.model.strip()
    if not model:
        raise HTTPException(status_code=422, detail="model is required")
    try:
        name = controller.add_model(model, body.name)
    except ValueError as exc:  # duplicate competitor name
        raise HTTPException(status_code=409, detail=str(exc))

    bench = _bench(request)
    created = next(
        (a for a in (_accounts(bench) if bench is not None else []) if a.get("name") == name),
        {"name": name, "prov": _provider_of(model), "model": model},
    )
    return {"created": name, "account": created}


@router.get("/api/leaderboard")
def leaderboard(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """Ranked standings (same row shape as accounts; cockpit recomputes from it)."""
    bench = _bench(request)
    return _accounts(bench) if bench is not None else []


@router.get("/api/positions")
def positions(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """One card per held symbol, with the competitors holding it. Empty -> mock."""
    bench = _bench(request)
    return _positions(bench) if bench is not None else []


@router.get("/api/activity")
def activity(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """Merged decision log across all competitors, newest first. Empty -> mock."""
    bench = _bench(request)
    if bench is None:
        return []
    return [_activity_row(entry) for entry in bench.recent_decisions()]
