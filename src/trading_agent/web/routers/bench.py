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

The add-trader wizard's ``cash``/``style`` are honored: ``cash`` funds the new
competitor's paper book and ``style`` is folded into its trader prompt. ``win``
is the realized win rate (% of closed trades that booked a gain), computed from
the bench's realized-P&L ledger.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user
from ...prompts.personas import get_persona_mandate

if TYPE_CHECKING:
    from ...bench.bench import Bench
    from ...bench.controller import BenchController

router = APIRouter(tags=["bench"])


# --- request bodies ----------------------------------------------------------


class CreateTrader(BaseModel):
    """Add-trader wizard payload. ``cash`` sets the competitor's starting paper
    balance and ``style`` is folded into its trader prompt (both honored).

    ``digest_mode`` (optional bool, default False): when True the trader is
    created in DIGEST mode (analyst-digest tier); when False/absent the trader
    runs in PULL mode (current default behaviour, unchanged).
    """

    model: str
    name: str | None = None
    cash: float | None = None
    style: str | None = None
    digest_mode: bool = False


# --- app.state plumbing ------------------------------------------------------


def _bench(request: Request) -> Bench | None:
    return getattr(request.app.state, "bench", None)


def _controller(request: Request) -> BenchController | None:
    return getattr(request.app.state, "bench_controller", None)


def _turn_store(request: Request) -> Any:
    return getattr(request.app.state, "turn_store", None)


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
    wins = int(row.get("wins", 0) or 0)
    losses = int(row.get("losses", 0) or 0)
    closed = wins + losses
    return {
        "name": row.get("name"),
        "prov": _provider_of(model),
        "model": model,
        "value": row.get("account_value", 0.0),
        "ret": row.get("return_pct", 0.0),
        "pnl": row.get("pnl", 0.0),
        "realized": row.get("realized_pnl", 0.0),
        "status": "error" if row.get("error") else ("trading" if decisions else "idle"),
        "cash": row.get("cash", 0.0),
        "pos": len(row.get("positions") or []),
        "trades": row.get("trades", 0),
        "win": round(wins / closed * 100) if closed else 0,  # realized win rate %
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


# --- turn-store → decision-row adapter (Gap B, WS-LOOKTOOL-WIRING) -----------

# Terminal actions that represent a real fill (vs. a hold/pass housekeeping turn).
_TRADE_TERMINALS = {"trade", "trade_batch", "confirm_trade"}


def turns_to_decision_rows(turns: list[Any]) -> list[dict[str, Any]]:
    """Adapt agent ``TurnRecord``s into the legacy decision-row dicts.

    Under the agent model, trades settle via ACT tools so ``Bench.recent_decisions()``
    stays empty.  Agent activity lives in the turn store instead.  This maps each
    completed turn's terminal action onto the exact dict shape both ``/api/activity``
    (``_activity_row``) and the notification feed (``notifications.build_items``)
    already consume — ``{timestamp, competitor, symbol, action, quantity, status,
    reason, detail}`` — so neither contract changes.

    Mapping:
      - ``trade`` / ``confirm_trade`` → action=BUY/SELL (from args.side), symbol +
        quantity from args, status="filled".
      - ``trade_batch`` → one row per item (each with its own symbol/side/qty).
      - ``hold`` / ``pass`` / ``done_for_day`` → action label, status="hold".

    MONEY IS REAL: uses ``TurnRecord``'s trader-safe fields only (no ``book_type``);
    nothing here discloses paper/sim status.
    """
    rows: list[dict[str, Any]] = []
    for rec in turns:
        ts = _rec_ts(rec)
        who = str(getattr(rec, "trader_id", ""))
        action = str(getattr(rec, "final_action", "") or "")
        args = getattr(rec, "final_action_args", {}) or {}
        wake = str(getattr(rec, "wake_reason", "") or "")
        if action == "trade_batch":
            for item in args.get("trades", []) or []:
                rows.append(_trade_row(ts, who, item, wake))
        elif action in ("trade", "confirm_trade"):
            rows.append(_trade_row(ts, who, args, wake))
        else:
            # hold / pass / done_for_day / abandon_trade / interrupted → housekeeping.
            reason = str(args.get("reason", "") or "")
            rows.append(
                {
                    "timestamp": ts,
                    "competitor": who,
                    "symbol": "",
                    "action": action or "hold",
                    "quantity": 0,
                    "status": "hold",
                    "reason": reason,
                    "detail": wake,
                }
            )
    return rows


def _trade_row(ts: str, who: str, args: dict[str, Any], wake: str) -> dict[str, Any]:
    side = str(args.get("side", "") or "").upper()
    action = "BUY" if side == "BUY" else ("SELL" if side == "SELL" else side or "TRADE")
    try:
        qty = float(args.get("qty", 0) or 0)
    except (TypeError, ValueError):
        qty = 0.0
    return {
        "timestamp": ts,
        "competitor": who,
        "symbol": str(args.get("symbol", "") or ""),
        "action": action,
        "quantity": qty,
        "status": "filled",
        "reason": "",
        "detail": wake,
    }


def _rec_ts(rec: Any) -> str:
    """ISO timestamp string from a TurnRecord (ended_at preferred, else started_at)."""
    dt = getattr(rec, "ended_at", None) or getattr(rec, "started_at", None)
    try:
        return dt.isoformat() if dt is not None else ""
    except Exception:
        return str(dt) if dt is not None else ""


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
    # Resolve persona id → full mandate string so the UI can pass the short id
    # (e.g. "animal_spirits") and the trader gets the full system-prompt mandate.
    # If style is not a known persona id it is used verbatim (free-text mandates
    # are still valid — the persona registry is additive, not exclusive).
    style = body.style
    if style is not None:
        resolved = get_persona_mandate(style.strip())
        if resolved is not None:
            style = resolved
    # Build intelligence_flags from the request: digest_mode goes in as an
    # explicit per-trader override (resolution order: this flag → per-user
    # setting → default False, handled inside controller.add_model).
    intelligence_flags: dict[str, bool] | None = None
    if body.digest_mode:
        intelligence_flags = {"digest_mode": True}
    try:
        name = controller.add_model(
            model, body.name, cash=body.cash, style=style,
            intelligence_flags=intelligence_flags,
        )
    except ValueError as exc:  # duplicate competitor name
        raise HTTPException(status_code=409, detail=str(exc))

    bench = _bench(request)
    created = next(
        (a for a in (_accounts(bench) if bench is not None else []) if a.get("name") == name),
        {"name": name, "prov": _provider_of(model), "model": model},
    )
    return {"created": name, "account": created}


@router.delete("/api/accounts/{name:path}")
def remove_account(
    name: str, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    """Remove a trader from the live bench (drops its competitor + unschedules it).

    ``{name:path}`` so slug-style names like ``deepseek/deepseek-v4-flash`` match.
    """
    controller = _controller(request)
    if controller is None:
        raise HTTPException(status_code=503, detail="bench engine not running")
    bench = _bench(request)
    existing = {a.get("name") for a in (_accounts(bench) if bench is not None else [])}
    if name not in existing:
        raise HTTPException(status_code=404, detail=f"no trader named {name!r}")
    controller.remove(name)
    return {"removed": name}


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
    """Merged activity log across all competitors, newest first. Empty -> mock.

    Gap B (WS-LOOKTOOL-WIRING): under the agent model trades settle via ACT tools,
    so ``Bench.recent_decisions()`` stays empty.  Agent turn activity lives in the
    turn store, so the feed reads the most recent turns (terminal action + symbol/qty
    for trade* terminals, hold/pass/done_for_day otherwise) and adapts them onto the
    unchanged ``{lv, text, ts}`` row contract the cockpit activity tile renders.
    Falls back to ``recent_decisions()`` when the turn store is absent (legacy / tests).
    """
    decisions = _activity_decisions(request)
    return [_activity_row(entry) for entry in decisions]


def _activity_decisions(request: Request) -> list[dict[str, Any]]:
    """Decision rows for the activity/notification feeds: turn store first, then bench.

    Returns the legacy decision-row dict shape either way.  Prefers agent turn
    activity (the live source under the agent model); falls back to the bench's
    decision log when no turn store is wired or it holds no completed turns yet.
    """
    store = _turn_store(request)
    if store is not None:
        try:
            turns = store.recent_all(limit=30)
        except Exception:
            turns = []
        if turns:
            return turns_to_decision_rows(turns)
    bench = _bench(request)
    if bench is None:
        return []
    return list(bench.recent_decisions())


# --- Dev-only endpoint (WS-COCKPIT-REAL-DATA R1) ----------------------------
# Only registered at HTTP-routing time if TRADING_AGENT_DEV_TRIGGER=1.
# When the env var is absent the route does not exist — 404 naturally.
# Registration happens at module import: build_cockpit calls create_cockpit_app
# which calls app.include_router(bench_router.router); the conditional below
# attaches (or not) the route onto `router` before include_router runs.
# Re-import after the env changes would not retroactively add the route, which
# is the intended behaviour (server restarts pick up the new env).

if os.environ.get("TRADING_AGENT_DEV_TRIGGER"):

    class _FireTurnBody(BaseModel):
        trader: str | None = None

    @router.post("/api/dev/fire-turn")
    def dev_fire_turn(
        request: Request,
        trader: str | None = None,
        body: _FireTurnBody | None = None,
        user_id: str = Depends(current_user),
    ) -> dict[str, Any]:
        """Dev-only: manually fire one decide() cycle for a named trader.

        Gated by ``TRADING_AGENT_DEV_TRIGGER=1`` env var at process start.
        Auth-required: same ``current_user`` dependency as every cockpit route.
        Useful for smoke-testing trader turns outside the cadence/schedule window.

        ``trader`` name comes from the query string or JSON body.  When the
        named trader is not found the full cadence round fires (all traders).
        503 when the bench engine is not running.
        """
        controller = _controller(request)
        if controller is None:
            raise HTTPException(status_code=503, detail="bench engine not running")

        # Prefer query-string trader; fall back to JSON body.
        name = trader or (body.trader if body else None)

        try:
            if name:
                controller.fire_trader(name)
                return {"fired": name, "mode": "single"}
            else:
                controller.tick_now()
                return {"fired": "__all__", "mode": "all"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"fire-turn failed: {exc}") from exc
