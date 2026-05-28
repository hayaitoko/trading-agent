"""Movers router: top gainers/losers across the bench's tracked universe.

The frontend currently derives "movers" from open positions, which only
surfaces what each trader already holds. This endpoint widens the lens to
every symbol the bench is tracking and ranks by absolute day-over-day
change so a fresh mover that nobody is in yet still shows up.

Universe sourcing (in order):
  1. ``?symbols=A,B,C`` (caller-supplied — caps at 50)
  2. ``app.state.bench.snapshot()["symbols"]`` (the actively-traded universe)

Per symbol the change is derived from :class:`HistoryService` 1D bars
(prev close → latest close). The latest *traded* price (preferred over the
close when available) comes from ``bench.snapshot()["last_prices"]``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.users import current_user

router = APIRouter(tags=["movers"])

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50
_MAX_UNIVERSE = 50
_VALID_DIRECTIONS = ("all", "up", "down")


def _clean_symbol(sym: str) -> str | None:
    s = (sym or "").strip().upper()
    if not s:
        return None
    if not s.replace(".", "").replace("-", "").isalnum() or len(s) > 12:
        return None
    return s


def _parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        sym = _clean_symbol(chunk)
        if sym is None or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    if len(out) > _MAX_UNIVERSE:
        raise HTTPException(
            status_code=400, detail=f"too many symbols (max {_MAX_UNIVERSE})"
        )
    return out


def _bench_snapshot(request: Request) -> dict[str, Any] | None:
    bench = getattr(request.app.state, "bench", None)
    if bench is None:
        return None
    try:
        return bench.snapshot()  # type: ignore[no-any-return]
    except Exception:
        return None


def _history(request: Request) -> Any:
    return getattr(request.app.state, "history", None)


def _prev_and_latest(
    history: Any, symbol: str
) -> tuple[float | None, float | None]:
    """Last two daily closes for ``symbol`` (prev, latest) — None if absent."""
    if history is None:
        return (None, None)
    try:
        bars = history.bars(symbol, "1D", 2)
    except Exception:
        return (None, None)
    if not bars:
        return (None, None)
    latest = float(bars[-1].close)
    prev = float(bars[-2].close) if len(bars) >= 2 else None
    return (prev, latest)


@router.get("/api/movers")
def movers(
    request: Request,
    limit: int = _DEFAULT_LIMIT,
    direction: str = "all",
    symbols: str = "",
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Top gainers/losers across the bench universe (sorted by |Δ%|).

    Params:
      * ``limit``     1..50, default 10 (rows returned).
      * ``direction`` ``all`` | ``up`` | ``down`` — filter the sign of the move.
      * ``symbols``   optional ``A,B,C`` override of the universe.

    Return: ``{"direction": ..., "movers": [{symbol, price, change_pct}, ...]}``.
    Symbols with no derivable change_pct are dropped (the tile wants real moves).
    Empty list (200) when the universe or history sources are missing.
    """
    if limit < 1 or limit > _MAX_LIMIT:
        raise HTTPException(
            status_code=400, detail=f"limit must be 1..{_MAX_LIMIT} (got {limit})"
        )
    dir_norm = (direction or "all").strip().lower()
    if dir_norm not in _VALID_DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"direction must be one of {_VALID_DIRECTIONS} (got {direction!r})",
        )

    snap = _bench_snapshot(request)
    bench_prices: dict[str, float] = (
        {str(k).upper(): float(v) for k, v in (snap.get("last_prices") or {}).items()}
        if snap is not None
        else {}
    )

    if symbols.strip():
        universe = _parse_symbols(symbols)
    elif snap is not None:
        universe = [s for s in (_clean_symbol(s) for s in snap.get("symbols") or []) if s]
    else:
        universe = []

    history = _history(request)
    rows: list[dict[str, Any]] = []
    for sym in universe:
        prev_close, latest_close = _prev_and_latest(history, sym)
        # Price: prefer bench's live tick over the daily close (more current).
        price = bench_prices.get(sym, latest_close)
        if price is None or prev_close is None or prev_close <= 0:
            continue
        change_pct = (price - prev_close) / prev_close * 100.0
        if dir_norm == "up" and change_pct <= 0:
            continue
        if dir_norm == "down" and change_pct >= 0:
            continue
        rows.append({"symbol": sym, "price": price, "change_pct": change_pct})

    rows.sort(key=lambda r: abs(r["change_pct"]), reverse=True)
    return {"direction": dir_norm, "movers": rows[:limit]}
