"""Quotes router: batched latest-quote endpoint for the cockpit's watchlist tile.

Today the frontend polls ``/api/history/{symbol}`` once per watchlist symbol
every ~45 s to fake a "live" price. This endpoint collapses that into a single
request that joins:

* the bench's already-tracked last prices (``Bench.snapshot()["last_prices"]``
  — the thread-safe public accessor; bench mutates ``_last_prices`` under its
  own ``_lock`` and ``snapshot`` snapshots that into a plain ``dict``), and
* :class:`HistoryService` 1D bars for the previous close (so the
  ``change_pct`` field can be filled in even when the price came from the
  bench).

Graceful-empty if neither source is attached — never invents numbers (same
principle as the rest of the cockpit data surface).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.users import current_user

router = APIRouter(tags=["quotes"])

_MAX_SYMBOLS = 50


def _clean_symbols(raw: str) -> list[str]:
    """Parse a ``?symbols=A,B,C`` query into a deduped, upper-cased list."""
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="symbols parameter is required")
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        sym = chunk.strip().upper()
        if not sym:
            continue
        if not sym.replace(".", "").replace("-", "").isalnum() or len(sym) > 12:
            raise HTTPException(status_code=400, detail=f"invalid symbol: {sym!r}")
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    if not out:
        raise HTTPException(status_code=400, detail="symbols parameter is required")
    if len(out) > _MAX_SYMBOLS:
        raise HTTPException(
            status_code=400, detail=f"too many symbols (max {_MAX_SYMBOLS})"
        )
    return out


def _bench_snapshot(request: Request) -> dict[str, Any] | None:
    """Return ``bench.snapshot()`` if a bench is attached, else ``None``.

    ``snapshot()`` is the thread-safe public read on the bench: it returns a
    plain dict (``last_prices`` is already copied), so the router can index it
    freely without holding ``Bench._lock``.
    """
    bench = getattr(request.app.state, "bench", None)
    if bench is None:
        return None
    try:
        return bench.snapshot()  # type: ignore[no-any-return]
    except Exception:
        return None


def _history(request: Request) -> Any:
    """``HistoryService`` if attached, else ``None`` (no fail-loud — quotes degrade)."""
    return getattr(request.app.state, "history", None)


def _prev_and_latest(history: Any, symbol: str) -> tuple[float | None, float | None, str | None]:
    """Two most-recent daily closes for ``symbol`` (prev, latest, latest_ts).

    Falls back to ``(None, None, None)`` on any provider error — change_pct
    simply omits when we can't derive it.
    """
    if history is None:
        return (None, None, None)
    try:
        bars = history.bars(symbol, "1D", 2)
    except Exception:
        return (None, None, None)
    if not bars:
        return (None, None, None)
    latest = bars[-1]
    prev = bars[-2] if len(bars) >= 2 else None
    return (
        float(prev.close) if prev is not None else None,
        float(latest.close),
        str(latest.timestamp),
    )


@router.get("/api/quotes")
def quotes(
    request: Request,
    symbols: str = "",
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Batched latest quote for each ``?symbols=AAPL,MSFT,...``.

    Result: ``{"quotes": [{symbol, price, change_pct, ts}, ...]}``. Each entry
    omits ``price``/``change_pct`` when the underlying source has nothing for
    that symbol — the frontend already renders dashes for nulls. Returns an
    empty list (200) when no data sources are attached at all.
    """
    requested = _clean_symbols(symbols)
    snap = _bench_snapshot(request)
    history = _history(request)
    bench_prices: dict[str, float] = (
        {str(k).upper(): float(v) for k, v in (snap.get("last_prices") or {}).items()}
        if snap is not None
        else {}
    )
    bench_ts = snap.get("generated_at") if snap is not None else None

    out: list[dict[str, Any]] = []
    for sym in requested:
        prev_close, hist_latest, hist_ts = _prev_and_latest(history, sym)
        # Price: prefer the bench's live tick; fall back to the last daily close.
        price: float | None
        ts: str | None
        if sym in bench_prices:
            price = bench_prices[sym]
            ts = bench_ts
        elif hist_latest is not None:
            price = hist_latest
            ts = hist_ts
        else:
            price = None
            ts = None
        # Change pct: needs a prev close + a price.
        change_pct: float | None
        if price is not None and prev_close is not None and prev_close > 0:
            change_pct = (price - prev_close) / prev_close * 100.0
        else:
            change_pct = None
        out.append(
            {"symbol": sym, "price": price, "change_pct": change_pct, "ts": ts}
        )
    return {"quotes": out}
