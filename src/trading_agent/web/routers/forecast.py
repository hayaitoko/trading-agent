"""Forecast router (WS-Situation C1): GET /api/forecast.

Returns the forward 1σ price-cone for a symbol over a 5/10/30 day horizon.
Degrades gracefully when providers are disabled or unavailable.

Route:   GET /api/forecast?symbol=AAPL&horizon=30
Auth:    current_user header (standard)
Returns: ForecastCone JSON (see intel/forecast.py ForecastCone.to_dict())

State dependencies (all optional — degrade when absent):
  app.state.history        HistoryService for empirical realized vol
  app.state.bench          Bench for current spot prices
  app.state.settings       SettingsStore for SITUATION_* flags
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ...config.settings_store import SettingsStore
from ...config.users import current_user

router = APIRouter(tags=["forecast"])


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


def _history(request: Request) -> Any:
    return getattr(request.app.state, "history", None)


def _spot_prices(request: Request) -> dict[str, float]:
    """Extract current spot prices from the bench snapshot if available."""
    bench = getattr(request.app.state, "bench", None)
    if bench is None:
        return {}
    try:
        snap = bench.snapshot()
        prices: dict[str, float] = {}
        for sym, bar in (snap.get("prices") or {}).items():
            close = bar.get("close") if isinstance(bar, dict) else getattr(bar, "close", None)
            if close is not None:
                prices[str(sym)] = float(close)
        return prices
    except Exception:  # noqa: BLE001
        return {}


@router.get("/api/forecast")
def get_forecast(
    request: Request,
    symbol: str,
    horizon: int = 30,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Return the forward 1σ price-cone forecast for ``symbol``.

    Degrades gracefully: if history is unavailable, the cone will have no
    empirical_sigma component.  The response always has the ForecastCone shape
    (possibly with null sigmas and an empty points list when all data is absent).

    Query params
    ------------
    symbol:   Ticker or instrument (e.g. AAPL, SPY, BTC/USD). Required.
    horizon:  Forward horizon in trading days: 5, 10, or 30. Default 30.
    """
    from ...intel.forecast import build_forecast

    if horizon not in (5, 10, 30):
        horizon = min((5, 10, 30), key=lambda h: abs(h - horizon))

    settings = _settings(request)
    history = _history(request)
    spots = _spot_prices(request)

    cone = build_forecast(
        symbol.strip().upper(),
        horizon_days=horizon,  # type: ignore[arg-type]
        history_service=history,
        settings_store=settings,
        user_id=user_id,
        spot_price=spots.get(symbol.strip().upper()),
    )
    return cone.to_dict()
