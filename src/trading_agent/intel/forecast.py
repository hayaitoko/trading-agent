"""Forward 1σ price-cone forecast (WS-Situation Track C).

Builds an empirical price-envelope — not a price prediction. The cone shows
where price *could* be in 5/10/30 days at ±1σ given available vol estimates.
The mid line is flat (current price; no drift assumption) because drift
forecasting is a separate and much harder problem.

Anti-overconfidence contract
----------------------------
This is an *envelope* derived from recent realized vol and, when available,
implied vol and prediction-market implied move. It is NOT a point forecast.
Nothing in this module should be interpreted as a directional opinion on where
price will go.

Three components (all optional — cone degrades gracefully):
  empirical_sigma   Annualized realized vol from 30D closing prices.
  iv_sigma          Annualized implied vol from the nearest near-the-money
                    options contract (iv * sqrt(252/252) = iv; stored annualized
                    so the horizon scaling below is uniform).
  pm_implied_move   Fraction of current price implied by a matched prediction-
                    market event (e.g. a "price above $X by date" market).
                    Only non-None for symbols that have matching PM events.

Combined sigma: max of all available annualized vol estimates (conservative —
the widest cone wins when multiple estimates are available).

Cone geometry (log-normal):
  t_years = t / 252
  band    = combined_sigma * sqrt(t_years)
  hi(t)   = current_price * exp(+band)
  lo(t)   = current_price * exp(-band)
  mid(t)  = current_price                 (flat — no drift)

MONEY IS REAL: This module returns float scores and prices only. No string
references to paper/sim/demo/account-type appear anywhere in the output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConePoint:
    """One point on the forecast cone.

    Attributes
    ----------
    t:   Day offset from today (0 = today, 5 = 5 days out, etc.).
    lo:  Lower 1σ price (exp(-band) from current).
    mid: Current price repeated flat (no drift estimate).
    hi:  Upper 1σ price (exp(+band) from current).
    """

    t: int
    lo: float
    mid: float
    hi: float


@dataclass(frozen=True)
class ForecastCone:
    """Forward 1σ price envelope for a symbol.

    This is an *envelope*, never a point estimate. The mid line is flat
    (current price) to make the anti-overconfidence framing explicit.

    Attributes
    ----------
    symbol:            The ticker / symbol this cone is for.
    horizon_days:      The requested horizon (5, 10, or 30 days).
    current_price:     Spot price used as the cone origin. None if unavailable.
    empirical_sigma:   Annualized realized vol from 30D history. None if
                       history unavailable.
    iv_sigma:          Annualized implied vol from options chain. None if no
                       options data or SITUATION_OPTIONS_IV flag is off.
    pm_implied_move:   Fraction of current price implied by a matched PM event.
                       None when no matching events are found.
    combined_sigma:    Max of available sigmas; the single input to cone math.
                       None only when ALL components are absent (degenerate cone).
    points:            Day-by-day cone points from t=0 to t=horizon_days.
                       Empty list when combined_sigma is None.
    components_used:   Names of sigma components that contributed (for display).
    """

    symbol: str
    horizon_days: int
    current_price: float | None
    empirical_sigma: float | None
    iv_sigma: float | None
    pm_implied_move: float | None
    combined_sigma: float | None
    points: list[ConePoint] = field(default_factory=list)
    components_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon_days": self.horizon_days,
            "current_price": self.current_price,
            "empirical_sigma": self.empirical_sigma,
            "iv_sigma": self.iv_sigma,
            "pm_implied_move": self.pm_implied_move,
            "combined_sigma": self.combined_sigma,
            "components_used": list(self.components_used),
            "points": [
                {"t": p.t, "lo": p.lo, "mid": p.mid, "hi": p.hi}
                for p in self.points
            ],
        }


# ---------------------------------------------------------------------------
# Cone builder
# ---------------------------------------------------------------------------


def build_forecast(
    symbol: str,
    *,
    horizon_days: Literal[5, 10, 30] = 30,
    history_service: Any = None,
    chain_provider: Any = None,
    pm_provider: Any = None,
    settings_store: Any = None,
    user_id: str | None = None,
    spot_price: float | None = None,
) -> ForecastCone:
    """Build a forward 1σ price envelope for ``symbol``.

    All provider parameters are optional; missing providers cause the
    corresponding component to be omitted from the cone.  At least one
    component must be available for the cone to be non-empty.

    Parameters
    ----------
    symbol:          Ticker / instrument symbol (e.g. 'AAPL', 'BTC/USD').
    horizon_days:    5, 10, or 30 — the requested forward horizon.
    history_service: Duck-typed HistoryService; provides ``get_bars()``.
    chain_provider:  Options chain provider; provides ``get_chain()``.
    pm_provider:     Prediction-markets provider; provides ``event_odds()``.
    settings_store:  Duck-typed ``get(user_id, key, default)`` for flag checks.
    user_id:         Owner user ID for flag checks.
    spot_price:      Current spot price, if already known. When None the
                     function attempts to derive it from the last history bar.

    Returns
    -------
    ForecastCone
        A filled cone when at least one sigma component is available.
        Degenerate (empty points list) when all components are absent.
    """
    # Step 1: resolve current price.
    current_price = spot_price
    if current_price is None:
        current_price = _get_spot_from_history(history_service, symbol)

    # Step 2: compute sigma components.
    empirical_sigma: float | None = None
    iv_sigma: float | None = None
    pm_implied_move: float | None = None
    components_used: list[str] = []

    empirical_sigma = _compute_empirical_sigma(history_service, symbol)
    if empirical_sigma is not None:
        components_used.append("empirical")

    if _flag_on(settings_store, user_id, "SITUATION_OPTIONS_IV"):
        iv_sigma = _compute_iv_sigma(chain_provider, symbol, horizon_days, current_price)
        if iv_sigma is not None:
            components_used.append("iv")

    if _flag_on(settings_store, user_id, "SITUATION_PREDICTION_MARKETS"):
        pm_implied_move = _compute_pm_implied_move(pm_provider, symbol)
        if pm_implied_move is not None:
            components_used.append("pm")

    # Step 3: combine sigmas (max = widest / most conservative cone).
    candidates: list[float] = []
    if empirical_sigma is not None:
        candidates.append(empirical_sigma)
    if iv_sigma is not None:
        candidates.append(iv_sigma)
    # pm_implied_move is a fractional price move, not an annualized vol;
    # convert to an approximate annualized vol equivalent: move / sqrt(horizon/252).
    if pm_implied_move is not None and horizon_days > 0:
        pm_vol_equiv = pm_implied_move / math.sqrt(horizon_days / 252.0)
        candidates.append(pm_vol_equiv)

    combined_sigma = max(candidates) if candidates else None

    # Step 4: build cone points.
    points: list[ConePoint] = []
    if combined_sigma is not None and current_price is not None and current_price > 0:
        for t in range(0, horizon_days + 1):
            t_years = t / 252.0
            band = combined_sigma * math.sqrt(t_years) if t > 0 else 0.0
            points.append(ConePoint(
                t=t,
                lo=round(current_price * math.exp(-band), 4),
                mid=round(current_price, 4),
                hi=round(current_price * math.exp(band), 4),
            ))

    return ForecastCone(
        symbol=symbol,
        horizon_days=horizon_days,
        current_price=current_price,
        empirical_sigma=empirical_sigma,
        iv_sigma=iv_sigma,
        pm_implied_move=pm_implied_move,
        combined_sigma=combined_sigma,
        points=points,
        components_used=components_used,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_spot_from_history(history_service: Any, symbol: str) -> float | None:
    """Extract most recent close price from the history service."""
    if history_service is None:
        return None
    try:
        bars = history_service.get_bars(symbol, "1D", 2)
        if bars:
            return float(bars[-1].close)
    except Exception:  # noqa: BLE001
        pass
    return None


def _compute_empirical_sigma(history_service: Any, symbol: str) -> float | None:
    """Annualized realized vol from 30D close prices via log-returns."""
    if history_service is None:
        return None
    try:
        bars = history_service.get_bars(symbol, "1D", 31)
        closes = [float(b.close) for b in bars if getattr(b, "close", None)]
        if len(closes) < 5:
            return None
        log_rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
        ]
        n = len(log_rets)
        mean = sum(log_rets) / n
        variance = sum((r - mean) ** 2 for r in log_rets) / max(n - 1, 1)
        daily_sigma = math.sqrt(variance)
        return round(daily_sigma * math.sqrt(252.0), 6)
    except Exception:  # noqa: BLE001
        return None


def _compute_iv_sigma(
    chain_provider: Any,
    symbol: str,
    horizon_days: int,
    spot: float | None,
) -> float | None:
    """Weighted average implied vol from near-the-money option contracts."""
    if chain_provider is None:
        return None
    try:
        quotes = chain_provider.get_chain(symbol, None)
        if not quotes:
            return None
        # Prefer contracts with IV data.
        quotes_with_iv = [q for q in quotes if getattr(q, "implied_vol", None) is not None]
        if not quotes_with_iv:
            return None
        # Filter near-the-money if spot available.
        if spot is not None and spot > 0:
            band = 0.20
            quotes_with_iv = [
                q for q in quotes_with_iv
                if spot * (1 - band) <= q.contract.strike <= spot * (1 + band)
            ] or quotes_with_iv
        ivs = [float(q.implied_vol) for q in quotes_with_iv if q.implied_vol > 0]
        if not ivs:
            return None
        # Simple mean of near-money IVs (already annualized from Alpaca).
        return round(sum(ivs) / len(ivs), 6)
    except Exception:  # noqa: BLE001
        return None


def _compute_pm_implied_move(pm_provider: Any, symbol: str) -> float | None:
    """Fractional implied price move from matched prediction-market events.

    Looks for events in the 'economics' and 'crypto' categories where the
    event title contains the symbol.  For a matched event, estimates the
    implied move as the probability-weighted distance between the two main
    outcome prices (a rough binary-option analogy).  Returns None when no
    matching events are found.

    This is deliberately conservative: the PM implied move is only used as
    an additional width constraint, not a directional signal.
    """
    if pm_provider is None:
        return None
    try:
        # Search across relevant categories for ticker mentions.
        ticker_upper = symbol.split("/")[0].upper()
        for category in ("economics", "crypto", "politics"):
            events = pm_provider.event_odds(category, query=ticker_upper)
            for ev in events:
                title = str(getattr(ev, "title", "")).upper()
                if ticker_upper in title:
                    prices = list(getattr(ev, "prices", []))
                    if len(prices) >= 2:
                        # Probability-weighted binary distance.
                        p_yes = float(prices[0])
                        p_no = float(prices[1])
                        implied = abs(p_yes - p_no)
                        if 0 < implied < 1:
                            return round(implied, 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _flag_on(settings_store: Any, user_id: str | None, key: str) -> bool:
    if settings_store is None:
        return False
    try:
        return bool(settings_store.get(user_id or "", key, False))
    except Exception:  # noqa: BLE001
        return False
