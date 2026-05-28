"""Deterministic vol-threshold regime classifier (P3).

Regime is the conditioning key for the pattern KB (P4) and the situation
context block injected into trader decisions. It is computed entirely from
market data (realized vol + calendar proximity) — no model call, no guessing.

Labels (coarsest to finest):
  CALM         — realized vol < low threshold, no imminent event
  ELEVATED     — realized vol between thresholds, or mild event proximity
  EVENT_WINDOW — high-impact calendar event within ``event_horizon_days``
  RISK_OFF     — realized vol >= high threshold (tail regime)

All thresholds are deterministic. Fail-loud if inputs are insufficient.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RegimeLabel(StrEnum):
    CALM = "calm"
    ELEVATED = "elevated"
    EVENT_WINDOW = "event-window"
    RISK_OFF = "risk-off"


# Annualized vol thresholds (these are fraction-based, e.g. 0.15 = 15% annual).
_VOL_CALM = 0.15        # below this → calm
_VOL_ELEVATED = 0.25    # between calm and this → elevated
_VOL_RISK_OFF = 0.40    # at or above this → risk-off


@dataclass(frozen=True)
class RegimeState:
    label: RegimeLabel
    realized_vol_annual: float   # annualized realized vol used to classify
    event_count: int             # how many high-impact events fall in the horizon
    note: str = ""               # human-readable reason (for context block)

    def to_context_lines(self) -> list[str]:
        ev = f", {self.event_count} event(s) in window" if self.event_count else ""
        return [
            f"  Regime: {self.label.value} (vol={self.realized_vol_annual:.1%}{ev})",
            f"  Note: {self.note}" if self.note else "",
        ]


class RegimeClassifier:
    """Classify market regime from recent close prices and calendar events.

    Parameters
    ----------
    vol_calm:     annualized vol below which regime is CALM
    vol_elevated: annualized vol below which regime is ELEVATED (else RISK_OFF)
    vol_risk_off: annualized vol at or above which regime is RISK_OFF
    event_horizon_days: how many calendar days ahead to look for events
    """

    def __init__(
        self,
        *,
        vol_calm: float = _VOL_CALM,
        vol_elevated: float = _VOL_ELEVATED,
        vol_risk_off: float = _VOL_RISK_OFF,
        event_horizon_days: int = 3,
        trading_days_per_year: int = 252,
    ) -> None:
        self._vol_calm = vol_calm
        self._vol_elevated = vol_elevated
        self._vol_risk_off = vol_risk_off
        self._horizon = event_horizon_days
        self._tpy = trading_days_per_year

    def classify(
        self,
        closes: list[float],
        events: list[dict[str, Any]] | None = None,
    ) -> RegimeState:
        """Compute regime from ``closes`` (recent price series) + ``events``.

        ``closes`` must have at least 2 values; fail-loud otherwise.
        ``events`` is a list of calendar event dicts with at least a
        ``days_away`` int (days until the event; 0 = today).
        """
        if len(closes) < 2:
            raise ValueError(
                f"regime classifier needs ≥2 closes, got {len(closes)}"
            )
        vol = self._realized_vol(closes)
        event_list = events or []
        imminent = [
            e for e in event_list
            if isinstance(e.get("days_away"), int) and 0 <= e["days_away"] <= self._horizon
        ]

        label, note = self._label(vol, len(imminent))
        return RegimeState(
            label=label,
            realized_vol_annual=vol,
            event_count=len(imminent),
            note=note,
        )

    def _realized_vol(self, closes: list[float]) -> float:
        """Daily log-return std dev, annualized."""
        log_rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0
        ]
        if len(log_rets) < 1:
            raise ValueError("insufficient non-zero closes to compute vol")
        if len(log_rets) == 1:
            daily_std = abs(log_rets[0])
        else:
            daily_std = statistics.stdev(log_rets)
        return daily_std * math.sqrt(self._tpy)

    def _label(self, vol: float, event_count: int) -> tuple[RegimeLabel, str]:
        if vol >= self._vol_risk_off:
            return RegimeLabel.RISK_OFF, f"vol {vol:.1%} ≥ risk-off threshold"
        if event_count > 0:
            return RegimeLabel.EVENT_WINDOW, f"{event_count} high-impact event(s) imminent"
        if vol < self._vol_calm:
            return RegimeLabel.CALM, f"vol {vol:.1%} below calm threshold"
        return RegimeLabel.ELEVATED, f"vol {vol:.1%} in elevated band"
