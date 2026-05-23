"""Placeholder mean-reversion strategy.

Per SPEC: 20-period SMA, buy on -2σ deviation, exit on touch of the SMA.
Maintains a rolling window of closes across ``on_data`` calls so it works
with both single-bar (dict) and DataFrame inputs.
"""

from __future__ import annotations

import math
import tomllib
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from ..strategy import Strategy


class MeanReversionStrategy(Strategy):
    """20-SMA / -2σ mean-reversion. Long-only.

    Internal state machine:
        flat → buy when ``close <= SMA - k*σ``  (emit ``LONG``)
        long → sell when ``close >= SMA``       (emit ``SHORT`` to close)
        otherwise                               (emit ``NEUTRAL``)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if params is None:
            if config_path is None:
                raise ValueError("Either config_path or params must be provided")
            params = self._load_toml(Path(config_path))

        strat = params.get("strategy", {})
        self.sma_period: int = int(strat.get("sma_period", 20))
        self.std_multiplier: float = float(strat.get("std_multiplier", 2.0))
        self.position_size: float = float(strat.get("position_size", 1.0))
        self.symbols: list[str] = list(params.get("symbols", []))
        self.timeframe: str = str(params.get("timeframe", "1m"))

        self._closes: deque[float] = deque(maxlen=self.sma_period)
        self._is_long: bool = False

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)

    # --- Strategy ABC -------------------------------------------------------

    def get_symbols(self) -> list[str]:
        return list(self.symbols)

    def get_timeframe(self) -> str:
        return self.timeframe

    def get_params(self) -> dict[str, Any]:
        return {
            "sma_period": self.sma_period,
            "std_multiplier": self.std_multiplier,
            "position_size": self.position_size,
        }

    def on_data(self, bar_or_tick: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        close = self._extract_close(bar_or_tick)
        asset = self._extract_symbol(bar_or_tick)

        self._closes.append(float(close))

        if len(self._closes) < self.sma_period:
            return self._signal(asset, side="NEUTRAL", price=close, sma=None, lower_band=None)

        sma = sum(self._closes) / len(self._closes)
        variance = sum((c - sma) ** 2 for c in self._closes) / len(self._closes)
        std = math.sqrt(variance)
        lower_band = sma - self.std_multiplier * std

        side = "NEUTRAL"
        if not self._is_long and close <= lower_band:
            side = "LONG"
            self._is_long = True
        elif self._is_long and close >= sma:
            side = "SHORT"
            self._is_long = False

        return self._signal(asset, side=side, price=close, sma=sma, lower_band=lower_band)

    # --- Helpers ------------------------------------------------------------

    def _signal(
        self,
        asset: str,
        side: str,
        price: float,
        sma: float | None,
        lower_band: float | None,
    ) -> dict[str, Any]:
        return {
            "asset": asset,
            "side": side,
            "amount": self.position_size,
            "price": price,
            "sma": sma,
            "lower_band": lower_band,
        }

    def _extract_close(self, data: dict[str, Any] | pd.DataFrame) -> float:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Empty DataFrame passed to on_data")
            return float(data["close"].iloc[-1])
        if "close" in data:
            return float(data["close"])
        raise KeyError("Bar/tick must contain a 'close' value")

    def _extract_symbol(self, data: dict[str, Any] | pd.DataFrame) -> str:
        if isinstance(data, dict) and "symbol" in data:
            return str(data["symbol"])
        if self.symbols:
            return self.symbols[0]
        return "unknown"
