from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def get_symbols(self) -> list:
        """Return list of symbols this strategy trades."""
        pass

    @abstractmethod
    def get_timeframe(self) -> str:
        """Return the timeframe this strategy operates on (e.g., '1h', '1d')."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return strategy parameters."""
        pass

    @abstractmethod
    def on_data(self, bar_or_tick: dict | pd.DataFrame) -> dict[str, Any]:
        """Process market data and return trading signals.

        Args:
            bar_or_tick: Market data (bar or tick)

        Returns:
            Dict containing trading signals and metadata
        """
        pass
