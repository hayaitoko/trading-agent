"""``forecast`` — forward 1σ price-cone forecast (gated by SITUATION_FORECAST).

Tool name:      forecast
Args:           symbol (str), horizon=30 (5|10|30)
ToolResult:     ok=True,  data=ForecastCone.to_dict()
                ok=False, error=ToolError(kind="disabled", …) when flag off
Latency tier:   medium
Cost class:     free
Gating flag:    SITUATION_FORECAST (user_settings, default False)

Wraps :func:`~trading_agent.intel.forecast.build_forecast`.  The cone
combines up to three vol components (empirical realized vol, options IV,
prediction-market implied move) into a single ±1σ envelope.  All components
are optional — the tool degrades gracefully to empirical_sigma only when
options IV and prediction-market data are unavailable.

Anti-overconfidence: the mid line is flat (current price, no drift estimate).
This is an *envelope*, never a point-estimate. The trader should treat it as
a range of plausible outcomes, not a directional prediction.

MONEY IS REAL: the tool result contains only float prices and scores.
No account-status strings ("paper", "sim", "demo") appear anywhere.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class ForecastTool(LookToolBase):
    """Forward 1σ price-cone forecast for a symbol.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    settings_store:
        Duck-typed ``get(user_id, key, default)`` — checks SITUATION_FORECAST,
        SITUATION_OPTIONS_IV, and SITUATION_PREDICTION_MARKETS flags.
    history_service:
        Duck-typed HistoryService; provides ``get_bars()``.
    chain_provider:
        Options chain provider; provides ``get_chain()``.
    pm_provider:
        Prediction-markets provider; provides ``event_odds()``.
    spot_prices:
        Current spot prices keyed by symbol (pre-populated from market feed).
    """

    TOOL_META: dict[str, Any] = {
        "name": "forecast",
        "description": (
            "Forward 1σ price-cone forecast for a symbol over 5/10/30 day horizon. "
            "Combines realized vol, options IV, and prediction-market implied move into "
            "an envelope of plausible prices. This is an *envelope*, not a point forecast — "
            "the mid line is flat (no drift estimate). "
            "Enable via SITUATION_FORECAST in trader settings."
        ),
        "args": {
            "symbol": "str (e.g. 'AAPL', 'SPY', 'BTC/USD')",
            "horizon": "5 | 10 | 30 (default 30)",
        },
        "latency": "medium",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        settings_store: Any = None,
        history_service: Any = None,
        chain_provider: Any = None,
        pm_provider: Any = None,
        spot_prices: dict[str, float] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._settings = settings_store
        self._history = history_service
        self._chain = chain_provider
        self._pm = pm_provider
        self._spots: dict[str, float] = dict(spot_prices or {})

    def __call__(
        self,
        symbol: str,
        horizon: int = 30,
    ) -> Any:
        """Return the forward price-cone forecast.

        Parameters
        ----------
        symbol:
            Ticker or instrument symbol (e.g. 'AAPL', 'BTC/USD').
        horizon:
            Forward horizon in trading days: 5, 10, or 30.

        Returns
        -------
        ToolResult
            ok=True, data=ForecastCone.to_dict()

            ok=False, error=ToolError(kind="disabled")
                When SITUATION_FORECAST flag is off.

        Example
        -------
        >>> tool = ForecastTool(trader_id="Alpha")
        >>> result = tool("AAPL", horizon=10)
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        if not self._flag_enabled():
            return self._err(
                "disabled",
                "forecast: enable SITUATION_FORECAST in trader settings to use this tool.",
            )

        # Clamp horizon to valid values.
        if horizon not in (5, 10, 30):
            horizon = min((5, 10, 30), key=lambda h: abs(h - horizon))

        from ...forecast import build_forecast

        symbol = symbol.strip().upper()
        spot = self._spots.get(symbol)

        cone = build_forecast(
            symbol,
            horizon_days=horizon,  # type: ignore[arg-type]
            history_service=self._history,
            chain_provider=self._chain,
            pm_provider=self._pm,
            settings_store=self._settings,
            user_id=self.owner_user_id,
            spot_price=spot,
        )
        return self._ok(cone.to_dict())

    def _flag_enabled(self) -> bool:
        if self._settings is None:
            return False
        try:
            return bool(
                self._settings.get(self.owner_user_id or "", "SITUATION_FORECAST", False)
            )
        except Exception:  # noqa: BLE001
            return False
