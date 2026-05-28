"""``situation`` — current market situation block.

Tool name:      situation
Args:           (none)
ToolResult:     ok=True, data={"regime": …, "social": {…}, "calendar_events": […]}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (degrades gracefully when situation layer absent)
Example use:    situation() to check the current market regime before deciding.

data.regime includes: label ("calm"|"elevated"|"event-window"|"risk-off"),
    realized_vol_annual, event_count, note.
data.social: per-symbol metrics from the social aggregator (WS-C).
data.calendar_events: upcoming high-impact events within the horizon.

Wraps :class:`~trading_agent.situation.regime.RegimeClassifier` (P3) and
:class:`~trading_agent.situation.social.SocialAggregator` (P3).
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class SituationTool(LookToolBase):
    """Current market regime, social metrics, and calendar events.

    Parameters
    ----------
    regime_classifier:
        Duck-typed: ``classify(closes, events) -> RegimeState``.  ``None`` → tool
        returns a graceful "unavailable" result.
    recent_closes:
        A list of recent close prices (any symbols) used for regime computation.
        Injected by the AgentTrader wiring layer on each turn from the price buffer.
    calendar_events:
        List of ``{days_away: int, event: str}`` dicts for upcoming events.
    social_aggregator:
        Duck-typed: ``aggregate(items, ticker=None) -> SocialMetrics``.
    social_items:
        Raw social items passed to the aggregator.
    symbols:
        The trader's tradable universe — used to generate per-symbol social metrics.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "situation",
        "description": (
            "Current market situation: regime label, realized vol, social metrics "
            "per symbol, and upcoming calendar events."
        ),
        "args": {},
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        regime_classifier: Any = None,
        recent_closes: list[float] | None = None,
        calendar_events: list[dict[str, Any]] | None = None,
        social_aggregator: Any = None,
        social_items: list[Any] | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._classifier = regime_classifier
        self._closes: list[float] = list(recent_closes or [])
        self._calendar: list[dict[str, Any]] = list(calendar_events or [])
        self._social_agg = social_aggregator
        self._social_items: list[Any] = list(social_items or [])
        self._symbols: list[str] = list(symbols or [])

    def __call__(self) -> Any:
        """Return the current market situation block.

        Returns
        -------
        ToolResult
            ok=True, data={"regime": {…}, "social": {…}, "calendar_events": […]}

        Example
        -------
        >>> tool = SituationTool(trader_id="Alpha")
        >>> result = tool()
        >>> result.ok
        True
        >>> result.data["regime"] is None
        True
        """
        regime_data = self._get_regime()
        social_data = self._get_social()
        return self._ok(
            {
                "regime": regime_data,
                "social": social_data,
                "calendar_events": list(self._calendar),
            }
        )

    def _get_regime(self) -> dict[str, Any] | None:
        clf = self._classifier
        if clf is None or len(self._closes) < 2:
            return None
        try:
            state = clf.classify(self._closes, self._calendar)
            return {
                "label": str(state.label),
                "realized_vol_annual": float(state.realized_vol_annual),
                "event_count": int(state.event_count),
                "note": str(state.note),
            }
        except Exception:
            return None

    def _get_social(self) -> dict[str, Any]:
        agg = self._social_agg
        if agg is None or not self._social_items:
            return {}
        result: dict[str, Any] = {}
        for symbol in self._symbols:
            try:
                metrics = agg.aggregate(self._social_items, ticker=symbol)
                result[symbol] = {
                    "mention_volume": getattr(metrics, "mention_volume", 0),
                    "velocity": float(getattr(metrics, "velocity", 0.0)),
                    "sentiment_mean": float(getattr(metrics, "sentiment_mean", 0.0)),
                    "bullish_pct": float(getattr(metrics, "bullish_pct", 0.0)),
                }
            except Exception:
                continue
        return result
