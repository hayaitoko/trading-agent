"""``prediction_market_odds`` — Polymarket/Kalshi event odds (gated by SITUATION_PREDICTION_MARKETS).

Tool name:      prediction_market_odds
Args:           category (str), query=None (str|None), min_liquidity=1000.0 (float)
ToolResult:     ok=True,  data={"events": [...]}
                ok=False, error=ToolError(kind="disabled", …) when flag off
                ok=False, error=ToolError(kind="network_error", …) on provider failure
Latency tier:   medium (60-second cache inside PredictionMarketsProvider)
Cost class:     free
Gating flag:    SITUATION_PREDICTION_MARKETS (user_settings, default False)

Purpose
-------
Wraps :class:`~trading_agent.data.providers.prediction_markets.PredictionMarketsProvider`
to surface prediction-market implied probabilities as a LOOK tool.  The trader
calls this tool to get forward-looking crowd-sourced probabilities on macro
events that may affect its positions:

  prediction_market_odds("economics")       → Fed/CPI/GDP events from both venues
  prediction_market_odds("politics", query="election")  → electoral events

The tool returns events from both Polymarket and Kalshi, sorted by 24h volume.
Results include outcomes + implied prices, liquidity, and resolution date.

The trader should read this tool when:
  - Assessing market expectations for an upcoming Fed decision or CPI print.
  - Understanding how prediction markets are pricing political risk.
  - Looking for events with directional implications for held positions.

Money-is-real note: this tool never discloses paper/live account status.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class PredictionMarketOddsTool(LookToolBase):
    """Polymarket + Kalshi implied probabilities for macro event categories.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    settings_store:
        Duck-typed: ``get(user_id, key, default)`` — checks
        ``SITUATION_PREDICTION_MARKETS`` flag.
    pm_provider:
        Pre-constructed :class:`~trading_agent.data.providers.prediction_markets.PredictionMarketsProvider`.
        ``None`` → returns disabled error even when flag is on.
    """

    TOOL_META: dict[str, Any] = {
        "name": "prediction_market_odds",
        "description": (
            "Prediction-market implied probabilities for macro events from "
            "Polymarket and Kalshi. "
            "Use to see what the crowd thinks about Fed decisions, elections, "
            "economic prints, or any other resolvable event. "
            "Enable via SITUATION_PREDICTION_MARKETS in trader settings."
        ),
        "args": {
            "category": "str (e.g. 'economics', 'politics', 'crypto', 'fed_rate')",
            "query": "str|None (default None — optional title substring filter)",
            "min_liquidity": "float (default 1000.0 — minimum USD liquidity)",
        },
        "latency": "medium",
        "cost_class": "free",
        "enabled": True,  # wired — returns disabled error when flag off
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        settings_store: Any = None,
        pm_provider: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._settings = settings_store
        self._pm = pm_provider

    def __call__(
        self,
        category: str,
        query: str | None = None,
        min_liquidity: float = 1_000.0,
    ) -> Any:
        """Fetch prediction-market event odds.

        Parameters
        ----------
        category:
            Category/topic filter (e.g. ``"economics"``, ``"politics"``,
            ``"crypto"``, ``"fed_rate"``).
        query:
            Optional case-insensitive substring filter on event titles.
        min_liquidity:
            Minimum USD liquidity for Polymarket markets (default $1,000).

        Returns
        -------
        ToolResult
            ok=True, data={"category": str, "events": [
              {"venue": str, "event_id": str, "title": str,
               "outcomes": [str, ...], "prices": [float, ...],
               "liquidity": float, "volume_24h": float,
               "end_date": str|None, "restricted": bool}, ...]}

            ok=False, error=ToolError(kind="disabled")
                When SITUATION_PREDICTION_MARKETS flag is off.

            ok=False, error=ToolError(kind="network_error", message=...)
                On provider network/HTTP failure.

        Example
        -------
        >>> tool = PredictionMarketOddsTool(trader_id="Alpha")
        >>> result = tool("economics")
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        if not self._flag_enabled():
            return self._err(
                "disabled",
                "prediction_market_odds: enable SITUATION_PREDICTION_MARKETS "
                "in trader settings to use this tool.",
            )

        if self._pm is None:
            return self._err(
                "disabled",
                "prediction_market_odds: prediction-markets provider not initialised.",
            )

        try:
            events = self._pm.event_odds(
                category, query, min_liquidity=min_liquidity
            )
        except Exception as exc:  # noqa: BLE001
            return self._err("network_error", f"prediction markets error: {exc}")

        return self._ok(
            {
                "category": category,
                "events": [_odds_to_dict(e) for e in events],
            }
        )

    # ------------------------------------------------------------------

    def _flag_enabled(self) -> bool:
        if self._settings is None:
            return False
        try:
            return bool(
                self._settings.get(
                    self.owner_user_id or "", "SITUATION_PREDICTION_MARKETS", False
                )
            )
        except Exception:  # noqa: BLE001
            return False


def _odds_to_dict(e: Any) -> dict[str, Any]:
    return {
        "venue": e.venue,
        "event_id": e.event_id,
        "title": e.title,
        "outcomes": list(e.outcomes),
        "prices": list(e.prices),
        "liquidity": e.liquidity,
        "volume_24h": e.volume_24h,
        "end_date": e.end_date.isoformat() if e.end_date is not None else None,
        "restricted": e.restricted,
    }
