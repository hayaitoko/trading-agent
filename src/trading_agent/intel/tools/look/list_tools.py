"""``list_tools`` — enumerate every tool the agent can call.

Tool name:      list_tools
Args:           (none)
ToolResult:     ok=True, data={"tools": [ToolEntry, …]}
Latency tier:   instant
Cost class:     free
Gating flag:    always enabled
Example use:    call at the start of a turn to discover available tools.

Returns a JSON catalog of every tool in the LOOK / NOTE / ACT / END catalogs.
Each entry includes:

    name            str    — callable tool name
    description     str    — one-sentence purpose
    args            dict   — arg name → type / default hint
    latency         str    — "instant" | "fast" | "medium" | "slow" | "queued"
    cost_class      str    — "free" | "model_call" | "queued"
    enabled         bool   — False for tools whose provider has not yet landed
    disabled_reason str|None — non-null only when enabled=False

Disabled tools are still listed so the model can reason about absence and
understand what capabilities are coming.  When WS-Situation Track A lands,
only the stubs get unwired — this wrapper stays.
"""

from __future__ import annotations

from typing import Any

from ....intel.tool_envelope import ToolResult
from ._base import LookToolBase


class ListToolsTool(LookToolBase):
    """Static catalog builder: collects metadata from every registered tool.

    A1 extends the A0 built-in catalog by merging in the full LOOK set.  A2 and
    A3 append NOTE and ACT entries when those waves land.  The catalog is built
    fresh on each call so enabling/disabling a provider flag is immediately
    reflected.

    Failure mode: catalog construction is pure-Python; it cannot fail.
    """

    TOOL_META: dict[str, Any] = {
        "name": "list_tools",
        "description": (
            "List every tool available to you with name, description, argument "
            "schema, latency tier, and cost class.  Call this first on unfamiliar "
            "turns to discover what you can do."
        ),
        "args": {},
        "latency": "instant",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    # The full A1 catalog is injected at construction so the trader loop can
    # pass a pre-built list (typically assembled once per trader instance and
    # reused across turns).
    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        extra_entries: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._extra_entries: list[dict[str, Any]] = list(extra_entries or [])

    def __call__(self) -> ToolResult:
        """Return the full tool catalog.

        Returns
        -------
        ToolResult
            ok=True, data={"tools": [<entry>, …]}

        Example
        -------
        >>> tool = ListToolsTool(trader_id="Alpha")
        >>> result = tool()
        >>> result.ok
        True
        >>> result.data["tools"][0]["name"]
        'list_tools'
        """
        catalog = self._build_catalog()
        return self._ok({"tools": catalog})

    def _build_catalog(self) -> list[dict[str, Any]]:
        """Assemble the complete tool catalog."""
        # A0 built-ins
        builtins: list[dict[str, Any]] = [
            {
                "name": "list_tools",
                "description": "List all available tools (this call).",
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "memory_search",
                "description": "Search your private memory for lessons and reflections.",
                "args": {"query": "str", "k": "int (default 5)"},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "hold",
                "description": "Terminal: end this turn having considered the situation.",
                "args": {"reason": "str"},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "pass",
                "description": "Terminal: end this turn — nothing interesting to act on.",
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
        ]

        # A1 LOOK tools
        look_tools: list[dict[str, Any]] = [
            {
                "name": "recent_turns",
                "description": (
                    "Retrieve your N most recent turns (first-look snapshot, "
                    "tool calls, final action, cost).  Use for continuity across turns."
                ),
                "args": {
                    "n": "int (default 5)",
                    "include_tool_calls": "bool (default true)",
                },
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "history",
                "description": (
                    "Historical OHLCV bars for a symbol over the last N days. "
                    "Returns close prices, volume, and key stats."
                ),
                "args": {"symbol": "str", "days": "int (default 30)"},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "news",
                "description": (
                    "Recent news/social headlines from the ingest pipeline. "
                    "Pass symbol to narrow to one ticker; omit for cross-ticker feed."
                ),
                "args": {"symbol": "str|None (default None)", "limit": "int (default 10)"},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "research_brief",
                "description": (
                    "Retrieve the latest distilled research brief for a symbol "
                    "(WS-C output — LLM summary of recent news/social items, "
                    "sentiment score, catalysts list)."
                ),
                "args": {"symbol": "str"},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "request_research",
                "description": (
                    "Queue a new research pass for a symbol+question.  Returns "
                    "immediately with a request_id; check research_brief() on the "
                    "next turn to see the result."
                ),
                "args": {"symbol": "str", "question": "str"},
                "latency": "queued",
                "cost_class": "queued",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "situation",
                "description": (
                    "Current market situation block: regime label (calm/elevated/"
                    "event-window/risk-off), realized vol, social metrics per symbol, "
                    "and upcoming calendar events."
                ),
                "args": {},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "world_events",
                "description": (
                    "GDELT-based global macro and geopolitical event feed filtered by "
                    "theme. Returns mention-volume timeline and recent headlines. "
                    "Returns disabled error when SITUATION_GDELT flag is off."
                ),
                "args": {
                    "theme": "str|None (default None — queries WAR + ELECTION + EPU_POLICY)",
                    "timespan": "str (default '24h'; e.g. '48h', '7d')",
                },
                "latency": "medium",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "prediction_market_odds",
                "description": (
                    "Polymarket + Kalshi implied probabilities for macro events by "
                    "category. Returns disabled error when "
                    "SITUATION_PREDICTION_MARKETS flag is off."
                ),
                "args": {
                    "category": "str (e.g. 'economics', 'politics', 'fed_rate', 'crypto')",
                    "query": "str|None (default None — optional title substring filter)",
                    "min_liquidity": "float (default 1000.0 — minimum USD liquidity)",
                },
                "latency": "medium",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "options_iv",
                "description": (
                    "Implied volatility and Greeks for near-the-money options on a "
                    "symbol. Returns disabled error when SITUATION_OPTIONS_IV flag is off."
                ),
                "args": {
                    "symbol": "str (equity ticker, e.g. 'AAPL', 'SPY')",
                    "expiry": "str|None (default None — nearest expiry; ISO 'YYYY-MM-DD')",
                },
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "forecast",
                "description": (
                    "Forward 1σ price-cone forecast for a symbol over 5/10/30 day horizon "
                    "combining realized vol, options IV, and prediction-market implied move. "
                    "Anti-overconfidence: mid line is flat (no drift estimate). "
                    "Returns disabled error when SITUATION_FORECAST flag is off."
                ),
                "args": {
                    "symbol": "str (e.g. 'AAPL', 'SPY', 'BTC/USD')",
                    "horizon": "5 | 10 | 30 (default 30)",
                },
                "latency": "medium",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "watchlist",
                "description": (
                    "Your own watchlist of symbols (set via A2 watch_symbol / "
                    "unwatch_symbol), overlaid with any symbols the operator has "
                    "pinned for you."
                ),
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "account_state",
                "description": (
                    "Fresh snapshot of your account: cash, open positions with "
                    "current market value, unrealized P&L, and recent fills."
                ),
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "advisor_notes",
                "description": (
                    "Operator-written advisor notes scoped to your account (scope='trader'), "
                    "a specific ticker (scope='ticker'), or global notes (scope='global'). "
                    "Returns notes visible to this trader only — never other traders' notes."
                ),
                "args": {
                    "symbol": "str|None (default None, required when scope='ticker')",
                    "scope": "'ticker' | 'trader' | 'global' (default 'trader')",
                },
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "ask_manager",
                "description": (
                    "Ask the overseer manager a question.  Cost-gated: at most once "
                    "per turn.  The manager cannot disclose peer-trader state.  "
                    "Use for strategic guidance only."
                ),
                "args": {"question": "str"},
                "latency": "slow",
                "cost_class": "model_call",
                "enabled": True,
                "disabled_reason": None,
            },
        ]

        return builtins + look_tools + self._extra_entries
