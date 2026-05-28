"""``recent_turns`` — retrieve this trader's recent decision history.

Tool name:      recent_turns
Args:           n=5 (int), include_tool_calls=True (bool)
ToolResult:     ok=True, data={"turns": [TurnSummary, …]}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (degrades gracefully when turn_store absent)
Example use:    recent_turns(n=3) to see what happened in the last 3 turns.

Each TurnSummary includes:
    turn_id         str
    started_at      str  (ISO-8601 UTC)
    wake_reason     str
    turn_type       str
    final_action    str  ("hold", "pass", "done_for_day", "interrupted", …)
    total_cost_usd  float
    tool_calls      list[{name, args_summary}]  — only when include_tool_calls=True

Wraps ``intel/turn_store.TurnStore`` (A5 ships the full store; A1 returns an empty
list gracefully when the store is absent, matching the A0 no-history state).
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class RecentTurnsTool(LookToolBase):
    """Read-only access to this trader's turn history.

    Parameters
    ----------
    turn_store:
        An object with a ``recent(trader_id, n)`` method returning a list of
        :class:`~trading_agent.intel.turn_store.TurnRecord`-like objects
        (duck-typed to avoid import cycle with A5).  ``None`` is fine — the
        tool returns an empty list with a note.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "recent_turns",
        "description": "Retrieve your N most recent turns (wake reason, tool calls, final action, cost).",
        "args": {"n": "int (default 5)", "include_tool_calls": "bool (default true)"},
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
        turn_store: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._turn_store = turn_store

    def __call__(
        self,
        n: int = 5,
        include_tool_calls: bool = True,
    ) -> Any:
        """Return the N most recent turns for this trader.

        Returns
        -------
        ToolResult
            ok=True, data={"turns": […]}  — list may be empty when store absent.

        Example
        -------
        >>> tool = RecentTurnsTool(trader_id="Alpha")
        >>> result = tool(n=3)
        >>> result.ok
        True
        >>> result.data["turns"]
        []
        """
        if self._turn_store is None:
            return self._ok(
                {
                    "turns": [],
                    "note": "turn history not yet available (A5 turn_store not wired)",
                }
            )
        n = max(1, min(int(n), 50))
        try:
            records = self._turn_store.recent(self.trader_id, n)
        except Exception as exc:
            return self._err("internal", f"turn_store.recent failed: {exc}")

        turns = []
        for rec in records:
            entry: dict[str, Any] = {
                "turn_id": getattr(rec, "turn_id", ""),
                "started_at": _iso(getattr(rec, "started_at", None)),
                "wake_reason": getattr(rec, "wake_reason", ""),
                "turn_type": getattr(rec, "turn_type", ""),
                "final_action": getattr(rec, "final_action", ""),
                "total_cost_usd": float(getattr(rec, "total_cost_usd", 0.0)),
            }
            if include_tool_calls:
                raw_calls = getattr(rec, "tool_calls", []) or []
                entry["tool_calls"] = [
                    {
                        "name": getattr(tc, "tool_name", str(tc)),
                        "args_summary": _summarize_args(
                            getattr(tc, "args", {}) or {}
                        ),
                    }
                    for tc in raw_calls
                ]
            turns.append(entry)

        return self._ok({"turns": turns})


# --------------------------------------------------------------------------- helpers


def _iso(dt: Any) -> str:
    if dt is None:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _summarize_args(args: dict[str, Any]) -> str:
    """Compact args summary — truncate large values to keep the snapshot small."""
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 60:
            sv = sv[:57] + "…"
        parts.append(f"{k}={sv}")
    return ", ".join(parts)
