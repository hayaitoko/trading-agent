"""Common base for NOTE toolkit tools.

All NOTE tools share:
  - access to :class:`~trading_agent.intel.attention_queue.AttentionQueue`
  - access to :class:`~trading_agent.memory.store.MemoryStore` (for ``reflect``)
  - ``owner_user_id`` + ``trader_id`` for namespacing
  - ``_ok`` / ``_err`` convenience constructors

Design role: centralise imports and keep individual tool files thin.

Failure mode: if the underlying store is absent (``None``), tools return
a degraded-but-not-crashing ``ToolResult`` so the agent can still call
them without errors.
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolError, ToolResult


class NoteToolBase:
    """Shared scaffolding for A2 NOTE tools.

    Parameters
    ----------
    attention_queue:
        The :class:`~trading_agent.intel.attention_queue.AttentionQueue`
        instance (may be ``None`` — tools degrade gracefully).
    memory:
        The :class:`~trading_agent.memory.store.MemoryStore` (may be ``None``).
    owner_user_id:
        The user who owns this trader.
    trader_id:
        The unique trader identifier (the bench competitor name).
    """

    def __init__(
        self,
        *,
        attention_queue: Any = None,
        memory: Any = None,
        owner_user_id: str | None = None,
        trader_id: str,
    ) -> None:
        self.attention_queue = attention_queue
        self.memory = memory
        self.owner_user_id = owner_user_id
        self.trader_id = trader_id

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ok(data: Any) -> ToolResult:
        return ToolResult(ok=True, data=data)

    @staticmethod
    def _err(kind: str, message: str, *, retry_after: int | None = None) -> ToolResult:
        return ToolResult(
            ok=False,
            error=ToolError(kind=kind, message=message, retry_after=retry_after),  # type: ignore[arg-type]
        )
