"""Common base for LOOK toolkit tools (A1).

All LOOK tools are **read-only** — they never write to the broker, memory, or
any other mutable store.  They share:

  - ``owner_user_id`` + ``trader_id`` for namespacing (memory / notes isolation)
  - ``_ok`` / ``_err`` / ``_disabled`` convenience constructors that shape every
    response into a uniform :class:`~trading_agent.intel.tool_envelope.ToolResult`
  - ``TOOL_META`` class-level dict: the entry the tool contributes to
    ``list_tools()`` output (name, description, args, latency, cost_class, enabled,
    disabled_reason)

Design role: keeps individual tool files thin and import-clean.  No business
logic lives here.

Failure mode: if any optional upstream service is absent (``None``), each tool
degrades to a ``ToolResult(ok=True, data={…, "note": "…unavailable"})`` or, for
hard failures, ``ToolResult(ok=False, error=ToolError(…))``.  The agent always
sees a structured response.
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolError, ToolResult


class LookToolBase:
    """Shared scaffolding for A1 LOOK tools.

    Parameters
    ----------
    owner_user_id:
        The user who owns this trader.  Used to namespace memory, research, and
        notes lookups.  May be ``None`` — tools degrade gracefully.
    trader_id:
        The unique trader identifier (bench competitor name / ``AgentTrader.name``).
    """

    #: Subclasses override this with their catalog entry.  Used by
    #: :class:`~trading_agent.intel.tools.look.list_tools.ListToolsTool` to build
    #: the full tool catalog without hard-coding each tool's metadata.
    TOOL_META: dict[str, Any] = {}

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
    ) -> None:
        self.owner_user_id = owner_user_id
        self.trader_id = trader_id

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ok(data: Any) -> ToolResult:
        """Wrap a successful payload."""
        return ToolResult(ok=True, data=data)

    @staticmethod
    def _err(
        kind: str,
        message: str,
        *,
        retry_after: int | None = None,
    ) -> ToolResult:
        """Wrap a failure into a structured error."""
        return ToolResult(
            ok=False,
            error=ToolError(kind=kind, message=message, retry_after=retry_after),  # type: ignore[arg-type]
        )

    @staticmethod
    def _disabled(tool_name: str) -> ToolResult:
        """Return the canonical disabled-tool error.

        Returned when a tool's feature flag is off or its backing provider is
        not initialised.  ``list_tools()`` surfaces these with ``enabled=false``
        and a human-readable ``disabled_reason``.
        """
        return ToolResult(
            ok=False,
            error=ToolError(
                kind="disabled",
                message=(
                    f"{tool_name}: provider not available. "
                    "Use situation() or research_brief() for current market context."
                ),
            ),
        )
