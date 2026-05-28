"""Always-on first-look context composer for the agent trader loop.

Every agent turn begins by calling :func:`build_first_look`, which renders a
structured context block the trader sees as its first user message.  The block
is deterministic — given the same :class:`TurnContext`, it always renders
identically — so the LLM's context cache gets maximum benefit from the stable
system-prompt prefix that precedes it.

Rendered format (all fields present; optional fields default to "n/a"):

    Identity:         <name>, <model>, mandate=<mandate>
    Account:          cash=$<x>, positions=<n>, last_decision=<short>
    Wake reason:      <why this turn fired>
    Turn type:        <SoD | regular | event | reminder | EoD | callback>
    Time:             <UTC>, <ET>, [<user-tz>]
    Cadence:          every <N> min during RTH
    Attention:        <a> active watchpoints / <s> soft-limit, <r> active reminders / <s> soft-limit
    Cost this turn:   $<x.xx> (rollup: model+nested LLM calls)
    Previous attempt: <tool_a, tool_b>   ← only on crash-recovery turns

**MONEY IS REAL invariant:** this module never discloses paper/sim/demo status.
The account state (cash, positions) is surfaced identically regardless of
whether the underlying broker is a paper book or a live account.  The string
``"paper"`` must not appear anywhere in the rendered output — see
design/TRADER-AGENT.md §MONEY IS REAL.

Failure mode: ``zoneinfo`` absent or unknown TZ → ET/TZ fields fall back to
``"(unavailable)"`` rather than crashing; the rest of the block renders fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

TurnType = Literal["SoD", "regular", "event", "reminder", "EoD", "callback", "tutorial"]


@dataclass
class TurnContext:
    """All inputs needed to render one turn's first-look block.

    Fields map one-to-one with rendered lines.  Optional fields yield "n/a"
    or are omitted when absent (e.g. ``previous_attempt_tools`` only adds the
    ``Previous attempt:`` line when the list is non-empty).

    The caller is responsible for populating ``cost_so_far_usd`` from
    :class:`~trading_agent.intel.cost_tracker.CostTracker` before each render
    (the field can be updated in-place between tool calls).
    """

    # Identity
    trader_name: str
    model: str
    mandate: str | None = None

    # Account state — broker-agnostic; paper/live distinction must NOT appear here
    cash: float = 0.0
    position_count: int = 0
    last_decision: str | None = None

    # Turn metadata
    wake_reason: str = "scheduled"
    turn_type: TurnType = "regular"

    # Time
    utc_now: datetime | None = None
    user_tz: str | None = None  # e.g. "America/Los_Angeles"

    # Cadence
    cadence_minutes: int = 30

    # Attention queue soft-limit counters (A2 populates; defaults safe for A0)
    active_watchpoints: int = 0
    watchpoint_soft_limit: int = 20
    active_reminders: int = 0
    reminder_soft_limit: int = 10

    # Cost accumulator — updated by CostTracker across the turn loop
    cost_so_far_usd: float = 0.0

    # Crash-recovery: tool names from the interrupted prior attempt (no results)
    previous_attempt_tools: list[str] = field(default_factory=list)

    # Extra verbatim lines appended at the end (used by A1+ for enriched context)
    extra_lines: list[str] = field(default_factory=list)


def build_first_look(ctx: TurnContext) -> str:
    """Render a :class:`TurnContext` into the always-on first-look string.

    Returns a multi-line string used as the first user message in the turn's
    message list.  Must never contain ``"paper"``, ``"sim"``, ``"demo"``, or
    ``"fake"`` — the MONEY IS REAL invariant.
    """
    now = ctx.utc_now or datetime.now(UTC)
    mandate_str = ctx.mandate or "none specified"
    last_str = f'"{ctx.last_decision}"' if ctx.last_decision else "none"

    time_parts = [now.strftime("%Y-%m-%d %H:%M:%S UTC"), _to_et(now)]
    if ctx.user_tz:
        tz_str = _to_tz(now, ctx.user_tz)
        if tz_str:
            time_parts.append(tz_str)

    lines = [
        f"Identity:         {ctx.trader_name}, {ctx.model}, mandate={mandate_str}",
        f"Account:          cash=${ctx.cash:,.2f}, positions={ctx.position_count}, last_decision={last_str}",
        f"Wake reason:      {ctx.wake_reason}",
        f"Turn type:        {ctx.turn_type}",
        f"Time:             {', '.join(time_parts)}",
        f"Cadence:          every {ctx.cadence_minutes} min during RTH",
        (
            f"Attention:        {ctx.active_watchpoints} active watchpoints / "
            f"{ctx.watchpoint_soft_limit} soft-limit, "
            f"{ctx.active_reminders} active reminders / "
            f"{ctx.reminder_soft_limit} soft-limit"
        ),
        f"Cost this turn:   ${ctx.cost_so_far_usd:.4f} (rollup: model+nested LLM calls)",
    ]

    if ctx.previous_attempt_tools:
        lines.append(f"Previous attempt: {', '.join(ctx.previous_attempt_tools)}")

    lines.extend(ctx.extra_lines)
    return "\n".join(lines)


def _to_et(dt: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        et = dt.astimezone(ZoneInfo("US/Eastern"))
        return et.strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return "(ET unavailable)"


def _to_tz(dt: datetime, tz_name: str) -> str:
    try:
        from zoneinfo import ZoneInfo
        local = dt.astimezone(ZoneInfo(tz_name))
        return local.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")
    except Exception:
        return ""
