"""Common base for ACT toolkit tools.

All ACT tools share:
  - access to broker (BrokerAdapter; duck-typed to avoid import cycle)
  - access to risk_manager (RiskManager; duck-typed)
  - access to pending_trade_queue (PendingTradeQueue; may be None → execute directly)
  - trader_id + turn_id for idempotency-key construction
  - requires_approval flag (True → route through PendingTradeQueue)
  - _ok / _err convenience constructors

Design role: centralise dependency plumbing and keep individual tool files thin.

Failure mode: broker absent → 'unavailable' error; the loop never raises.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...tool_envelope import ToolError, ToolResult


class ActToolBase:
    """Shared scaffolding for A3 ACT tools.

    Parameters
    ----------
    broker:
        BrokerAdapter instance (duck-typed).  May be None — tools return
        ``unavailable`` rather than raising.
    risk_manager:
        RiskManager instance (duck-typed).  May be None — checks are skipped.
    pending_trade_queue:
        PendingTradeQueue for approval-required paths.  None → execute directly.
    trader_id:
        Bench competitor name; used in idempotency key.
    turn_id:
        UUID generated once per ``AgentTrader.decide()`` call; scopes idem keys.
    requires_approval:
        When True, ``trade()`` routes through PendingTradeQueue instead of
        executing directly.
    scheduler:
        Optional :class:`~trading_agent.bench.scheduler.MarketScheduler`.  When
        present, ``trade()`` calls
        :meth:`~trading_agent.bench.scheduler.MarketScheduler.wire_pending_trade_callbacks`
        after enqueueing a pending trade so that approve / deny / TTL-expire
        events schedule a callback turn for the trader (A4-b wiring).
    """

    def __init__(
        self,
        *,
        broker: Any = None,
        risk_manager: Any = None,
        pending_trade_queue: Any = None,
        trader_id: str,
        turn_id: str,
        requires_approval: bool = False,
        scheduler: Any = None,
    ) -> None:
        self.broker = broker
        self.risk_manager = risk_manager
        self.pending_trade_queue = pending_trade_queue
        self.trader_id = trader_id
        self.turn_id = turn_id
        self.requires_approval = requires_approval
        self.scheduler = scheduler

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


def _idempotency_key(trader_id: str, turn_id: str, symbol: str, side: str, qty: float) -> str:
    """Deterministic hash used to detect crash-replay double-fires.

    Components: trader_id + turn_id + symbol (uppercase) + side (uppercase) +
    qty (rounded to 6dp).  Changing any component produces a different key,
    so a new turn always gets a fresh namespace.
    """
    raw = json.dumps(
        [trader_id, turn_id, symbol.upper(), side.upper(), round(qty, 6)],
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _scrub_fill(result: dict[str, Any]) -> dict[str, Any]:
    """Return broker result with no paper/sim disclosure strings.

    Any value-string containing a forbidden word is dropped from the output.
    Key names are never altered (they don't embed disclosure strings).
    """
    _forbidden = {"paper", "sim", "demo", "fake", "test mode"}
    scrubbed: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, str) and any(w in v.lower() for w in _forbidden):
            continue
        scrubbed[k] = v
    return scrubbed
