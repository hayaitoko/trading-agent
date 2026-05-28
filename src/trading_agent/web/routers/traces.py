"""Cockpit observability router — turn traces, replay, cost rollup (A5).

**Design role:** operator-facing REST endpoints that expose the agent turn
trace store (:mod:`~trading_agent.intel.turn_store`) to the cockpit SPA.
The trader-side ``recent_turns()`` LOOK tool reads the same store directly.

**MONEY IS REAL — critical invariant:**
These endpoints are **operator-facing only**.  They return ``book_type`` and
``book_badge`` so the operator can clearly see paper vs live status.
The trader's ``recent_turns()`` tool uses :meth:`TurnRecord.to_trader_dict`
(no ``book_type``); these endpoints use :meth:`TurnRecord.to_operator_dict`
or :meth:`TurnRecord.to_summary_dict` (both include ``book_type``).

**MANAGER FRUGALITY — no LLM calls here.**
This router is pure SQLite reads.  No model is invoked on any endpoint.
The cockpit tiles poll these endpoints on a timer; each poll is a DB read,
never a model call.

**Endpoints:**

``GET /api/traces?trader_id=<id>&limit=<N>``
    Return a JSON array of recent turn summaries for one trader.
    Includes ``book_type`` / ``book_badge`` for the operator paper/live badge.
    Degrades to an empty list when the store is absent or the trader has no turns.

``GET /api/traces/{turn_id}``
    Return the full TurnRecord for one turn — first-look snapshot, ordered
    tool-call list (name / args / result / latency / cost), final action, cost.
    Used by the ``turnReplay`` cockpit modal and the ``traderTrace`` expandable rows.

``GET /api/traces/cost?trader_id=<id>``
    Return rolling spend for one trader: today / week / lifetime in USD.
    Used by the ``costPerTrader`` tile.

``GET /api/traces/attention?trader_id=<id>``
    Return active (unfired) watchpoints and reminders for one trader.
    Used by the ``attentionPending`` tile.

All endpoints require authentication (``current_user`` dependency) and degrade
gracefully when ``app.state.turn_store`` is not wired.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...config.users import current_user
from ...intel.turn_store import TurnStore

router = APIRouter(tags=["traces"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(request: Request) -> TurnStore | None:
    """Pull the TurnStore from app state, or None if not wired."""
    return getattr(request.app.state, "turn_store", None)


def _attention_queue(request: Request) -> Any | None:
    """Pull the AttentionQueue from app state, or None if not wired."""
    return getattr(request.app.state, "attention_queue", None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/traces")
def list_traces(
    request: Request,
    trader_id: str = Query(..., description="Bench trader identifier"),
    limit: int = Query(20, ge=1, le=200, description="Max turns to return"),
    user_id: str = Depends(current_user),
) -> list[dict[str, Any]]:
    """Return recent turn summaries for one trader (operator view).

    Each summary includes: ``turn_id``, ``started_at``, ``ended_at``,
    ``wake_reason``, ``turn_type``, ``final_action``, ``total_cost_usd``,
    ``tool_call_count``, ``book_type``, ``book_badge``.

    The ``book_badge`` is ``"[PAPER]"`` or ``"[LIVE]"`` — visible to the
    operator in the cockpit ``traderTrace`` tile.

    **OPERATOR PATH:** ``book_type`` / ``book_badge`` are ONLY present here.
    The trader-side ``recent_turns()`` tool NEVER returns these fields.

    Returns an empty list when the store is absent or the trader has no turns.
    No LLM calls — pure SQLite read.
    """
    store = _store(request)
    if store is None:
        return []
    return store.summaries(trader_id, limit=limit)


@router.get("/api/traces/cost")
def trace_cost(
    request: Request,
    trader_id: str = Query(..., description="Bench trader identifier"),
    user_id: str = Depends(current_user),
) -> dict[str, float]:
    """Return rolling spend totals for one trader (operator view).

    Returns::

        {
            "today":    <USD spent in last 24 h>,
            "week":     <USD spent in last 7 d>,
            "lifetime": <USD spent total>
        }

    Used by the ``costPerTrader`` cockpit tile.
    No LLM calls — pure SQLite aggregation.
    """
    store = _store(request)
    if store is None:
        return {"today": 0.0, "week": 0.0, "lifetime": 0.0}
    return store.cost_rollup(trader_id)


@router.get("/api/traces/attention")
def trace_attention(
    request: Request,
    trader_id: str = Query(..., description="Bench trader identifier"),
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Return active watchpoints and reminders for one trader (operator view).

    Returns::

        {
            "watchpoints": [{"id": .., "symbol": .., "why": .., "expires_at": ..}, ...],
            "reminders":   [{"id": .., "about": .., "when_unix": .., "expires_at": ..}, ...],
            "total": <int>
        }

    Used by the ``attentionPending`` tile.  Includes a ``prune`` hint field
    so the cockpit can surface a "Mark expired" action.
    No LLM calls — pure SQLite read.
    """
    aq = _attention_queue(request)
    if aq is None:
        return {"watchpoints": [], "reminders": [], "total": 0}

    try:
        import json

        rows = aq._conn.execute(
            """
            SELECT id, kind, payload_json, expires_at, created_at
            FROM attention_queue
            WHERE trader_id=? AND fired_at IS NULL
            ORDER BY created_at ASC
            """,
            (trader_id,),
        ).fetchall()
    except Exception:
        return {"watchpoints": [], "reminders": [], "total": 0}

    watchpoints = []
    reminders = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        item = {
            "id": row["id"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
        }
        if row["kind"] == "watchpoint":
            item["symbol"] = payload.get("symbol", "")
            item["why"] = payload.get("why", "")
            item["condition"] = payload.get("condition")
            watchpoints.append(item)
        else:
            item["about"] = payload.get("about", payload.get("why", ""))
            item["when_unix"] = payload.get("when_unix")
            reminders.append(item)

    return {
        "watchpoints": watchpoints,
        "reminders": reminders,
        "total": len(watchpoints) + len(reminders),
    }


@router.get("/api/traces/{turn_id}")
def get_trace(
    turn_id: str,
    request: Request,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Return the full TurnRecord for one turn (operator view).

    Used by the ``turnReplay`` cockpit modal.  Returns:

    - ``turn_id``, ``trader_id``, ``started_at``, ``ended_at``
    - ``wake_reason``, ``turn_type``
    - ``first_look_snapshot`` — the full structured context block the trader saw
    - ``tool_calls`` — ordered list of ``{tool_name, args, result, latency_ms, cost_usd}``
    - ``final_action``, ``final_action_args``
    - ``total_cost_usd``, ``total_tokens``
    - ``book_type``, ``book_badge`` (OPERATOR ONLY — never sent to the trader)
    - ``previous_attempt_turn_id`` — set on crash-recovery turns

    ``tool_calls[*].result`` is the raw ``ToolResult.to_dict()`` payload — the
    exact JSON the agent received.  The ``turnReplay`` modal renders this so the
    operator can inspect every step.

    No LLM calls — pure SQLite read.
    Raises 404 if the turn is not found.
    """
    store = _store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="turn store not wired")
    rec = store.get(turn_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"turn {turn_id!r} not found")
    return rec.to_operator_dict(include_tool_calls=True)
