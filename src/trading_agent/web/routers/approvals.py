"""Approvals router (WS-I): human-in-the-loop trade gate over the live queue.

Two independent surfaces:

1. ``/api/approvals`` — legacy :class:`~trading_agent.approval_queue.ApprovalQueue`
   endpoints.  Reads/decides pending proposals on the queue attached at
   ``app.state.approvals``.  Degrades to an empty list when absent.

2. ``/api/pending-trades`` — WS-Agent A3 :class:`~trading_agent.approval_queue.PendingTradeQueue`
   endpoints.  Approve or deny a pending AgentTrader trade; the approval fires
   registered callbacks synchronously so the trader's callback turn is triggered
   from the same request context.  Attached at ``app.state.pending_trades``; all
   endpoints degrade gracefully when absent.

Pending records are shaped to the cockpit's ``APPROVALS`` array:
``{id, m (who), t (human title), meta}`` — see ``design/cockpit.html`` ``resolveAp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user

if TYPE_CHECKING:
    from ...approval_queue import ApprovalQueue, ApprovalRecord, PendingTradeQueue

router = APIRouter(tags=["approvals"])


class Decision(BaseModel):
    """Optional body for approve/reject — a free-text operator note."""

    note: str | None = None


def _queue(request: Request) -> ApprovalQueue | None:
    return getattr(request.app.state, "approvals", None)


def _pending_queue(request: Request) -> PendingTradeQueue | None:
    return getattr(request.app.state, "pending_trades", None)


_VERB = {"BUY": "Buy", "LONG": "Buy", "SELL": "Sell", "SHORT": "Sell"}


def _approval_public(record: ApprovalRecord) -> dict[str, Any]:
    """Map an ApprovalRecord onto the cockpit's pending-approval card."""
    sig = record.signal or {}
    side = str(sig.get("side") or sig.get("action") or "").upper()
    symbol = sig.get("asset") or sig.get("symbol") or sig.get("ticker") or "?"
    amount = sig.get("amount") or sig.get("quantity") or sig.get("qty")
    verb = _VERB.get(side, side.title() or "Trade")
    qty = f"{amount:g} shares of " if isinstance(amount, (int, float)) else ""
    who = sig.get("strategy") or sig.get("model") or sig.get("name") or sig.get("trader") or "trader"
    meta = sig.get("reason") or sig.get("rationale") or f"expires {record.expires_at:%H:%M:%S}"
    return {
        "id": record.proposal_id,
        "m": str(who),
        "t": f"{verb} {qty}{symbol}".strip(),
        "meta": str(meta),
    }


@router.get("/api/approvals")
def approvals(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """Pending proposals awaiting the operator. Empty -> cockpit shows 'all clear'."""
    queue = _queue(request)
    if queue is None:
        return []
    return [_approval_public(r) for r in queue.pending()]


@router.post("/api/approvals/{proposal_id}/approve")
def approve(
    proposal_id: str,
    request: Request,
    body: Decision | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    queue = _queue(request)
    if queue is None:
        raise HTTPException(status_code=503, detail="approval queue not running")
    try:
        result = queue.approve(proposal_id, note=body.note if body else None)
    except KeyError:
        raise HTTPException(status_code=404, detail="proposal not found")
    except ValueError as exc:  # not pending (already decided / expired)
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:  # no executor configured on the queue
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "approved", "result": result}


@router.post("/api/approvals/{proposal_id}/reject")
def reject(
    proposal_id: str,
    request: Request,
    body: Decision | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    queue = _queue(request)
    if queue is None:
        raise HTTPException(status_code=503, detail="approval queue not running")
    try:
        queue.reject(proposal_id, note=body.note if body else None)
    except KeyError:
        raise HTTPException(status_code=404, detail="proposal not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "rejected"}


# ---------------------------------------------------------------------------
# WS-Agent A3: PendingTrade endpoints — approval fires event-driven callback
# ---------------------------------------------------------------------------


def _pending_trade_public(pt: Any) -> dict[str, Any]:
    """Shape a PendingTrade for the cockpit's pending-approvals card."""
    intent = pt.proposed
    verb = "Buy" if str(intent.side).upper() == "BUY" else "Sell"
    ttl_str = (
        pt.approval_ttl_expires_at.strftime("%H:%M:%S UTC")
        if pt.approval_ttl_expires_at
        else f"expires in {_pending_queue_ttl_display(pt)}"
    )
    return {
        "id": pt.pending_trade_id,
        "m": str(pt.trader_id),
        "t": f"{verb} {intent.qty:g} shares of {intent.symbol}",
        "meta": f"status={pt.status}, {ttl_str}",
        "status": pt.status,
        "symbol": intent.symbol,
        "side": intent.side,
        "qty": intent.qty,
    }


def _pending_queue_ttl_display(pt: Any) -> str:
    return "awaiting operator approval"


@router.get("/api/pending-trades")
def list_pending_trades(
    request: Request,
    trader_id: str | None = None,
    user_id: str = Depends(current_user),
) -> list[dict[str, Any]]:
    """Pending trades awaiting operator decision.  Empty → all clear."""
    ptq = _pending_queue(request)
    if ptq is None:
        return []
    if trader_id:
        pts = ptq.pending_for_trader(trader_id)
    else:
        # Return all pending by querying every trader via a broad status scan.
        with ptq._lock:
            rows = ptq._conn.execute(
                "SELECT * FROM pending_trades "
                "WHERE status IN ('awaiting_approval', 'approved') "
                "ORDER BY proposed_at ASC"
            ).fetchall()
        pts = [ptq._row_to_pending_trade(r) for r in rows]
    return [_pending_trade_public(pt) for pt in pts]


@router.post("/api/pending-trades/{pending_trade_id}/approve")
def approve_pending_trade(
    pending_trade_id: str,
    request: Request,
    body: Decision | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Approve a pending trade.  Fires the trader's callback turn synchronously."""
    ptq = _pending_queue(request)
    if ptq is None:
        raise HTTPException(status_code=503, detail="pending trade queue not running")
    try:
        pt = ptq.set_decision(
            pending_trade_id, "approved", note=body.note if body else None
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="pending trade not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    ttl_iso = (
        pt.approval_ttl_expires_at.isoformat() if pt.approval_ttl_expires_at else None
    )
    return {
        "status": "approved",
        "pending_trade_id": pending_trade_id,
        "approval_ttl_expires_at": ttl_iso,
    }


@router.post("/api/pending-trades/{pending_trade_id}/deny")
def deny_pending_trade(
    pending_trade_id: str,
    request: Request,
    body: Decision | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Deny a pending trade.  Fires the trader's callback turn synchronously."""
    ptq = _pending_queue(request)
    if ptq is None:
        raise HTTPException(status_code=503, detail="pending trade queue not running")
    try:
        ptq.set_decision(
            pending_trade_id, "denied", note=body.note if body else None
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="pending trade not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "denied", "pending_trade_id": pending_trade_id}
