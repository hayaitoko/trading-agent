"""Approvals router (WS-I): human-in-the-loop trade gate over the live queue.

Reads/decides pending proposals on the :class:`~trading_agent.approval_queue.ApprovalQueue`
attached at ``app.state.approvals`` (the serve process). The legacy single-process
notification-center app (:func:`trading_agent.web.app.create_app`) wires its own
queue directly; this is the multi-user cockpit version that reads it off
``app.state`` and degrades to an empty list when no queue is attached.

Pending records are shaped to the cockpit's ``APPROVALS`` array:
``{id, m (who), t (human title), meta}`` — see ``design/cockpit.html`` ``resolveAp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user

if TYPE_CHECKING:
    from ...approval_queue import ApprovalQueue, ApprovalRecord

router = APIRouter(tags=["approvals"])


class Decision(BaseModel):
    """Optional body for approve/reject — a free-text operator note."""

    note: str | None = None


def _queue(request: Request) -> ApprovalQueue | None:
    return getattr(request.app.state, "approvals", None)


_VERB = {"BUY": "Buy", "LONG": "Buy", "SELL": "Sell", "SHORT": "Sell"}


def _approval_public(record: ApprovalRecord) -> dict[str, Any]:
    """Map an ApprovalRecord onto the cockpit's pending-approval card."""
    sig = record.signal or {}
    side = str(sig.get("side") or sig.get("action") or "").upper()
    symbol = sig.get("symbol") or sig.get("ticker") or "?"
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
