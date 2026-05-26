"""Approvals router (WS-H/G fill in; over approval_queue). Stubbed 501.

Note: the legacy single-process notification-center app
(:func:`trading_agent.web.app.create_app`) implements approve/reject against a
live in-memory ``ApprovalQueue``. This router is the multi-user, DB-backed
cockpit version and is filled in by its owning stream.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["approvals"])

_NOT_IMPLEMENTED = "not implemented yet"


@router.get("/api/approvals")
def approvals(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/approvals/{proposal_id}/approve")
def approve(proposal_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/approvals/{proposal_id}/reject")
def reject(proposal_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
