"""Advisor-notes router (WS-H fills in; over notes table, scope∈{trader,ticker}). Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["notes"])

_NOT_IMPLEMENTED = "not implemented yet (WS-H)"


@router.get("/api/notes")
def get_notes(
    scope: str | None = None, ref: str | None = None, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.put("/api/notes")
def put_note(
    scope: str | None = None, ref: str | None = None, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
