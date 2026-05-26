"""Research router (WS-C fills in). ``run`` is cost-gated + explicit. Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["research"])

_NOT_IMPLEMENTED = "not implemented yet (WS-C)"


@router.get("/api/research")
def research(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/research/run")
def research_run(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
