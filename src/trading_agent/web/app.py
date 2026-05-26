"""FastAPI application for the notification center.

Endpoints:
    GET  /                              -> single-page UI
    GET  /api/notifications             -> NotificationCenter.snapshot()
    POST /api/approvals/{pid}/approve   -> ApprovalQueue.approve() (executes)
    POST /api/approvals/{pid}/reject    -> ApprovalQueue.reject()
    GET  /api/health                    -> liveness

The app holds references to a live ``NotificationCenter`` and ``ApprovalQueue``;
approve/reject run the queue's executor, so the app must share the process (and
the in-memory broker) that owns that executor. ``trading-agent-serve`` wires
this up end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..approval_queue import ApprovalQueue
    from .notifications import NotificationCenter

_STATIC_DIR = Path(__file__).parent / "static"


class Decision(BaseModel):
    """Optional body for approve/reject — a free-text operator note."""

    note: str | None = None


def create_app(
    center: NotificationCenter,
    approval_queue: ApprovalQueue,
    *,
    title: str = "Trading Agent — Alerts",
) -> FastAPI:
    app = FastAPI(title=title, docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/notifications")
    def notifications() -> JSONResponse:
        return JSONResponse(center.snapshot())

    @app.post("/api/approvals/{proposal_id}/approve")
    def approve(proposal_id: str, decision: Decision | None = None) -> JSONResponse:
        note = decision.note if decision else None
        try:
            result = approval_queue.approve(proposal_id, note=note)
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found")
        except ValueError as exc:  # not pending (already decided / expired)
            raise HTTPException(status_code=409, detail=str(exc))
        except RuntimeError as exc:  # no executor configured
            raise HTTPException(status_code=503, detail=str(exc))
        return JSONResponse({"status": "approved", "result": result})

    @app.post("/api/approvals/{proposal_id}/reject")
    def reject(proposal_id: str, decision: Decision | None = None) -> JSONResponse:
        note = decision.note if decision else None
        try:
            approval_queue.reject(proposal_id, note=note)
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return JSONResponse({"status": "rejected"})

    return app
