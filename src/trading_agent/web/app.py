"""FastAPI applications for the trading agent.

Two apps live here:

1. :func:`create_app` — the original single-process **notification-center** app
   (live ``NotificationCenter`` + ``ApprovalQueue``), wired end to end by
   ``trading-agent-serve``. Unchanged.

2. :func:`create_cockpit_app` — the **multi-user cockpit** app (WS-0). It mounts
   one router per workstream (``CONTRACTS.md §HTTP route table``), wires the
   SQLite-backed per-user spine (auth/sessions, settings, endpoint registry) onto
   ``app.state``, and serves the cockpit SPA. ``config`` is fully implemented;
   every other router answers 501 until its owning stream fills it in.

The module exposes a lazily-built default cockpit ``app`` so
``uvicorn trading_agent.web.app:app`` works without import-time DB side effects
for callers that only want :func:`create_app` / :func:`create_cockpit_app`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config.db import Database
from ..config.endpoints import EndpointRegistry
from ..config.settings_store import SettingsStore
from .routers import approvals as approvals_router
from .routers import bench as bench_router
from .routers import config as config_router
from .routers import manager as manager_router
from .routers import market as market_router
from .routers import notes as notes_router
from .routers import notifications as notifications_router
from .routers import requests as requests_router
from .routers import research as research_router
from .routers import risk as risk_router

if TYPE_CHECKING:
    from ..approval_queue import ApprovalQueue
    from .notifications import NotificationCenter

_STATIC_DIR = Path(__file__).parent / "static"

# Every workstream router, in route-table order. WS-G copies cockpit.html into
# static/ and swaps the mock data for fetch() calls against these.
_COCKPIT_ROUTERS = (
    config_router.router,
    bench_router.router,
    research_router.router,
    market_router.router,
    manager_router.router,
    risk_router.router,
    approvals_router.router,
    notifications_router.router,
    requests_router.router,
    notes_router.router,
)


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


def create_cockpit_app(
    db: Database | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    title: str = "Trading Agent — Cockpit",
) -> FastAPI:
    """Build the multi-user cockpit app (WS-0 spine + all stream routers).

    ``db`` lets tests pass an isolated ``Database(tmp_path)``; ``transport`` is a
    test seam injected into every endpoint client so model calls can be mocked.
    """
    database = db if db is not None else Database()
    app = FastAPI(title=title, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.db = database
    app.state.settings = SettingsStore(database)
    app.state.endpoints = EndpointRegistry(database, transport=transport)

    for router in _COCKPIT_ROUTERS:
        app.include_router(router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        cockpit = _STATIC_DIR / "cockpit.html"
        if cockpit.exists():  # WS-G drops the wired SPA here
            return FileResponse(cockpit)
        return JSONResponse(
            {"app": "trading-agent cockpit", "note": "UI pending WS-G; API under /api"}
        )

    return app


# Lazily-built default cockpit app for `uvicorn trading_agent.web.app:app`.
# Built on first attribute access (PEP 562) so importing this module for its
# factories doesn't touch the default data/config.db.
_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    if name == "app":
        global _default_app
        if _default_app is None:
            _default_app = create_cockpit_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
