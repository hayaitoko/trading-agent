"""Stock-requests router (WS-H): list requests, allow/decline.

Creation is programmatic (a trader emits a request via
:meth:`RequestService.submit`), so there is no POST-create route — only the
operator-facing list + decision endpoints from ``CONTRACTS.md §HTTP route table``:
``GET /api/requests``, ``POST /api/requests/{id}/allow|decline``. Allow adds the
symbol to the trader's universe and marks the request fulfilled; decline leaves
the universe unchanged.

B0 universe_listener: when a symbol is allowed, :func:`seed_sa_ticker` is called
idempotently so the per-ticker SeekingAlpha combined RSS feed is registered in the
ingest pipeline for the user.  This wires the gap noted in B0 — operators no longer
need to call seed_sa_ticker manually when approving a new symbol.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.users import current_user, get_db
from ...requests import RequestError, RequestService

router = APIRouter(tags=["requests"])


def _universe_listener(db: Any):
    """Return a universe_listener that seeds the SA per-ticker RSS on symbol-allow."""
    from ...ingest.seed_sources import seed_sa_ticker

    def _on_symbol_allowed(user_id: str, trader_id: str, symbol: str) -> None:
        try:
            seed_sa_ticker(db, user_id, symbol)
        except Exception:  # noqa: BLE001
            pass  # seed failures must never block the allow response

    return _on_symbol_allowed


def _service(request: Request) -> RequestService:
    db = get_db(request)
    return RequestService(db, universe_listener=_universe_listener(db))


@router.get("/api/requests")
def list_requests(
    request: Request,
    status: str | None = None,
    user_id: str = Depends(current_user),
) -> list[dict[str, Any]]:
    """A user's stock requests, newest first; ``?status=pending`` to filter."""
    svc = _service(request)
    return [r.as_dict() for r in svc.requests.list(user_id, status=status)]


@router.post("/api/requests/{request_id}/allow")
def allow(
    request: Request,
    request_id: str,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    svc = _service(request)
    try:
        req = svc.allow(user_id, request_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="request not found")
    except RequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "status": "allowed",
        "request": req.as_dict(),
        "universe": svc.universe.get(user_id, req.trader_id),
    }


@router.post("/api/requests/{request_id}/decline")
def decline(
    request: Request,
    request_id: str,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    svc = _service(request)
    try:
        req = svc.decline(user_id, request_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="request not found")
    except RequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "declined", "request": req.as_dict()}
