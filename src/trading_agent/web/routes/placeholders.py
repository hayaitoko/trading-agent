"""Routes for pages whose data layers don't exist yet.

These render the structural shape of each page plus a status banner so the
nav doesn't have to hide them behind disabled labels. Replace the placeholder
templates with real data binding when the corresponding engine layer lands.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/today/", response_class=HTMLResponse)
async def today(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "today.html", {"page": "today"}
    )


@router.get("/signals/", response_class=HTMLResponse)
async def signals(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "signals.html", {"page": "signals"}
    )


@router.get("/strategy/", response_class=HTMLResponse)
async def strategy(request: Request):
    state = request.app.state.app_state
    return request.app.state.templates.TemplateResponse(
        request,
        "strategy.html",
        {"page": "strategy", "accounts": list(state.accounts.values())},
    )
