from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from trading_agent.web.persistence import SECRET_KEYS

router = APIRouter(prefix="/settings")


def _is_secret(key: str) -> bool:
    return any(token in key for token in ("secret", "token", "password"))


def _redact(value: str) -> str:
    if not value:
        return ""
    return "*" * min(len(value), 12)


def _view(secrets: dict[str, str]) -> list[dict]:
    fields = []
    for key in SECRET_KEYS:
        raw = secrets.get(key, "")
        fields.append({
            "key": key,
            "label": key.replace("_", " "),
            "is_secret": _is_secret(key),
            "has_value": bool(raw),
            "redacted": _redact(raw) if _is_secret(key) else raw,
            "value": raw if not _is_secret(key) else "",
        })
    return fields


@router.get("/", response_class=HTMLResponse)
async def get_settings(request: Request):
    state = request.app.state.app_state
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"fields": _view(state.secrets), "saved": False, "page": "settings"},
    )


@router.post("/", response_class=HTMLResponse)
async def save_settings(request: Request):
    state = request.app.state.app_state
    templates = request.app.state.templates
    form = await request.form()
    updates: dict[str, str] = {}
    for key in SECRET_KEYS:
        if key not in form:
            continue
        value = str(form[key])
        # An empty submission for a secret keeps the existing value. This lets
        # the form re-render redacted without forcing the user to re-enter it.
        if _is_secret(key) and not value:
            continue
        updates[key] = value
    state.update_secrets(updates)
    return templates.TemplateResponse(
        request,
        "_settings_form.html",
        {"fields": _view(state.secrets), "saved": True},
    )


@router.get("/api/netlog")
async def api_netlog(request: Request, limit: int = Query(50, ge=1, le=200)):
    netlog = request.app.state.netlog
    return JSONResponse({"entries": netlog.snapshot(limit=limit)})
