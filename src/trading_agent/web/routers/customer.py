"""Customer product UI router: serves the end-user SPA at /app/*.

Mounts the customer-facing pages (Accounts, News-Sources, Memory, Settings,
and the persistent chat dock) as a sub-application at ``/app``.  All API
traffic still goes through the same CONTRACTS routes (``/api/...``);  this
router only handles the page shell and its static HTML.

The SPA is a vanilla-JS single-page app (no build step, inlined dependencies)
served from ``web/static/customer.html``.  All sub-paths under ``/app``
redirect to the shell so browser navigation and refresh work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["customer-ui"])

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_CUSTOMER_HTML = _STATIC_DIR / "customer.html"


def _serve_shell() -> Any:
    return FileResponse(_CUSTOMER_HTML, media_type="text/html")


@router.get("/app", include_in_schema=False)
def customer_root() -> Any:
    return _serve_shell()


@router.get("/app/{path:path}", include_in_schema=False)
def customer_spa(path: str) -> Any:  # noqa: ARG001
    """Catch-all: every /app/* path serves the SPA shell."""
    return _serve_shell()
