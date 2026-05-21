import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker

router = APIRouter(prefix="/accounts")

_SLUG_RE = re.compile(r"[^a-z0-9-]")


@dataclass
class AccountRow:
    id: str
    name: str
    enabled: bool
    starting_cash: Decimal


def _rows(state) -> list[AccountRow]:
    return [
        AccountRow(id=a.id, name=a.name, enabled=a.enabled, starting_cash=a.starting_cash)
        for a in state.accounts.values()
    ]


def _slugify(name: str) -> str:
    base = _SLUG_RE.sub("", name.lower().replace(" ", "-")).strip("-")
    return base or "account"


def _unique_id(state, base: str) -> str:
    if base not in state.accounts:
        return base
    i = 2
    while f"{base}-{i}" in state.accounts:
        i += 1
    return f"{base}-{i}"


@router.get("/", response_class=HTMLResponse)
async def list_accounts(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "accounts.html",
        {"rows": _rows(request.app.state.app_state), "page": "accounts"},
    )


@router.post("/", response_class=HTMLResponse)
async def create_account(
    request: Request,
    name: str = Form(...),
    starting_cash: str = Form(...),
):
    state = request.app.state.app_state
    templates = request.app.state.templates

    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="name is required")
    try:
        cash = Decimal(starting_cash)
    except InvalidOperation as e:
        raise HTTPException(status_code=422, detail="starting_cash must be a number") from e
    if cash <= 0:
        raise HTTPException(status_code=422, detail="starting_cash must be positive")

    account_id = _unique_id(state, _slugify(clean_name))
    broker = MockBroker(cash=cash, quote_fn=state.quote_fn)
    state.add_account(Account(
        id=account_id,
        name=clean_name,
        broker=broker,
        starting_cash=cash,
    ))

    return templates.TemplateResponse(
        request, "_accounts_table.html", {"rows": _rows(state)}
    )


@router.delete("/{account_id}", response_class=HTMLResponse)
async def delete_account(request: Request, account_id: str):
    state = request.app.state.app_state
    templates = request.app.state.templates
    if account_id not in state.accounts:
        raise HTTPException(status_code=404, detail="account not found")
    state.remove_account(account_id)
    return templates.TemplateResponse(
        request, "_accounts_table.html", {"rows": _rows(state)}
    )


@router.post("/{account_id}/toggle", response_class=HTMLResponse)
async def toggle_account(request: Request, account_id: str):
    state = request.app.state.app_state
    templates = request.app.state.templates
    if account_id not in state.accounts:
        raise HTTPException(status_code=404, detail="account not found")
    account = state.toggle_account(account_id)
    return templates.TemplateResponse(
        request,
        "_account_row.html",
        {"row": AccountRow(
            id=account.id, name=account.name,
            enabled=account.enabled, starting_cash=account.starting_cash,
        )},
    )
