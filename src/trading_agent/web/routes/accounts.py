import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from trading_agent.accounts import DEFAULT_ACCOUNT_MODEL, Account
from trading_agent.brokers import MockBroker
from trading_agent.chat.models import MODELS, find_model

router = APIRouter(prefix="/accounts")

_SLUG_RE = re.compile(r"[^a-z0-9-]")


@dataclass
class AccountRow:
    id: str
    name: str
    enabled: bool
    starting_cash: Decimal
    model: str
    model_display: str


def _model_display(model_id: str) -> str:
    spec = find_model(model_id)
    return spec.display if spec else model_id


def _rows(state) -> list[AccountRow]:
    return [
        AccountRow(
            id=a.id,
            name=a.name,
            enabled=a.enabled,
            starting_cash=a.starting_cash,
            model=a.model,
            model_display=_model_display(a.model),
        )
        for a in state.accounts.values()
    ]


def _model_options() -> list[dict]:
    return [{"id": m.id, "display": m.display} for m in MODELS]


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
        {
            "rows": _rows(request.app.state.app_state),
            "models": _model_options(),
            "default_model": DEFAULT_ACCOUNT_MODEL,
            "page": "accounts",
        },
    )


@router.post("/", response_class=HTMLResponse)
async def create_account(
    request: Request,
    name: str = Form(...),
    starting_cash: str = Form(...),
    model: str = Form(DEFAULT_ACCOUNT_MODEL),
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
    if find_model(model) is None:
        raise HTTPException(status_code=422, detail=f"unknown model: {model}")

    account_id = _unique_id(state, _slugify(clean_name))
    broker = MockBroker(cash=cash, quote_fn=state.quote_fn)
    state.add_account(Account(
        id=account_id,
        name=clean_name,
        broker=broker,
        starting_cash=cash,
        model=model,
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
            id=account.id,
            name=account.name,
            enabled=account.enabled,
            starting_cash=account.starting_cash,
            model=account.model,
            model_display=_model_display(account.model),
        )},
    )
