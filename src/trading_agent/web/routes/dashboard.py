from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from trading_agent.accounts import Account
from trading_agent.models import Position, Trade

router = APIRouter()


@dataclass
class AccountView:
    id: str
    name: str
    enabled: bool
    cash: Decimal
    equity: Decimal
    total: Decimal
    positions: list[Position]
    trades_today: list[Trade]


async def _build_view(account: Account) -> AccountView:
    cash = await account.broker.get_cash()
    positions = await account.broker.get_positions()
    equity = sum((p.qty * p.current_price for p in positions), start=Decimal(0))
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    trades_today = await account.broker.get_trades(since=today_start)
    return AccountView(
        id=account.id,
        name=account.name,
        enabled=account.enabled,
        cash=cash,
        equity=equity,
        total=cash + equity,
        positions=positions,
        trades_today=trades_today,
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    state = request.app.state.app_state
    templates = request.app.state.templates
    views = [await _build_view(a) for a in state.accounts.values()]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"accounts": views, "page": "dashboard"},
    )
