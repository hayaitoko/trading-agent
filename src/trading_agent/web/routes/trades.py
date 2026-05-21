from dataclasses import dataclass
from decimal import Decimal

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from trading_agent.accounts import Account
from trading_agent.models import Trade

router = APIRouter(prefix="/trades")


@dataclass
class TradeGroup:
    account_id: str
    account_name: str
    trades: list[Trade]
    total_volume: Decimal


async def _group(account: Account) -> TradeGroup:
    trades = await account.broker.get_trades()
    trades_sorted = sorted(trades, key=lambda t: t.executed_at, reverse=True)
    volume = sum((t.price * t.qty for t in trades), start=Decimal(0))
    return TradeGroup(
        account_id=account.id,
        account_name=account.name,
        trades=trades_sorted,
        total_volume=volume,
    )


@router.get("/", response_class=HTMLResponse)
async def list_trades(request: Request):
    state = request.app.state.app_state
    templates = request.app.state.templates
    groups = [await _group(a) for a in state.accounts.values()]
    return templates.TemplateResponse(
        request,
        "trades.html",
        {"groups": groups, "page": "trades"},
    )
