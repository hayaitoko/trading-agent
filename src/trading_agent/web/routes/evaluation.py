"""Evaluation page — the model-comparison harness view.

Frames the agent as a built-in A/B test: multiple accounts, multiple models,
the same signal feed, ranked head-to-head. Right now we can compute trivial
P&L vs starting cash; richer metrics (Sharpe, max drawdown, win rate) light
up when historical snapshots and closed-trade tracking ship.
"""
from dataclasses import dataclass
from decimal import Decimal

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from trading_agent.accounts import Account
from trading_agent.chat.models import find_model

router = APIRouter(prefix="/eval")


@dataclass
class EvalRow:
    rank: int
    id: str
    name: str
    enabled: bool
    model: str
    model_display: str
    starting_cash: Decimal
    total: Decimal
    pnl_abs: Decimal
    pnl_pct: Decimal
    trades: int


async def _build_row(account: Account) -> tuple[Decimal, EvalRow]:
    cash = await account.broker.get_cash()
    positions = await account.broker.get_positions()
    equity = sum((p.qty * p.current_price for p in positions), start=Decimal(0))
    total = cash + equity
    pnl_abs = total - account.starting_cash
    pnl_pct = (
        pnl_abs / account.starting_cash * Decimal(100)
        if account.starting_cash else Decimal(0)
    )
    trades = len(await account.broker.get_trades())
    spec = find_model(account.model)
    row = EvalRow(
        rank=0,
        id=account.id,
        name=account.name,
        enabled=account.enabled,
        model=account.model,
        model_display=spec.display if spec else account.model,
        starting_cash=account.starting_cash,
        total=total,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        trades=trades,
    )
    return pnl_abs, row


@router.get("/", response_class=HTMLResponse)
async def evaluation(request: Request):
    state = request.app.state.app_state
    pairs = [await _build_row(a) for a in state.accounts.values()]
    pairs.sort(key=lambda p: p[0], reverse=True)
    rows = []
    for i, (_, row) in enumerate(pairs, start=1):
        row.rank = i
        rows.append(row)
    return request.app.state.templates.TemplateResponse(
        request,
        "evaluation.html",
        {"rows": rows, "page": "eval"},
    )
