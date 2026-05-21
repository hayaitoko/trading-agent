"""Boot the web UI with demo accounts pre-seeded.

Invoke via `uv run trading-agent-web`. Accounts persist to accounts.json
in the cwd, secrets to trading_agent_secrets.json. Both are gitignored.

Demo positions are reseeded into MockBroker state every restart since
MockBroker is in-memory only. This goes away when InvestopediaBroker
takes over (it reads state from the actual simulator).
"""
import asyncio
from decimal import Decimal
from pathlib import Path

import uvicorn

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker
from trading_agent.models import Order
from trading_agent.web import AppState, create_app

DEMO_PRICES: dict[str, Decimal] = {
    "AAPL": Decimal("185.50"),
    "NVDA": Decimal("142.30"),
    "TSLA": Decimal("248.10"),
    "GME": Decimal("31.20"),
}


def demo_quote(ticker: str) -> Decimal:
    return DEMO_PRICES.get(ticker, Decimal("100.00"))


async def _seed_aggressive(broker: MockBroker) -> None:
    await broker.place_order(Order(ticker="NVDA", side="buy", qty=20))
    await broker.place_order(Order(ticker="TSLA", side="buy", qty=10))
    await broker.place_order(Order(ticker="GME", side="buy", qty=50))


async def _seed_conservative(broker: MockBroker) -> None:
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=15))


def _make_demo_account(account_id: str, name: str, cash: Decimal) -> Account:
    broker = MockBroker(cash=cash, quote_fn=demo_quote)
    return Account(
        id=account_id,
        name=name,
        broker=broker,
        starting_cash=cash,
    )


def build_demo_state(accounts_path: Path, secrets_path: Path) -> AppState:
    state = AppState(
        accounts_path=accounts_path,
        secrets_path=secrets_path,
        quote_fn=demo_quote,
    )
    state.hydrate()

    if not state.accounts:
        state.add_account(_make_demo_account(
            "paper-aggressive", "Paper Aggressive", Decimal("25000")
        ))
        state.add_account(_make_demo_account(
            "paper-conservative", "Paper Conservative", Decimal("25000")
        ))

    seeders = {
        "paper-aggressive": _seed_aggressive,
        "paper-conservative": _seed_conservative,
    }
    for account_id, seeder in seeders.items():
        if account_id in state.accounts:
            asyncio.run(seeder(state.accounts[account_id].broker))

    return state


def main() -> None:
    state = build_demo_state(
        accounts_path=Path("accounts.json"),
        secrets_path=Path("trading_agent_secrets.json"),
    )
    app = create_app(state)
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
