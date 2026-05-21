"""Run the web UI with two demo accounts so the dashboard has something to show.

Invoke via `uv run trading-agent-web`. Mutates state synchronously at startup
(asyncio.run) so the demo trades are visible on first page load.
"""
import asyncio
from decimal import Decimal

import uvicorn

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker
from trading_agent.models import Order
from trading_agent.web import AppState, create_app


def _quotes():
    prices = {
        "AAPL": Decimal("185.50"),
        "NVDA": Decimal("142.30"),
        "TSLA": Decimal("248.10"),
        "GME": Decimal("31.20"),
    }
    return lambda ticker: prices[ticker]


async def _seed_aggressive(broker: MockBroker) -> None:
    await broker.place_order(Order(ticker="NVDA", side="buy", qty=20))
    await broker.place_order(Order(ticker="TSLA", side="buy", qty=10))
    await broker.place_order(Order(ticker="GME", side="buy", qty=50))


async def _seed_conservative(broker: MockBroker) -> None:
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=15))


def build_demo_state() -> AppState:
    state = AppState()

    aggressive_broker = MockBroker(cash=Decimal("25000"), quote_fn=_quotes())
    asyncio.run(_seed_aggressive(aggressive_broker))
    state.add_account(Account(
        id="paper-aggressive",
        name="Paper Aggressive",
        broker=aggressive_broker,
    ))

    conservative_broker = MockBroker(cash=Decimal("25000"), quote_fn=_quotes())
    asyncio.run(_seed_conservative(conservative_broker))
    state.add_account(Account(
        id="paper-conservative",
        name="Paper Conservative",
        broker=conservative_broker,
    ))

    return state


def main() -> None:
    app = create_app(build_demo_state())
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
