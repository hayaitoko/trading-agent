from decimal import Decimal

from fastapi.testclient import TestClient

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker
from trading_agent.models import Order
from trading_agent.web import AppState, create_app


def _quotes(prices: dict[str, Decimal]):
    return lambda ticker: prices[ticker]


def _build_app_with_one_position() -> TestClient:
    broker = MockBroker(cash=Decimal("10000"), quote_fn=_quotes({"NVDA": Decimal("140")}))

    import asyncio
    asyncio.run(broker.place_order(Order(ticker="NVDA", side="buy", qty=10)))

    state = AppState()
    state.add_account(Account(id="acc-1", name="Test Account", broker=broker))
    return TestClient(create_app(state))


def test_dashboard_renders_account_name():
    client = _build_app_with_one_position()
    response = client.get("/")
    assert response.status_code == 200
    assert "Test Account" in response.text


def test_dashboard_shows_cash_and_position():
    client = _build_app_with_one_position()
    response = client.get("/")
    assert "8600.00" in response.text  # cash after buying 10 NVDA @ 140
    assert "NVDA" in response.text


def test_dashboard_with_no_accounts():
    state = AppState()
    client = TestClient(create_app(state))
    response = client.get("/")
    assert response.status_code == 200
    assert "No accounts configured" in response.text


def test_disabled_nav_pages_marked_not_clickable():
    state = AppState()
    client = TestClient(create_app(state))
    response = client.get("/")
    assert "not built yet" in response.text
