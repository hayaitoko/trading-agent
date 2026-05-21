import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker
from trading_agent.models import Order
from trading_agent.web import AppState, create_app


def _quote_fn(prices: dict[str, Decimal]):
    return lambda ticker: prices.get(ticker, Decimal("0"))


def _build_state(tmp_path: Path, prices: dict[str, Decimal] | None = None) -> AppState:
    return AppState(
        accounts_path=tmp_path / "accounts.json",
        secrets_path=tmp_path / "secrets.json",
        quote_fn=_quote_fn(prices or {}),
    )


def _client_with_one_position(tmp_path: Path) -> TestClient:
    state = _build_state(tmp_path, {"NVDA": Decimal("140")})
    broker = MockBroker(cash=Decimal("10000"), quote_fn=state.quote_fn)
    asyncio.run(broker.place_order(Order(ticker="NVDA", side="buy", qty=10)))
    state.add_account(Account(
        id="acc-1", name="Test Account", broker=broker, starting_cash=Decimal("10000"),
    ))
    return TestClient(create_app(state))


def test_dashboard_renders_account_name(tmp_path):
    client = _client_with_one_position(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Test Account" in response.text


def test_dashboard_shows_cash_and_position(tmp_path):
    client = _client_with_one_position(tmp_path)
    response = client.get("/")
    assert "8600.00" in response.text  # cash after buying 10 NVDA @ 140
    assert "NVDA" in response.text


def test_dashboard_with_no_accounts(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert response.status_code == 200
    assert "No accounts" in response.text


def test_placeholder_nav_pages_marked_in_nav(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    # Today, Signals, Strategy are placeholders with a status dot
    assert "placeholder until data layer is wired" in response.text


def test_today_placeholder_renders(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/today/")
    assert response.status_code == 200
    assert "Today" in response.text
    assert "Placeholder" in response.text
    assert "v0.1" in response.text


def test_signals_placeholder_renders(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/signals/")
    assert response.status_code == 200
    assert "Signals" in response.text
    assert "v0.3" in response.text


def test_strategy_placeholder_renders_account_names(tmp_path):
    client = _client_with_one_position(tmp_path)
    response = client.get("/strategy/")
    assert response.status_code == 200
    assert "Strategy" in response.text
    assert "Test Account" in response.text
    assert "v0.6" in response.text
    assert "no strategy bound" in response.text


def test_sidebar_has_collapse_button(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert 'id="chat-collapse"' in response.text
    assert 'id="chat-body"' in response.text


def test_dashboard_has_add_account_button(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert "/accounts/" in response.text
    assert "open new account" in response.text


@pytest.fixture
def client(tmp_path):
    state = _build_state(tmp_path, {"AAPL": Decimal("100")})
    return TestClient(create_app(state)), state


def test_accounts_page_empty(client):
    c, _ = client
    response = c.get("/accounts/")
    assert response.status_code == 200
    assert "No accounts" in response.text


def test_create_account_via_form(client):
    c, state = client
    response = c.post("/accounts/", data={"name": "Paper Momentum", "starting_cash": "50000"})
    assert response.status_code == 200
    assert "Paper Momentum" in response.text
    assert "paper-momentum" in state.accounts
    assert state.accounts["paper-momentum"].starting_cash == Decimal("50000")
    # Defaults to the default account model when not specified.
    assert state.accounts["paper-momentum"].model == "anthropic/claude-sonnet-4.6"


def test_create_account_with_chosen_model(client):
    c, state = client
    response = c.post("/accounts/", data={
        "name": "Opus Trader",
        "starting_cash": "10000",
        "model": "anthropic/claude-opus-4.7",
    })
    assert response.status_code == 200
    assert state.accounts["opus-trader"].model == "anthropic/claude-opus-4.7"


def test_create_account_unknown_model_rejected(client):
    c, _ = client
    response = c.post("/accounts/", data={
        "name": "Bad",
        "starting_cash": "1000",
        "model": "openai/gpt-4-not-real",
    })
    assert response.status_code == 422


def test_accounts_page_has_model_picker(client):
    c, _ = client
    response = c.get("/accounts/")
    assert response.status_code == 200
    assert 'name="model"' in response.text
    assert "Claude Sonnet 4.6" in response.text  # at least one model option


def test_eval_page_empty(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/eval/")
    assert response.status_code == 200
    assert "Evaluation" in response.text
    assert "No accounts" in response.text


def test_eval_page_ranks_accounts(tmp_path):
    import asyncio

    from trading_agent.accounts import Account
    from trading_agent.models import Order

    state = _build_state(tmp_path, {"NVDA": Decimal("200")})
    # Account A: starts with 1000, buys low — gains
    broker_a = MockBroker(cash=Decimal("1000"), quote_fn=state.quote_fn)
    state.add_account(Account(
        id="winner", name="Winner", broker=broker_a,
        starting_cash=Decimal("1000"), model="anthropic/claude-opus-4.7",
    ))
    # Account B: same start, no trades — flat
    broker_b = MockBroker(cash=Decimal("1000"), quote_fn=state.quote_fn)
    state.add_account(Account(
        id="flat", name="Flat", broker=broker_b,
        starting_cash=Decimal("1000"), model="anthropic/claude-haiku-4.5",
    ))
    # Winner buys 5 shares at $200, NVDA jumps to $250 (we change the quote)
    asyncio.run(broker_a.place_order(Order(ticker="NVDA", side="buy", qty=5)))
    state.quote_fn = lambda t: Decimal("250") if t == "NVDA" else Decimal("0")
    # Re-wire brokers to use the new quote_fn
    broker_a._quote_fn = state.quote_fn
    broker_b._quote_fn = state.quote_fn

    client = TestClient(create_app(state))
    response = client.get("/eval/")
    assert response.status_code == 200
    body = response.text
    # Winner appears before Flat in the table
    assert body.index("Winner") < body.index("Flat")
    # Model labels appear
    assert "Claude Opus 4.7" in body
    assert "Claude Haiku 4.5" in body


def test_eval_appears_in_nav(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert 'href="/eval/"' in response.text
    assert "Evaluation" in response.text


def test_theme_toggle_present_in_header(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert 'id="theme-toggle"' in response.text
    # Both sun and moon icons emitted; CSS hides one based on data-theme.
    assert 'data-lucide="sun"' in response.text
    assert 'data-lucide="moon"' in response.text


def test_chat_sidebar_uses_panel_icons(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state))
    response = client.get("/")
    assert 'data-lucide="panel-left-close"' in response.text
    assert 'data-lucide="panel-left-open"' in response.text


def test_netlog_records_inbound_requests(tmp_path):
    state = _build_state(tmp_path)
    app = create_app(state, start_consolidator=False)
    client = TestClient(app)
    # Hit a real route — middleware should record it.
    client.get("/")
    client.get("/accounts/")
    log = app.state.netlog.snapshot()
    paths = [e["target"] for e in log]
    assert "/" in paths
    assert "/accounts/" in paths
    for entry in log:
        assert entry["direction"] == "in"
        assert entry["status"] == 200
        assert entry["duration_ms"] >= 0


def test_netlog_endpoint_returns_entries(tmp_path):
    state = _build_state(tmp_path)
    app = create_app(state, start_consolidator=False)
    client = TestClient(app)
    client.get("/")  # generate at least one entry
    response = client.get("/settings/api/netlog")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_netlog_skips_static_and_itself(tmp_path):
    state = _build_state(tmp_path)
    app = create_app(state, start_consolidator=False)
    client = TestClient(app)
    client.get("/static/chat.js")  # static — should not record
    client.get("/settings/api/netlog")  # the netlog endpoint itself — should not record
    client.get("/")  # this should record
    log = app.state.netlog.snapshot()
    paths = [e["target"] for e in log]
    assert not any(p.startswith("/static") for p in paths)
    assert "/settings/api/netlog" not in paths
    assert "/" in paths


def test_settings_page_has_netlog_section(tmp_path):
    state = _build_state(tmp_path)
    client = TestClient(create_app(state, start_consolidator=False))
    response = client.get("/settings/")
    assert response.status_code == 200
    assert 'id="netlog-section"' in response.text
    assert "network log" in response.text.lower()


def test_create_account_persists_to_disk(client):
    c, state = client
    c.post("/accounts/", data={"name": "Aggro", "starting_cash": "1000"})
    assert state.accounts_path.exists()
    raw = state.accounts_path.read_text()
    assert "Aggro" in raw
    assert "1000" in raw


def test_create_account_rejects_zero_cash(client):
    c, _ = client
    response = c.post("/accounts/", data={"name": "X", "starting_cash": "0"})
    assert response.status_code == 422


def test_create_account_unique_ids_on_name_collision(client):
    c, state = client
    c.post("/accounts/", data={"name": "Test", "starting_cash": "100"})
    c.post("/accounts/", data={"name": "Test", "starting_cash": "100"})
    assert "test" in state.accounts
    assert "test-2" in state.accounts


def test_toggle_account(client):
    c, state = client
    c.post("/accounts/", data={"name": "Toggle Me", "starting_cash": "100"})
    assert state.accounts["toggle-me"].enabled is True

    response = c.post("/accounts/toggle-me/toggle")
    assert response.status_code == 200
    assert state.accounts["toggle-me"].enabled is False
    assert "paused" in response.text


def test_delete_account(client):
    c, state = client
    c.post("/accounts/", data={"name": "Doomed", "starting_cash": "100"})
    assert "doomed" in state.accounts

    response = c.delete("/accounts/doomed")
    assert response.status_code == 200
    assert "doomed" not in state.accounts


def test_delete_unknown_account_404(client):
    c, _ = client
    response = c.delete("/accounts/nonexistent")
    assert response.status_code == 404


def test_hydrate_loads_persisted_accounts(tmp_path):
    state1 = _build_state(tmp_path, {"AAPL": Decimal("100")})
    c1 = TestClient(create_app(state1))
    c1.post("/accounts/", data={"name": "Persistent", "starting_cash": "777"})

    state2 = _build_state(tmp_path, {"AAPL": Decimal("100")})
    state2.hydrate()
    assert "persistent" in state2.accounts
    assert state2.accounts["persistent"].starting_cash == Decimal("777")


def test_settings_page_renders(client):
    c, _ = client
    response = c.get("/settings/")
    assert response.status_code == 200
    assert "reddit client id" in response.text
    assert "investopedia username" in response.text


def test_settings_save_persists(client):
    c, state = client
    response = c.post("/settings/", data={
        "reddit_client_id": "abc123",
        "reddit_client_secret": "shh",
        "reddit_user_agent": "trading-agent/test",
        "stocktwits_token": "tok",
        "investopedia_username": "lukas",
        "investopedia_password": "hunter2",
    })
    assert response.status_code == 200
    assert "saved" in response.text
    assert state.secrets["reddit_client_id"] == "abc123"
    assert state.secrets["reddit_user_agent"] == "trading-agent/test"
    assert state.secrets_path.exists()


def test_settings_blank_secret_keeps_existing(client):
    c, state = client
    c.post("/settings/", data={
        "reddit_client_id": "first",
        "reddit_client_secret": "keep-me",
        "reddit_user_agent": "ua",
        "stocktwits_token": "",
        "investopedia_username": "",
        "investopedia_password": "",
    })
    c.post("/settings/", data={
        "reddit_client_id": "second",
        "reddit_client_secret": "",
        "reddit_user_agent": "ua2",
        "stocktwits_token": "",
        "investopedia_username": "",
        "investopedia_password": "",
    })
    assert state.secrets["reddit_client_id"] == "second"
    assert state.secrets["reddit_client_secret"] == "keep-me"


def test_settings_redacts_secrets_in_response(client):
    c, _ = client
    c.post("/settings/", data={
        "reddit_client_id": "visible-id",
        "reddit_client_secret": "supersecret",
        "reddit_user_agent": "ua",
        "stocktwits_token": "",
        "investopedia_username": "",
        "investopedia_password": "",
    })
    response = c.get("/settings/")
    assert "visible-id" in response.text  # non-secret shown
    assert "supersecret" not in response.text  # secret redacted


def test_trades_page_empty(client):
    c, _ = client
    response = c.get("/trades/")
    assert response.status_code == 200
    assert "No accounts" in response.text


def test_trades_page_shows_trades(tmp_path):
    state = _build_state(tmp_path, {"NVDA": Decimal("140")})
    broker = MockBroker(cash=Decimal("10000"), quote_fn=state.quote_fn)
    asyncio.run(broker.place_order(Order(ticker="NVDA", side="buy", qty=5)))
    state.add_account(Account(
        id="t", name="Trader", broker=broker, starting_cash=Decimal("10000"),
    ))
    c = TestClient(create_app(state))

    response = c.get("/trades/")
    assert response.status_code == 200
    assert "Trader" in response.text
    assert "NVDA" in response.text
    assert "buy" in response.text
