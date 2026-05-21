import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from trading_agent.accounts import Account
from trading_agent.brokers import MockBroker
from trading_agent.chat import ChatMessage, ChatService
from trading_agent.chat.history import load_history, save_history
from trading_agent.chat.models import MODELS, estimate_tokens, find_model
from trading_agent.chat.tools import execute
from trading_agent.models import Order
from trading_agent.web import AppState, create_app


def _quotes(prices):
    return lambda ticker: prices.get(ticker, Decimal("0"))


def _state(tmp_path, prices=None):
    return AppState(
        accounts_path=tmp_path / "accounts.json",
        secrets_path=tmp_path / "secrets.json",
        quote_fn=_quotes(prices or {}),
    )


def _state_with_account(tmp_path, account_id="acc-1", name="Test Account"):
    """Build state with one empty account. Seed positions yourself."""
    state = _state(tmp_path, {"NVDA": Decimal("140")})
    broker = MockBroker(cash=Decimal("10000"), quote_fn=state.quote_fn)
    state.add_account(Account(
        id=account_id, name=name, broker=broker, starting_cash=Decimal("10000"),
    ))
    return state


async def _seed_one_position(state, account_id="acc-1"):
    broker = state.accounts[account_id].broker
    await broker.place_order(Order(ticker="NVDA", side="buy", qty=10))


def fake_caller(responses):
    iterator = iter(responses)
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return next(iterator)

    return call, calls


def test_chat_message_dict_round_trip():
    m = ChatMessage(role="user", content="hi", images=["data:image/png;base64,xxx"])
    restored = ChatMessage.from_dict(m.to_dict())
    assert restored.role == "user"
    assert restored.content == "hi"
    assert restored.images == ["data:image/png;base64,xxx"]
    assert restored.timestamp == m.timestamp


def test_history_save_and_load(tmp_path):
    path = tmp_path / "h.json"
    msgs = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello", model="anthropic/claude-sonnet-4.6"),
    ]
    save_history(msgs, path)
    loaded = load_history(path)
    assert len(loaded) == 2
    assert loaded[0].content == "hi"
    assert loaded[1].model == "anthropic/claude-sonnet-4.6"


def test_estimate_tokens_uses_3_chars_per_token():
    msgs = [ChatMessage(role="user", content="a" * 300)]
    assert estimate_tokens(msgs) == 100


def test_find_model_returns_spec_for_known_id():
    spec = find_model("anthropic/claude-sonnet-4.6")
    assert spec is not None
    assert spec.is_anthropic
    assert spec.context_limit > 0


def test_find_model_returns_none_for_unknown():
    assert find_model("not-a-real-model") is None


async def test_tool_list_accounts(tmp_path):
    state = _state_with_account(tmp_path)
    await _seed_one_position(state)
    result = await execute(state, "list_accounts", {})
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["id"] == "acc-1"
    assert data[0]["position_count"] == 1


async def test_tool_get_account(tmp_path):
    state = _state_with_account(tmp_path)
    await _seed_one_position(state)
    result = await execute(state, "get_account", {"account_id": "acc-1"})
    data = json.loads(result)
    assert data["id"] == "acc-1"
    assert len(data["positions"]) == 1
    assert data["positions"][0]["ticker"] == "NVDA"


async def test_tool_get_account_unknown_returns_error(tmp_path):
    state = _state(tmp_path)
    result = await execute(state, "get_account", {"account_id": "missing"})
    data = json.loads(result)
    assert "error" in data


async def test_tool_get_trades(tmp_path):
    state = _state_with_account(tmp_path)
    await _seed_one_position(state)
    result = await execute(state, "get_trades", {"account_id": "acc-1", "limit": 5})
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["ticker"] == "NVDA"
    assert data[0]["side"] == "buy"


async def test_tool_unknown_returns_error(tmp_path):
    state = _state(tmp_path)
    result = await execute(state, "made_up_tool", {})
    data = json.loads(result)
    assert "error" in data


async def test_service_send_basic_round_trip(tmp_path):
    state = _state(tmp_path)
    caller, calls = fake_caller([
        {"content": "hi there", "tool_calls": None},
    ])
    service = ChatService(
        state=state,
        history_path=tmp_path / "chat.json",
        model_caller=caller,
    )
    history = await service.send("hello", [], "anthropic/claude-sonnet-4.6")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"
    assert history[1].content == "hi there"
    assert history[1].model == "anthropic/claude-sonnet-4.6"
    assert len(calls) == 1


async def test_service_send_executes_tool_loop(tmp_path):
    state = _state_with_account(tmp_path)
    await _seed_one_position(state)
    caller, calls = fake_caller([
        {
            "content": "",
            "tool_calls": [{
                "id": "tc1",
                "type": "function",
                "function": {"name": "list_accounts", "arguments": "{}"},
            }],
        },
        {"content": "you have 1 account", "tool_calls": None},
    ])
    service = ChatService(
        state=state,
        history_path=tmp_path / "chat.json",
        model_caller=caller,
    )
    history = await service.send("show me", [], "anthropic/claude-sonnet-4.6")
    assert len(history) == 4
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert history[1].tool_calls
    assert history[2].role == "tool"
    assert "acc-1" in history[2].content
    assert history[3].role == "assistant"
    assert history[3].content == "you have 1 account"
    assert len(calls) == 2


async def test_service_persists_history_across_sends(tmp_path):
    state = _state(tmp_path)
    history_path = tmp_path / "chat.json"
    caller1, _ = fake_caller([{"content": "first reply", "tool_calls": None}])
    s1 = ChatService(state, history_path, model_caller=caller1)
    await s1.send("first", [], "anthropic/claude-sonnet-4.6")

    caller2, _ = fake_caller([{"content": "second reply", "tool_calls": None}])
    s2 = ChatService(state, history_path, model_caller=caller2)
    history = await s2.send("second", [], "anthropic/claude-sonnet-4.6")
    assert len(history) == 4
    assert history[0].content == "first"
    assert history[1].content == "first reply"
    assert history[2].content == "second"
    assert history[3].content == "second reply"


async def test_service_reset(tmp_path):
    state = _state(tmp_path)
    history_path = tmp_path / "chat.json"
    caller, _ = fake_caller([{"content": "reply", "tool_calls": None}])
    service = ChatService(state, history_path, model_caller=caller)
    await service.send("hi", [], "anthropic/claude-sonnet-4.6")
    assert history_path.exists()
    service.reset()
    assert not history_path.exists()
    assert service.load() == []


@pytest.fixture
def client_with_fake_caller(tmp_path):
    state = _state_with_account(tmp_path)
    app = create_app(state, chat_history_path=tmp_path / "chat.json")
    caller, calls = fake_caller([{"content": "ok", "tool_calls": None}])
    app.state.chat_service._model_caller = caller
    return TestClient(app), state, calls


def test_route_chat_models(tmp_path):
    state = _state(tmp_path)
    client = TestClient(create_app(state, chat_history_path=tmp_path / "chat.json"))
    response = client.get("/chat/models")
    assert response.status_code == 200
    data = response.json()
    assert data["default"]
    assert any(m["id"] == "anthropic/claude-sonnet-4.6" for m in data["models"])
    assert all("context_limit" in m for m in data["models"])
    assert len(data["models"]) == len(MODELS)


def test_route_chat_history_empty(tmp_path):
    state = _state(tmp_path)
    client = TestClient(create_app(state, chat_history_path=tmp_path / "chat.json"))
    response = client.get("/chat/history")
    assert response.status_code == 200
    assert response.json() == {"messages": [], "tokens": 0}


def test_route_chat_send_persists(client_with_fake_caller):
    client, _state, _ = client_with_fake_caller
    response = client.post("/chat/send", json={
        "text": "hello",
        "images": [],
        "model": "anthropic/claude-sonnet-4.6",
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["content"] == "ok"
    assert data["tokens"] > 0


def test_route_chat_send_empty_rejected(client_with_fake_caller):
    client, _, _ = client_with_fake_caller
    response = client.post("/chat/send", json={
        "text": "", "images": [], "model": "anthropic/claude-sonnet-4.6",
    })
    assert response.status_code == 422


def test_route_chat_reset(client_with_fake_caller):
    client, _, _ = client_with_fake_caller
    client.post("/chat/send", json={
        "text": "hi", "images": [], "model": "anthropic/claude-sonnet-4.6",
    })
    response = client.post("/chat/reset")
    assert response.status_code == 200
    assert response.json() == {"messages": [], "tokens": 0}

    follow_up = client.get("/chat/history")
    assert follow_up.json() == {"messages": [], "tokens": 0}


def test_chat_sidebar_renders_in_page(tmp_path):
    state = _state(tmp_path)
    client = TestClient(create_app(state, chat_history_path=tmp_path / "chat.json"))
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="chat-sidebar"' in response.text
    assert 'id="chat-model"' in response.text
    assert 'id="chat-input"' in response.text
    assert 'id="chat-reset"' in response.text
    assert '/static/chat.js' in response.text


def test_static_chat_js_served(tmp_path):
    state = _state(tmp_path)
    client = TestClient(create_app(state, chat_history_path=tmp_path / "chat.json"))
    response = client.get("/static/chat.js")
    assert response.status_code == 200
    assert "chat-input" in response.text


def test_settings_includes_openrouter_field(tmp_path):
    state = _state(tmp_path)
    client = TestClient(create_app(state, chat_history_path=tmp_path / "chat.json"))
    response = client.get("/settings/")
    assert response.status_code == 200
    assert "openrouter api key" in response.text.lower()


def test_chat_send_fails_clearly_when_no_api_key(tmp_path):
    """Without injecting a fake_caller, real call_model runs and fails on missing key."""
    state = _state(tmp_path)
    app = create_app(state, chat_history_path=tmp_path / "chat.json")
    client = TestClient(app)
    response = client.post("/chat/send", json={
        "text": "hi", "images": [], "model": "anthropic/claude-sonnet-4.6",
    })
    assert response.status_code == 502
    assert "OPENROUTER_API_KEY" in response.json()["detail"]


