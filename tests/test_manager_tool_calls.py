"""PHASE 8: ManagerAgent tool-calling.

Covers the JSON envelope parser (``parse_tool_response``), the new
``chat_with_actions`` return shape, the HTTP surface that exposes
``actions`` alongside ``reply``, and the backwards-compatible behavior of the
existing ``chat()``/``/api/chat`` paths when the model just chats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.config.endpoints import ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.manager.agent import (
    TOOL_PROMPT,
    ChatTurn,
    ManagerAgent,
    parse_tool_response,
)
from trading_agent.manager.chat import ConversationStore
from trading_agent.web.app import create_cockpit_app

# --- parse_tool_response (pure unit tests) ----------------------------------


def test_parse_plain_text_returns_no_actions() -> None:
    reply, actions = parse_tool_response("Plain answer — no JSON here.")
    assert reply == "Plain answer — no JSON here."
    assert actions == []


def test_parse_json_envelope_extracts_actions() -> None:
    payload = json.dumps(
        {
            "reply": "Pulling up Apple.",
            "actions": [{"type": "open_quote", "symbol": "AAPL"}],
        }
    )
    reply, actions = parse_tool_response(payload)
    assert reply == "Pulling up Apple."
    assert actions == [{"type": "open_quote", "symbol": "AAPL"}]


def test_parse_strips_code_fence() -> None:
    fenced = "```json\n" + json.dumps(
        {"reply": "ok", "actions": [{"type": "open_tab", "tab": "research"}]}
    ) + "\n```"
    reply, actions = parse_tool_response(fenced)
    assert reply == "ok"
    assert actions == [{"type": "open_tab", "tab": "research"}]


def test_parse_drops_unknown_action_types() -> None:
    payload = json.dumps(
        {
            "reply": "filtered",
            "actions": [
                {"type": "open_quote", "symbol": "AAPL"},
                {"type": "place_trade", "symbol": "TSLA"},  # forbidden
                {"type": "rm-rf-broker"},
            ],
        }
    )
    _, actions = parse_tool_response(payload)
    assert actions == [{"type": "open_quote", "symbol": "AAPL"}]


def test_parse_drops_action_missing_required_key() -> None:
    payload = json.dumps(
        {
            "reply": "x",
            "actions": [{"type": "open_account"}, {"type": "open_quote", "symbol": ""}],
        }
    )
    _, actions = parse_tool_response(payload)
    assert actions == []


def test_parse_caps_actions_at_three() -> None:
    payload = json.dumps(
        {
            "reply": "many",
            "actions": [
                {"type": "open_quote", "symbol": f"S{i}"} for i in range(10)
            ],
        }
    )
    _, actions = parse_tool_response(payload)
    assert len(actions) == 3


def test_parse_malformed_json_falls_back_to_plain() -> None:
    # Note: must look like an object (starts with `{` and ends with `}`) so the
    # parser actually attempts json.loads; a real trailing `}` keeps the shape
    # valid-looking while the content is broken.
    bad = '{"reply": "broken", actions:}'
    reply, actions = parse_tool_response(bad)
    assert reply == bad
    assert actions == []


def test_parse_dict_without_reply_falls_back() -> None:
    payload = json.dumps({"actions": [{"type": "open_quote", "symbol": "AAPL"}]})
    reply, actions = parse_tool_response(payload)
    # No `reply` key → treated as plain text; actions ignored.
    assert reply == payload
    assert actions == []


def test_parse_brace_inside_sentence_not_misparsed() -> None:
    text = "The watchlist {AAPL, MSFT} looks healthy."
    reply, actions = parse_tool_response(text)
    assert reply == text
    assert actions == []


def test_parse_open_tab_id_alias_preserved() -> None:
    payload = json.dumps(
        {"reply": "go", "actions": [{"type": "open_tab", "tab": "x", "id": "x"}]}
    )
    _, actions = parse_tool_response(payload)
    assert actions == [{"type": "open_tab", "tab": "x", "id": "x"}]


# --- ManagerAgent.chat / chat_with_actions ----------------------------------


@dataclass
class _ChatResult:
    content: str
    model: str = "m"
    usage: dict[str, Any] | None = None
    cost: float | None = 0.001


class _FakeRegistry:
    """Stand-in for EndpointRegistry.chat: returns a fixed content."""

    def __init__(self, content: str = "hello") -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        user_id: str,
        ref: ModelRef,
        messages: list[dict[str, str]],
        **opts: Any,
    ) -> _ChatResult:
        self.calls.append({"user_id": user_id, "messages": messages, "opts": opts})
        return _ChatResult(content=self._content)


@pytest.fixture
def agent_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "c.db")


def test_chat_with_actions_returns_chat_turn(agent_db: Database) -> None:
    reg = _FakeRegistry(
        content=json.dumps(
            {"reply": "opening AAPL", "actions": [{"type": "open_quote", "symbol": "AAPL"}]}
        )
    )
    store = ConversationStore(agent_db)
    agent = ManagerAgent(reg, SettingsStore(agent_db), store)
    conv = store.create("u1")
    turn = agent.chat_with_actions("u1", conv.id, "show me apple", ModelRef("e", "m"))
    assert isinstance(turn, ChatTurn)
    assert turn.reply == "opening AAPL"
    assert turn.actions == [{"type": "open_quote", "symbol": "AAPL"}]


def test_chat_returns_plain_string_for_backwards_compat(agent_db: Database) -> None:
    reg = _FakeRegistry(content="just chatting")
    store = ConversationStore(agent_db)
    agent = ManagerAgent(reg, SettingsStore(agent_db), store)
    conv = store.create("u1")
    reply = agent.chat("u1", conv.id, "hi", ModelRef("e", "m"))
    assert reply == "just chatting"
    # And system prompt now teaches tool-calling.
    system = reg.calls[0]["messages"][0]["content"]
    assert TOOL_PROMPT.strip().split("\n", 1)[0] in system


# --- HTTP /api/chat shape ----------------------------------------------------


def _mock_transport(content: str, captured: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"body": json.loads(request.content) if request.content else {}})
        return httpx.Response(
            200,
            json={
                "model": "x",
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 9, "cost": 0.001},
            },
        )

    return httpx.MockTransport(handler)


def _make_http(tmp_path: Path, content: str, captured: list[dict[str, Any]]) -> TestClient:
    app = create_cockpit_app(Database(tmp_path / "c.db"), transport=_mock_transport(content, captured))
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "ada", "password": "pw"})
    c.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "k"})
    return c


def test_http_chat_returns_actions_when_model_emits_envelope(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "reply": "Opening Apple for you.",
            "actions": [{"type": "open_quote", "symbol": "AAPL"}],
        }
    )
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, payload, captured)
    r = c.post("/api/chat", json={"message": "show me apple"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "Opening Apple for you."
    assert body["actions"] == [{"type": "open_quote", "symbol": "AAPL"}]


def test_http_chat_empty_actions_when_plain_text(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, "just-chatting", captured)
    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "just-chatting"
    assert body["actions"] == []


def test_http_chat_persists_only_reply_text(tmp_path: Path) -> None:
    """The stored assistant turn is the reply text, not the JSON envelope."""
    payload = json.dumps(
        {
            "reply": "Done.",
            "actions": [{"type": "open_tab", "tab": "research"}],
        }
    )
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, payload, captured)
    body = c.post("/api/chat", json={"message": "open research"}).json()
    cid = body["conversation_id"]
    # Send a follow-up — the prior assistant turn is replayed into the messages.
    c.post("/api/chat", json={"message": "now what", "conversation_id": cid})
    last_call = captured[-1]["body"]["messages"]
    assistant_turns = [m for m in last_call if m["role"] == "assistant"]
    assert assistant_turns, "expected the prior reply to be replayed"
    # The stored content is the user-visible reply, never the raw JSON envelope.
    assert assistant_turns[-1]["content"] == "Done."
