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


# --- Full end-to-end proving tests -----------------------------------------------
# These tests prove the complete path:
#   operator message → registry → HTTP client (mocked) → parse_tool_response → actions
#
# Design note: the manager uses a *text-shim* approach to "tool calling":
# TOOL_PROMPT embeds the action schema in the system message, instructing the
# model to return ``{"reply": "...", "actions": [...]}`` as plain text when it
# wants to surface a dashboard action.  There are NO ``tools`` parameters sent
# to the LLM API — the model learns the format from the prompt.
#
# This is intentional: it works with every OpenAI-compatible provider (including
# local Ollama) without requiring native tool-calling support, and the action
# types are a closed read-only set (open_quote/open_chart/open_account/open_tab)
# that carry zero execution risk.  The ``parse_tool_response`` function is the
# single source of truth for validating and normalising the model's output.


def test_full_path_tool_prompt_in_request_and_action_in_response(tmp_path: Path) -> None:
    """Prove the complete manager tool-calling path end-to-end.

    Given: the mocked LLM endpoint returns a JSON envelope with an action.
    When:  POST /api/chat is called.
    Then:
      1. The actual HTTP request body sent to the LLM includes ``TOOL_PROMPT``
         text in the system message — proving the model received its instruction
         sheet, not just a silent client-side shim.
      2. The parsed action appears in the API response ``actions`` list —
         proving the text envelope flows from LLM response → parse_tool_response
         → ChatTurn → router → JSON output without any intermediate intent shim
         inventing the action.
    """
    envelope = json.dumps(
        {
            "reply": "Here is the Apple chart.",
            "actions": [{"type": "open_quote", "symbol": "AAPL"}],
        }
    )
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, envelope, captured)

    r = c.post("/api/chat", json={"message": "show me the Apple chart"})
    assert r.status_code == 200, r.text

    # 1. The request body that reached the (mocked) LLM endpoint must contain
    #    TOOL_PROMPT in the system message — the LLM instructed, not bypassed.
    assert captured, "no HTTP request was captured; EndpointRegistry never called the client"
    llm_messages = captured[0]["body"]["messages"]
    system_msgs = [m for m in llm_messages if m.get("role") == "system"]
    assert system_msgs, "no system message in LLM request"
    system_content = system_msgs[0]["content"]
    # The opening line of TOOL_PROMPT anchors the match; avoids full-string
    # fragility while still confirming the section is present.
    tool_prompt_anchor = TOOL_PROMPT.strip().split("\n", 1)[0]
    assert tool_prompt_anchor in system_content, (
        f"TOOL_PROMPT not found in system message sent to LLM.\n"
        f"Expected anchor: {tool_prompt_anchor!r}\n"
        f"System content (first 500 chars): {system_content[:500]!r}"
    )

    # 2. The API response carries the action the LLM returned — no shim invented it.
    body = r.json()
    assert body["reply"] == "Here is the Apple chart."
    assert body["actions"] == [{"type": "open_quote", "symbol": "AAPL"}], (
        f"Expected open_quote action from LLM envelope but got: {body['actions']!r}"
    )


def test_full_path_plain_text_response_yields_empty_actions(tmp_path: Path) -> None:
    """When the mocked LLM returns plain text (no envelope), actions is empty.

    Proves the fallback path: if the model ignores TOOL_PROMPT and replies as
    plain prose, the router returns ``actions: []`` rather than crashing or
    leaking a partial parse.
    """
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, "All books are performing well today.", captured)

    r = c.post("/api/chat", json={"message": "how are things?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "All books are performing well today."
    assert body["actions"] == []
    # TOOL_PROMPT still reached the LLM — the model *could* have surfaced an action
    assert captured
    system_content = next(
        m["content"] for m in captured[0]["body"]["messages"] if m.get("role") == "system"
    )
    assert TOOL_PROMPT.strip().split("\n", 1)[0] in system_content


def test_full_path_multiple_valid_actions_all_surfaced(tmp_path: Path) -> None:
    """When the LLM emits multiple actions, all valid ones pass through."""
    envelope = json.dumps(
        {
            "reply": "Here are AAPL and TSLA, and the research tab.",
            "actions": [
                {"type": "open_quote", "symbol": "AAPL"},
                {"type": "open_chart", "symbol": "TSLA"},
                {"type": "open_tab", "tab": "research"},
            ],
        }
    )
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, envelope, captured)

    r = c.post("/api/chat", json={"message": "show me apple, tsla, and research"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["actions"]) == 3
    types = [a["type"] for a in body["actions"]]
    assert "open_quote" in types
    assert "open_chart" in types
    assert "open_tab" in types


def test_full_path_unknown_action_type_stripped_by_parse(tmp_path: Path) -> None:
    """An unknown action type from the LLM is silently dropped — no junk reaches the UI."""
    envelope = json.dumps(
        {
            "reply": "Placing your order.",
            "actions": [
                {"type": "place_trade", "symbol": "AAPL"},   # forbidden — drops
                {"type": "open_account", "name": "opus"},    # valid — passes
            ],
        }
    )
    captured: list[dict[str, Any]] = []
    c = _make_http(tmp_path, envelope, captured)

    r = c.post("/api/chat", json={"message": "buy apple in opus"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Only the allowed action type survives; the forbidden one is dropped.
    assert body["actions"] == [{"type": "open_account", "name": "opus"}]
    # The reply still arrives intact.
    assert body["reply"] == "Placing your order."


def test_fake_registry_action_path_proves_agent_not_shim(agent_db: Database) -> None:
    """Unit-level proof: ManagerAgent maps a mocked LLM tool call to an action.

    Uses _FakeRegistry so only the agent logic is under test (no HTTP stack).
    If the agent were merely running a client-side shim (fabricating actions
    without involving the LLM response), this test would pass even with a
    registry that returns plain text — but the action would come from nowhere.
    Here the registry returns the JSON envelope, and we assert the action
    originates from *that content* passing through parse_tool_response.
    """
    tool_call_envelope = json.dumps(
        {
            "reply": "Pulling up the Opus account.",
            "actions": [{"type": "open_account", "name": "opus"}],
        }
    )
    reg = _FakeRegistry(content=tool_call_envelope)
    store = ConversationStore(agent_db)
    agent = ManagerAgent(reg, SettingsStore(agent_db), store)
    conv = store.create("u1")

    turn = agent.chat_with_actions("u1", conv.id, "open opus account", ModelRef("ep1", "mdl"))

    # The LLM was actually called (not bypassed).
    assert len(reg.calls) == 1, "registry.chat was not called — agent is bypassing the LLM"
    # The action in the turn came from parsing the registry's returned content.
    assert turn.reply == "Pulling up the Opus account."
    assert turn.actions == [{"type": "open_account", "name": "opus"}], (
        f"Expected open_account action from LLM content but got: {turn.actions!r}"
    )
