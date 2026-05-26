"""Tests for the OpenRouter client and JSON extraction (no network)."""

import json

import httpx
import pytest

from trading_agent.llm.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    parse_json_object,
)


def _chat_transport(capture: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["body"] = json.loads(request.content)
        capture["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "anthropic/claude-opus-4.7",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"total_tokens": 42, "cost": 0.001},
            },
        )

    return httpx.MockTransport(handler)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError):
        OpenRouterClient()


def test_chat_sends_auth_and_parses() -> None:
    cap: dict = {}
    client = OpenRouterClient(api_key="sk-test", transport=_chat_transport(cap))
    res = client.chat("anthropic/claude-opus-4.7", [{"role": "user", "content": "hi"}])
    assert res.content == '{"ok": true}'
    assert res.usage["total_tokens"] == 42
    assert res.cost == 0.001
    assert cap["auth"] == "Bearer sk-test"
    assert cap["body"]["model"] == "anthropic/claude-opus-4.7"


def test_zdr_adds_provider_block() -> None:
    cap: dict = {}
    client = OpenRouterClient(api_key="sk-test", zdr=True, transport=_chat_transport(cap))
    client.chat("m", [{"role": "user", "content": "x"}], json_mode=True)
    assert cap["body"]["provider"] == {"data_collection": "deny"}
    assert cap["body"]["response_format"] == {"type": "json_object"}


def test_zdr_off_omits_provider() -> None:
    cap: dict = {}
    client = OpenRouterClient(api_key="sk-test", zdr=False, transport=_chat_transport(cap))
    client.chat("m", [{"role": "user", "content": "x"}])
    assert "provider" not in cap["body"]


def test_chat_http_error_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = OpenRouterClient(api_key="sk-test", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenRouterError, match="429"):
        client.chat("m", [{"role": "user", "content": "x"}])


def test_list_models() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "anthropic/claude-opus-4.7"}]})

    client = OpenRouterClient(api_key="sk-test", transport=httpx.MockTransport(handler))
    models = client.list_models()
    assert models[0]["id"] == "anthropic/claude-opus-4.7"


@pytest.mark.parametrize(
    "text",
    [
        '{"decisions": []}',
        '```json\n{"decisions": []}\n```',
        'Sure! Here is my answer:\n{"decisions": []}\nHope that helps.',
        '```\n{"decisions": []}\n```',
    ],
)
def test_parse_json_object_variants(text: str) -> None:
    assert parse_json_object(text) == {"decisions": []}


def test_parse_json_object_failure() -> None:
    with pytest.raises(ValueError):
        parse_json_object("no json here at all")
