"""OpenRouter chat client (OpenAI-compatible) over httpx.

Provides two call styles:

* :meth:`OpenRouterClient.chat` — plain completion (JSON or free text), used
  by the legacy :class:`~trading_agent.llm.trader.LLMTrader` structured-output
  path and the manager agent.

* :meth:`OpenRouterClient.chat_with_tools` — OpenAI-compatible tool-calling,
  used by the new :class:`~trading_agent.llm.trader.AgentTrader` ReAct loop.
  Returns :class:`ToolCallChatResult` which carries either ``content`` (text)
  or a list of :class:`ToolCall` objects (one per function the model invoked).

ZDR is enforced per-request via the ``provider`` routing block
(``data_collection: deny``) so only non-retaining providers serve each call —
matching Artoo's posture.  Providers that require data retention (e.g. Alibaba
closed-weight Qwens) are excluded under this policy.

Inject a custom ``transport`` (e.g. ``httpx.MockTransport``) to test without a
network or a live key.

Cache breakpoint: for Anthropic-backend models, callers can add
``{"cache_control": {"type": "ephemeral"}}`` to the last stable message block
(system prompt + tool list) to engage Anthropic's prompt-caching tier.
Non-Anthropic providers ignore the field.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx


class OpenRouterError(RuntimeError):
    """Raised for missing credentials or a non-2xx OpenRouter response."""


@dataclass
class ChatResult:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None


@dataclass
class ToolCall:
    """One function invocation returned by the model in tool-call mode."""

    id: str
    name: str
    arguments: dict[str, Any]  # pre-parsed from the JSON string


@dataclass
class ToolCallChatResult:
    """Result from :meth:`OpenRouterClient.chat_with_tools`.

    Exactly one of ``content`` or ``tool_calls`` carries data per turn:
    * ``tool_calls`` is non-empty when the model invoked tools (finish_reason
      ``"tool_calls"``).
    * ``content`` is non-empty when the model returned plain text (finish_reason
      ``"stop"`` or ``"end_turn"``).
    * Both empty → treat as implicit hold (model returned nothing useful).
    """

    content: str | None
    tool_calls: list[ToolCall]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    finish_reason: str = ""


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        zdr: bool = True,
        timeout: float = 60.0,
        title: str = "trading-agent-bench",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise OpenRouterError(
                "OpenRouter API key missing: set OPENROUTER_API_KEY "
                "(create one at https://openrouter.ai/keys)."
            )
        self.api_key = key
        self.zdr = zdr
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {key}",
                "X-Title": title,
                "Content-Type": "application/json",
            },
        )

    # --- API ----------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.zdr:
            # Route only to providers that do not retain prompt/response data.
            body["provider"] = {"data_collection": "deny"}

        resp = self._client.post("/chat/completions", json=body)
        if resp.status_code >= 400:
            raise OpenRouterError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response shape: {exc}")
        return ChatResult(
            content=content,
            model=data.get("model", model),
            usage=data.get("usage", {}) or {},
            cost=(data.get("usage", {}) or {}).get("cost"),
        )

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tool_choice: str = "auto",
    ) -> ToolCallChatResult:
        """OpenAI-compatible tool-calling completion.

        ``tools`` should be a list of OpenAI function-tool dicts::

            [{"type": "function", "function": {"name": ..., "description": ...,
              "parameters": {...}}}]

        The response carries either ``content`` (model returned text) or a list
        of :class:`ToolCall` objects (model called one or more functions).

        Args are pre-parsed from JSON so callers never deal with raw argument
        strings.  A malformed arguments JSON from the model becomes an empty
        dict with the raw string logged to ``arguments["_raw"]``.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.zdr:
            body["provider"] = {"data_collection": "deny"}

        resp = self._client.post("/chat/completions", json=body)
        if resp.status_code >= 400:
            raise OpenRouterError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response shape: {exc}")

        finish_reason = choice.get("finish_reason", "")
        raw_calls = msg.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {"_raw": str(args)},
                )
            )

        usage = data.get("usage", {}) or {}
        return ToolCallChatResult(
            content=msg.get("content") or None,
            tool_calls=tool_calls,
            model=data.get("model", model),
            usage=usage,
            cost=usage.get("cost"),
            finish_reason=finish_reason,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """Return OpenRouter's model catalog (for the UI menu). Raises on error."""
        resp = self._client.get("/models")
        if resp.status_code >= 400:
            raise OpenRouterError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        models = payload.get("data", payload)
        return list(models) if isinstance(models, list) else []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response.

    Tolerates ```json fences and leading/trailing prose by slicing to the
    outermost ``{ ... }``. Raises ``ValueError`` if nothing parses.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # drop the opening fence (``` or ```json) and the closing fence
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError("no JSON object found in response")
