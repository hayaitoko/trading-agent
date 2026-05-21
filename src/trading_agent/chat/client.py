"""Thin OpenRouter client. Returns the parsed assistant message as a dict.

Kept deliberately small. The service layer drives the tool-use loop on top.
"""
from typing import Any

import httpx

from trading_agent.chat.models import ChatMessage, ModelSpec

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


def _format_message(m: ChatMessage) -> dict:
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "name": m.tool_name,
            "content": m.content,
        }
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": m.tool_calls,
        }
    if m.images:
        parts: list[dict] = []
        if m.content:
            parts.append({"type": "text", "text": m.content})
        for img in m.images:
            parts.append({"type": "image_url", "image_url": {"url": img}})
        return {"role": m.role, "content": parts}
    return {"role": m.role, "content": m.content}


def _system_message(prompt: str, cache: bool) -> dict:
    if not cache:
        return {"role": "system", "content": prompt}
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
        ],
    }


def _provider_block(model: ModelSpec) -> dict | None:
    if model.is_anthropic:
        return {"order": ["anthropic"], "allow_fallbacks": True}
    return None


async def call_model(
    *,
    api_key: str,
    model: ModelSpec,
    system_prompt: str,
    history: list[ChatMessage],
    tools: list[dict],
    timeout: float = 60.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set. Add it in Settings.")

    payload: dict[str, Any] = {
        "model": model.id,
        "messages": [
            _system_message(system_prompt, cache=model.is_anthropic),
            *(_format_message(m) for m in history),
        ],
    }
    if tools:
        payload["tools"] = tools
    provider = _provider_block(model)
    if provider is not None:
        payload["provider"] = provider

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/hayaitoko/trading-agent",
        "X-Title": "trading-agent",
    }

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as http:
        response = await http.post(OPENROUTER_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise OpenRouterError(
                f"OpenRouter returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError(f"no choices in response: {data}")
    return choices[0]["message"]
