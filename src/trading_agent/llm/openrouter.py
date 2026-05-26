"""Minimal OpenRouter chat client (OpenAI-compatible) over httpx.

Kept deliberately thin: one ``chat`` call for completions and ``list_models``
for the UI's model menu. ZDR is enforced per-request via the ``provider``
routing block (``data_collection: deny``) so only non-retaining providers serve
the call — matching Artoo's posture. Note that some providers (e.g. Alibaba's
closed Qwen weights) are excluded under that policy.

Inject a custom ``transport`` (e.g. ``httpx.MockTransport``) to test without a
network or a live key.
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
