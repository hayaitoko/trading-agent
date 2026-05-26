"""Provider/endpoint registry — the single seam every model call goes through.

No agent constructs its own HTTP client or hardcodes a base_url/key; it names a
model with a :class:`ModelRef` (``endpoint_id`` + ``model``) and asks the
:class:`EndpointRegistry`. Endpoints are per-user rows; several may be enabled at
once (OpenRouter + a local Ollama, say).

Adapters behind the common :class:`ChatClient` interface:
- :class:`OpenAICompatibleClient` — OpenRouter, OpenAI, and local (Ollama /
  llama.cpp) all speak the OpenAI ``/chat/completions`` wire format. ZDR routing
  (``data_collection: deny``) is applied for OpenRouter only, matching the
  reference adapter in ``llm/openrouter.py``.
- :class:`AnthropicClient` — Anthropic's ``/messages`` format (system pulled out,
  ``x-api-key`` header) behind the same interface.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from ..llm.openrouter import ChatResult  # reuse the canonical result type

# Alias for chat message lists. Used inside EndpointRegistry, whose ``list``
# method would otherwise shadow the builtin ``list`` in annotations.
Messages = list[dict[str, str]]

# Endpoint.type ∈ these. Each maps to a default base_url (mirrors the cockpit
# mock's PROVIDERS block); users may override the URL per endpoint.
DEFAULT_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "local": "http://localhost:11434/v1",
}
ENDPOINT_TYPES = frozenset(DEFAULT_BASE_URLS)


class EndpointError(RuntimeError):
    """Unknown endpoint, unsupported type, or a disabled endpoint being used."""


@dataclass
class ModelRef:
    """How every agent names *which* model: an endpoint + a model id on it."""

    endpoint_id: str
    model: str


@dataclass
class Endpoint:
    id: str
    user_id: str
    type: str
    name: str
    base_url: str
    api_key: str
    enabled: bool

    def public(self) -> dict[str, Any]:
        """API-safe view: the secret key is masked to its last 4 chars."""
        key = self.api_key or ""
        preview = f"…{key[-4:]}" if len(key) >= 4 else ("•••" if key else "")
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "base_url": self.base_url,
            "key_preview": preview,
            "has_key": bool(key),
            "enabled": self.enabled,
        }


@runtime_checkable
class ChatClient(Protocol):
    """The narrow surface agents use; both adapters implement it."""

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult: ...

    def close(self) -> None: ...


class OpenAICompatibleClient:
    """Generalized OpenRouter/OpenAI/local client over the OpenAI wire format."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        zdr: bool = False,
        timeout: float = 60.0,
        title: str = "trading-agent",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.zdr = zdr
        headers = {"Content-Type": "application/json", "X-Title": title}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport, headers=headers
        )

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
            body["provider"] = {"data_collection": "deny"}
        resp = self._client.post("/chat/completions", json=body)
        if resp.status_code >= 400:
            raise EndpointError(f"chat completion {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(f"unexpected response shape: {exc}")
        usage = data.get("usage", {}) or {}
        return ChatResult(
            content=content, model=data.get("model", model), usage=usage, cost=usage.get("cost")
        )

    def list_models(self) -> list[dict[str, Any]]:
        resp = self._client.get("/models")
        if resp.status_code >= 400:
            raise EndpointError(f"list models {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        models = payload.get("data", payload)
        return list(models) if isinstance(models, list) else []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AnthropicClient:
    """Adapter for Anthropic's ``/messages`` API behind the ChatClient interface."""

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.ANTHROPIC_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport, headers=headers
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,  # noqa: ARG002 - no native JSON mode; accepted for parity
    ) -> ChatResult:
        # Anthropic separates the system prompt from the message turns.
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system:
            body["system"] = system
        resp = self._client.post("/messages", json=body)
        if resp.status_code >= 400:
            raise EndpointError(f"anthropic {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            blocks = data["content"]
            content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise EndpointError(f"unexpected anthropic response shape: {exc}")
        return ChatResult(content=content, model=data.get("model", model), usage=data.get("usage", {}) or {})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AnthropicClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EndpointRegistry:
    """CRUD over per-user endpoints + client resolution and a chat convenience.

    ``transport`` is a test seam: when set it is injected into every built
    client so tests can supply an ``httpx.MockTransport`` (no network, no key).
    """

    def __init__(self, db: Any, *, transport: httpx.BaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    # --- CRUD ----------------------------------------------------------------

    def _row_to_endpoint(self, row: Any) -> Endpoint:
        return Endpoint(
            id=row["id"],
            user_id=row["user_id"],
            type=row["type"],
            name=row["name"],
            base_url=row["base_url"],
            api_key=row["api_key"],
            enabled=bool(row["enabled"]),
        )

    def list(self, user_id: str) -> list[Endpoint]:
        rows = self._db.query(
            "SELECT * FROM endpoints WHERE user_id = ? ORDER BY name", (user_id,)
        )
        return [self._row_to_endpoint(r) for r in rows]

    def get(self, user_id: str, endpoint_id: str) -> Endpoint | None:
        row = self._db.query_one(
            "SELECT * FROM endpoints WHERE id = ? AND user_id = ?", (endpoint_id, user_id)
        )
        return self._row_to_endpoint(row) if row else None

    def add(
        self,
        user_id: str,
        type: str,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str = "",
        enabled: bool = True,
    ) -> Endpoint:
        if type not in ENDPOINT_TYPES:
            raise EndpointError(f"unknown endpoint type: {type}")
        ep = Endpoint(
            id=uuid.uuid4().hex,
            user_id=user_id,
            type=type,
            name=name,
            base_url=base_url or DEFAULT_BASE_URLS[type],
            api_key=api_key,
            enabled=enabled,
        )
        self._db.execute(
            "INSERT INTO endpoints (id, user_id, type, name, base_url, api_key, enabled)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ep.id, ep.user_id, ep.type, ep.name, ep.base_url, ep.api_key, int(ep.enabled)),
        )
        return ep

    def update(self, user_id: str, endpoint_id: str, **fields: Any) -> Endpoint:
        current = self.get(user_id, endpoint_id)
        if current is None:
            raise EndpointError(f"no such endpoint: {endpoint_id}")
        allowed = {"type", "name", "base_url", "api_key", "enabled"}
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if "type" in sets and sets["type"] not in ENDPOINT_TYPES:
            raise EndpointError(f"unknown endpoint type: {sets['type']}")
        if "enabled" in sets:
            sets["enabled"] = int(bool(sets["enabled"]))
        if sets:
            cols = ", ".join(f"{k} = ?" for k in sets)
            self._db.execute(
                f"UPDATE endpoints SET {cols} WHERE id = ? AND user_id = ?",
                (*sets.values(), endpoint_id, user_id),
            )
        result = self.get(user_id, endpoint_id)
        assert result is not None
        return result

    def remove(self, user_id: str, endpoint_id: str) -> bool:
        cur = self._db.execute(
            "DELETE FROM endpoints WHERE id = ? AND user_id = ?", (endpoint_id, user_id)
        )
        return cur.rowcount > 0

    def toggle(self, user_id: str, endpoint_id: str, enabled: bool) -> Endpoint:
        return self.update(user_id, endpoint_id, enabled=enabled)

    # --- resolution ----------------------------------------------------------

    def client_for(self, user_id: str, endpoint_id: str) -> ChatClient:
        ep = self.get(user_id, endpoint_id)
        if ep is None:
            raise EndpointError(f"no such endpoint: {endpoint_id}")
        if ep.type == "anthropic":
            return AnthropicClient(ep.base_url, ep.api_key, transport=self._transport)
        return OpenAICompatibleClient(
            ep.base_url, ep.api_key, zdr=(ep.type == "openrouter"), transport=self._transport
        )

    def chat(
        self, user_id: str, ref: ModelRef, messages: Messages, **opts: Any
    ) -> ChatResult:
        """Resolve + call. Refuses disabled endpoints (cost/control safety)."""
        ep = self.get(user_id, ref.endpoint_id)
        if ep is None:
            raise EndpointError(f"no such endpoint: {ref.endpoint_id}")
        if not ep.enabled:
            raise EndpointError(f"endpoint disabled: {ref.endpoint_id}")
        with self.client_for(user_id, ref.endpoint_id) as client:  # type: ignore[attr-defined]
            return client.chat(ref.model, messages, **opts)

    # --- seeding -------------------------------------------------------------

    def seed_defaults(self, user_id: str) -> None:
        """On first run, seed a default OpenRouter endpoint from env, if any.

        Idempotent: only seeds when the user has no endpoints yet and
        ``OPENROUTER_API_KEY`` is set.
        """
        if self.list(user_id):
            return
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            self.add(user_id, "openrouter", "OpenRouter", api_key=key, enabled=True)
