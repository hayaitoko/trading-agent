"""Local text embedding.

Two implementations behind one :class:`Embedder` protocol:

- :class:`LocalEmbedder` — production. Resolves a **local** endpoint through the
  :class:`~trading_agent.config.endpoints.EndpointRegistry` (Ollama / llama.cpp,
  OpenAI-compatible ``/embeddings``) and uses the user's ``embed_model`` setting
  (default ``bge-small-en-v1.5``, 384-dim). It refuses non-local endpoints —
  embeddings must never leave the box (``D-memory.md``: "No WAN calls for
  embeddings"). Base url + key come from the registry, never hardcoded.
- :class:`FakeEmbedder` — tests/CI. A deterministic hashing bag-of-words
  embedder: same text → same vector, and texts sharing words are genuinely more
  similar, so recall/dedup logic is exercised without a live model.

CONTRACTS.md names a free ``embed(text) -> list[float]``; here that is realized
as ``Embedder.embed(text)`` so the model/endpoint is resolved from per-user
settings rather than a global.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from ..config.endpoints import EndpointRegistry
    from ..config.settings_store import SettingsStore

DEFAULT_EMBED_MODEL = "bge-small-en-v1.5"
DEFAULT_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbedError(RuntimeError):
    """Embedding could not be produced (no local endpoint, bad response, ...)."""


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic hashing bag-of-words embedder for tests (no network).

    Hashes each token into a fixed-width vector and L2-normalizes. Shared
    vocabulary ⇒ higher cosine, so semantic-style dedup/recall tests are
    meaningful while staying fully reproducible.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class LocalEmbedder:
    """Embeds via a user's **local** OpenAI-compatible ``/embeddings`` endpoint.

    The endpoint is resolved through the registry: the ``embed_endpoint_id``
    setting if present, else the first enabled endpoint of type ``local``.
    """

    def __init__(
        self,
        registry: EndpointRegistry,
        settings: SettingsStore,
        user_id: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._user_id = user_id
        self._transport = transport
        self._timeout = timeout
        self.model = settings.get(user_id, "embed_model", DEFAULT_EMBED_MODEL)
        self.dim = int(settings.get(user_id, "embed_dim", DEFAULT_DIM))

    def _resolve_endpoint(self) -> tuple[str, str]:
        """Return ``(base_url, api_key)`` of the local embedding endpoint."""
        ep_id = self._settings.get(self._user_id, "embed_endpoint_id", "")
        ep = None
        if ep_id:
            ep = self._registry.get(self._user_id, ep_id)
        if ep is None:
            ep = next(
                (e for e in self._registry.list(self._user_id) if e.type == "local" and e.enabled),
                None,
            )
        if ep is None:
            raise EmbedError(
                "no local embedding endpoint configured "
                "(add a 'local' endpoint or set embed_endpoint_id)"
            )
        if ep.type != "local":
            # Hard rule: embeddings stay on-box, never over the WAN.
            raise EmbedError(f"embed endpoint must be type 'local', got '{ep.type}'")
        return ep.base_url, ep.api_key

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        base_url, api_key = self._resolve_endpoint()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        with httpx.Client(
            base_url=base_url, timeout=self._timeout, transport=self._transport, headers=headers
        ) as client:
            resp = client.post("/embeddings", json={"model": self.model, "input": texts})
        if resp.status_code >= 400:
            raise EmbedError(f"embeddings {resp.status_code}: {resp.text[:300]}")
        try:
            rows = resp.json()["data"]
            vectors = [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbedError(f"unexpected embeddings response shape: {exc}")
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def make_embedder(
    registry: EndpointRegistry,
    settings: SettingsStore,
    user_id: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LocalEmbedder:
    """Convenience builder mirroring the per-user resolution agents expect."""
    return LocalEmbedder(registry, settings, user_id, transport=transport)
