"""Research router (WS-C): list shared per-ticker briefs + trigger a gated pass.

``GET /api/research`` lists the user's recent briefs (the Research tab read
surface). ``POST /api/research/run`` triggers **one** cost-gated pass of the
:class:`~trading_agent.research.agent.ResearchAgent` — never an uncapped loop,
and refused once the user's daily $ ceiling is hit.

All model access goes through the per-user :class:`EndpointRegistry` (no
hardcoded provider/url/key). The brief's :class:`ModelRef` is resolved from the
request body if given, else from the user's ``research_model`` /
``research_endpoint_id`` settings, else the first enabled chat endpoint.

The store/agent are built lazily from ``app.state`` (the WS-0 spine: ``db``,
``settings``, ``endpoints``) and the shared WS-D vector store; the vector store
is cached on ``app.state`` so repeated requests reuse one connection.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.endpoints import EndpointError, EndpointRegistry, ModelRef
from ...config.settings_store import SettingsStore
from ...config.users import current_user
from ...ingest.store import IngestStore
from ...memory.embed import LocalEmbedder
from ...memory.reflect import CostGateError
from ...memory.vector import DEFAULT_MEMORY_DB, VectorStore, make_vector_store
from ...research.agent import DEFAULT_RUN_USD, ResearchAgent
from ...research.store import Brief, ResearchStore

router = APIRouter(tags=["research"])

# Endpoint types that can serve a chat/completions research call. ``local`` is
# allowed too (a local model is just another endpoint).
_CHAT_TYPES = ("openrouter", "openai", "anthropic", "local")


class RunRequest(BaseModel):
    """Optional body for an explicit research pass.

    All fields optional: omit them to run the user's configured research model
    over the whole new backlog.
    """

    tickers: list[str] | None = None
    endpoint_id: str | None = None
    model: str | None = None
    estimated_usd: float | None = None


# --- app.state plumbing ------------------------------------------------------


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


def _registry(request: Request) -> EndpointRegistry:
    return request.app.state.endpoints  # type: ignore[no-any-return]


def _vector_store(request: Request, user_id: str) -> VectorStore:
    """The shared WS-D vector store, cached on ``app.state`` per backend name.

    The memory-db path is read from the environment at call time so tests can
    isolate it (``TRADING_AGENT_MEMORY_DB``); production uses the WS-D default.
    """
    cache: dict[str, VectorStore] = getattr(request.app.state, "_research_vstores", {})
    name = str(_settings(request).get(user_id, "vstore", "sqlite-vec"))
    store = cache.get(name)
    if store is None:
        path = os.environ.get("TRADING_AGENT_MEMORY_DB", DEFAULT_MEMORY_DB)
        store = make_vector_store(name, path=path)
        cache[name] = store
        request.app.state._research_vstores = cache
    return store


def _research_store(request: Request, user_id: str) -> ResearchStore:
    registry = _registry(request)
    # Share the registry's (test) transport so on-box embeddings honor the same
    # seam model calls do; None in production → real local endpoint.
    transport = getattr(registry, "_transport", None)
    embedder = LocalEmbedder(registry, _settings(request), user_id, transport=transport)
    return ResearchStore(request.app.state.db, _vector_store(request, user_id), embedder)


def _agent(request: Request, user_id: str) -> ResearchAgent:
    return ResearchAgent(
        IngestStore(request.app.state.db),
        _research_store(request, user_id),
        _registry(request),
        _settings(request),
    )


def _resolve_ref(request: Request, user_id: str, body: RunRequest) -> ModelRef:
    """Pick the (endpoint, model) for this pass: body → settings → first enabled."""
    settings = _settings(request)
    registry = _registry(request)

    model = body.model or settings.get(user_id, "research_model")
    endpoint_id = body.endpoint_id or settings.get(user_id, "research_endpoint_id")

    if endpoint_id:
        ep = registry.get(user_id, endpoint_id)
        if ep is None:
            raise HTTPException(status_code=400, detail=f"no such endpoint: {endpoint_id}")
    else:
        ep = next(
            (e for e in registry.list(user_id) if e.enabled and e.type in _CHAT_TYPES),
            None,
        )
        if ep is None:
            raise HTTPException(
                status_code=400,
                detail="no chat endpoint configured — add one in Settings",
            )

    if not model:
        raise HTTPException(
            status_code=400,
            detail="no research model set — choose one in Settings or pass 'model'",
        )
    return ModelRef(endpoint_id=ep.id, model=str(model))


# --- serialization -----------------------------------------------------------


def _brief_public(brief: Brief) -> dict[str, Any]:
    """Canonical Brief fields + cockpit ``RESEARCH`` aliases.

    Research is shared per user, so there is no per-trader ``who``; the cockpit
    groups on it, so it gets the constant ``"research"``. WS-G2 maps the rest:
    ``topic``←ticker, ``text``←summary, ``tags``←catalysts.
    """
    return {
        # canonical (CONTRACTS Brief)
        "id": brief.id,
        "ticker": brief.ticker,
        "summary": brief.summary,
        "sentiment": brief.sentiment,
        "catalysts": brief.catalysts,
        "sources": brief.sources,
        "ts": brief.ts,
        # cockpit RESEARCH aliases
        "who": "research",
        "topic": brief.ticker,
        "text": brief.summary,
        "tags": brief.catalysts,
    }


# --- routes ------------------------------------------------------------------


@router.get("/api/research")
def research(
    request: Request,
    ticker: str | None = None,
    n: int = 20,
    user_id: str = Depends(current_user),
) -> list[dict[str, Any]]:
    """Recent briefs for the user (optionally one ticker), newest first."""
    store = ResearchStore(request.app.state.db)  # read path needs no vector/embedder
    briefs = store.get(user_id, ticker) if ticker else store.recent(user_id, n)
    return [_brief_public(b) for b in briefs]


@router.post("/api/research/run")
def research_run(
    request: Request,
    body: RunRequest | None = None,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Trigger one cost-gated research pass; returns the briefs written."""
    body = body or RunRequest()
    ref = _resolve_ref(request, user_id, body)
    estimated = body.estimated_usd if body.estimated_usd is not None else DEFAULT_RUN_USD
    try:
        briefs = _agent(request, user_id).run(
            user_id, body.tickers, ref, estimated_usd=estimated
        )
    except CostGateError as exc:
        # Daily $ ceiling reached — refuse, don't silently overspend.
        raise HTTPException(status_code=402, detail=str(exc))
    except EndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ran": True, "count": len(briefs), "briefs": [_brief_public(b) for b in briefs]}
