"""``search_context`` — semantic search against the local analyst-digest vault.

Tool name:      search_context
Args:           query (str), k=5 (int)
ToolResult:     ok=True, data={"results": [...], "query": str}
Latency tier:   fast (local vector search, no external fetch)
Cost class:     free
Gating:         only available in digest mode; absent when digest_store is None

This tool replaces the slow external-fetch LOOK tools (news, world_events,
prediction_market_odds, forecast, research, situation) when a trader is in
digest mode.  Instead of making live round-trips, the agent queries the local
vector vault (pre-populated by the DigestDaemon) for the context it needs.

The vault holds two kinds of points:
  - Analyst digests (``d:{user_id}:digests`` collection)
  - Research briefs  (``r:{user_id}:briefs`` collection from WS-C ResearchStore)

Both are searched and the top-k hits are merged and returned as ranked snippets
so the agent can ask targeted questions without re-gathering data.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

_MAX_K = 20


class SearchContextTool(LookToolBase):
    """Local semantic search over the digest + research vault.

    Parameters
    ----------
    digest_store:
        :class:`~trading_agent.digest.store.DigestStore` — the local vault.
        When ``None`` the tool returns an unavailable error.
    research_store:
        Optional :class:`~trading_agent.research.store.ResearchStore`.  When
        wired, brief search results are merged into the output.
    owner_user_id, trader_id:
        Standard namespace params.
    """

    TOOL_META: dict[str, Any] = {
        "name": "search_context",
        "description": (
            "Semantic search over the local analyst-digest vault. "
            "Returns ranked snippets from pre-compiled digests and research "
            "briefs. Replaces live fetch tools (news, research_brief, "
            "world_events, situation) in digest mode — no external calls."
        ),
        "args": {
            "query": "str — what to look for (e.g. 'AAPL earnings sentiment')",
            "k": "int (default 5, max 20) — number of results",
        },
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        digest_store: Any = None,
        research_store: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._digest_store = digest_store
        self._research_store = research_store

    def __call__(self, query: str, k: int = 5) -> Any:
        """Run a semantic search and return ranked context snippets.

        Returns
        -------
        ToolResult
            ok=True, data={"results": [...], "query": str, "total": int}

        Example
        -------
        >>> tool = SearchContextTool(trader_id="Alpha", digest_store=None)
        >>> result = tool("NVDA GPU demand")
        >>> result.ok
        True
        >>> result.data["results"]
        []
        """
        if not query or not query.strip():
            return self._err("invalid_input", "query must not be empty")

        if self._digest_store is None:
            return self._ok(
                {
                    "results": [],
                    "query": query,
                    "total": 0,
                    "note": "digest vault not wired — digest mode may be disabled",
                }
            )

        user_id = self.owner_user_id
        if user_id is None:
            return self._ok(
                {
                    "results": [],
                    "query": query,
                    "total": 0,
                    "note": "no user context",
                }
            )

        k = max(1, min(int(k), _MAX_K))
        results: list[dict[str, Any]] = []

        # 1. Search the digest vault.
        try:
            digest_hits = self._digest_store.search_vector(user_id, query, k=k)
            for hit in digest_hits:
                text = str(hit.get("digest_text", ""))[:500]
                if text:
                    results.append(
                        {
                            "source": "digest",
                            "universe": hit.get("universe_key", ""),
                            "regime": hit.get("regime_label"),
                            "text": text,
                        }
                    )
        except Exception as exc:
            results.append({"source": "digest", "error": str(exc)})

        # 2. Search the research briefs vault.
        if self._research_store is not None:
            try:
                briefs = self._research_store.search(user_id, query, k=min(k, 5))
                for b in briefs:
                    snippet = f"[{b.ticker}] {b.summary}"
                    if b.catalysts:
                        snippet += f" | catalysts: {'; '.join(b.catalysts[:3])}"
                    results.append(
                        {
                            "source": "research_brief",
                            "ticker": b.ticker,
                            "sentiment": b.sentiment,
                            "text": snippet[:400],
                        }
                    )
            except Exception as exc:
                results.append({"source": "research_brief", "error": str(exc)})

        return self._ok({"results": results[:k], "query": query, "total": len(results)})
