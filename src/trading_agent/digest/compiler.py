"""DigestCompiler — one cheap-model pass that distils slow signals into a digest.

The compiler gathers available slow-data sources (news items from IngestStore,
research briefs from ResearchStore, and situation data when providers are wired),
then calls a cheap model to DISTIL and RANK the signals into a compact, token-
budgeted digest record.

Design invariants:
  - Never dumps raw data — always distils.
  - Token-budgeted: compiled text is capped at ``max_chars`` before storage.
  - Cost-gated via :class:`~trading_agent.memory.reflect.CostGate`.
  - No external fetches in the compiler itself — all data comes from local stores
    that have already been hydrated by the ingest/research daemons.
  - Material-event detection: if any headline contains a high-impact keyword, the
    digest record is flagged ``material_flag=True`` so the daemon can fire a
    research-bombshell event-wake.

Materiality keywords (conservative set; operator-configurable):
  MATERIAL_KEYWORDS covers earnings/surprise beats, halts, acquisitions,
  regulatory actions, Fed decisions, and major macro prints.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from .store import DEFAULT_MAX_CHARS, Digest, DigestStore, universe_key

if TYPE_CHECKING:
    from ..config.endpoints import EndpointRegistry, ModelRef
    from ..config.settings_store import SettingsStore
    from ..research.store import ResearchStore

logger = logging.getLogger(__name__)

# Default spend estimate per compile pass.
DEFAULT_COMPILE_USD: float = 0.01

# Maximum news items fed into the distillation prompt per pass.
MAX_NEWS_ITEMS: int = 30

# Maximum research briefs fed into the distillation prompt per pass.
MAX_BRIEFS: int = 10

# Conservative set of materiality signal words. Match is case-insensitive
# substring.  Keep this list short — false positives waste event-wake budget.
MATERIAL_KEYWORDS: list[str] = [
    "earnings surprise",
    "beats expectations",
    "misses expectations",
    "halted",
    "trading halt",
    "acquisition",
    "merger",
    "sec investigation",
    "sec charges",
    "subpoena",
    "layoffs",
    "bankruptcy",
    "fed decision",
    "rate cut",
    "rate hike",
    "emergency meeting",
    "circuit breaker",
    "gdp miss",
    "cpi surprise",
]

_SYSTEM = (
    "You are a markets analyst assistant. You receive raw news headlines and "
    "research briefs for a set of tickers. Your task is to DISTIL and RANK the "
    "most actionable signals into a compact analyst digest for a trading agent.\n\n"
    "Rules:\n"
    "  1. Be concise — the trader's context window is limited.\n"
    "  2. Lead with the most market-moving items.\n"
    "  3. Compress each headline to ONE line (ticker: key point).\n"
    "  4. After headlines, write a 1-line regime label "
    "(e.g. 'Regime: calm / elevated / event-window / risk-off').\n"
    "  5. List up to 3 key forecasts or prediction-market odds if available.\n"
    "  6. Flag any material events (earnings surprise, halt, acquisition, "
    "regulatory action, Fed decision) with a leading [MATERIAL] tag.\n"
    "  7. Total output must fit in 400 words.\n\n"
    "Reply ONLY as JSON:\n"
    '{"headlines": ["AAPL: ..."], "regime": "calm", "key_points": ["..."], '
    '"material_event": false}'
)


class DigestCompiler:
    """Distils slow-data signals into a compact analyst digest (one model pass).

    Parameters
    ----------
    digest_store:
        :class:`~.store.DigestStore` where compiled records are persisted.
    research_store:
        :class:`~trading_agent.research.store.ResearchStore` for brief lookup.
    registry:
        :class:`~trading_agent.config.endpoints.EndpointRegistry` to resolve
        the model ref for the compile call.
    settings:
        :class:`~trading_agent.config.settings_store.SettingsStore` for per-
        user settings (cost ceiling, compile model, etc.).
    db:
        The shared :class:`~trading_agent.config.db.Database` (for news lookup).
    max_chars:
        Maximum character budget for the stored digest text.
    """

    def __init__(
        self,
        digest_store: DigestStore,
        research_store: ResearchStore | None,
        registry: EndpointRegistry,
        settings: SettingsStore,
        db: Any = None,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._digest_store = digest_store
        self._research_store = research_store
        self._registry = registry
        self._settings = settings
        self._db = db
        self._max_chars = max_chars

    # ------------------------------------------------------------------
    # Public

    def compile(
        self,
        user_id: str,
        symbols: list[str],
        ref: ModelRef,
        *,
        estimated_usd: float = DEFAULT_COMPILE_USD,
    ) -> Digest | None:
        """Compile a fresh analyst digest for ``symbols``.

        Returns the persisted :class:`~.store.Digest` on success, ``None``
        when the cost gate blocks the call or the model is unreachable.
        """
        from ..memory.reflect import CostGate, CostGateError

        gate = CostGate(self._settings)
        try:
            gate.check(user_id, estimated_usd)
        except CostGateError as exc:
            logger.info("digest_compiler: cost gate blocked for user=%s: %s", user_id, exc)
            return None

        news_items = self._gather_news(user_id, symbols)
        briefs = self._gather_briefs(user_id, symbols)
        prompt = self._build_prompt(symbols, news_items, briefs)

        try:
            raw = self._call_model(user_id, ref, prompt)
        except Exception as exc:
            logger.warning("digest_compiler: model call failed: %s", exc)
            return None

        # Record actual spend (model may not report; estimate is acceptable).
        cost_usd = estimated_usd
        if isinstance(raw, dict) and "cost" in raw:
            cost_usd = float(raw.get("cost") or estimated_usd)
        try:
            gate.record(user_id, cost_usd)
        except Exception:
            pass

        content = raw if isinstance(raw, str) else raw.get("content", "")
        parsed = self._parse_response(content)
        digest_text = self._render_text(parsed, symbols)
        digest_text = digest_text[: self._max_chars]

        material = parsed.get("material_event", False) or self._detect_material(
            parsed.get("headlines", [])
        )

        digest = Digest(
            user_id=user_id,
            universe_key=universe_key(symbols),
            as_of=time.time(),
            digest_text=digest_text,
            headlines=list(parsed.get("headlines", []))[:20],
            regime_label=parsed.get("regime"),
            material_flag=bool(material),
        )
        try:
            self._digest_store.put(digest)
        except Exception as exc:
            logger.warning("digest_compiler: store write failed: %s", exc)
            return None

        logger.info(
            "digest_compiler: compiled user=%s symbols=%s material=%s chars=%d",
            user_id,
            ",".join(symbols),
            material,
            len(digest_text),
        )
        return digest

    # ------------------------------------------------------------------
    # Helpers — data gathering

    def _gather_news(self, user_id: str, symbols: list[str]) -> list[dict[str, Any]]:
        """Pull recent news items from the ingest store (no external fetch)."""
        if self._db is None:
            return []
        try:
            from ..ingest.store import IngestStore

            IngestStore(self._db)  # ensures schema; idempotent

            tickers_sql = ",".join("?" * len(symbols)) if symbols else "?"
            params: list[Any] = [user_id] + [s.upper() for s in symbols] + [MAX_NEWS_ITEMS]
            rows = self._db.query(
                f"SELECT ticker, text, url, ts FROM raw_items "
                f"WHERE user_id = ? AND ticker IN ({tickers_sql}) "
                f"ORDER BY fetched_at DESC LIMIT ?",
                tuple(params),
            )
            return [
                {
                    "ticker": str(r["ticker"]),
                    "text": str(r["text"])[:200],
                    "ts": str(r["ts"]),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("digest_compiler: news gather failed: %s", exc)
            return []

    def _gather_briefs(self, user_id: str, symbols: list[str]) -> list[dict[str, Any]]:
        """Pull recent research briefs from the research store."""
        if self._research_store is None:
            return []
        try:
            out: list[dict[str, Any]] = []
            for sym in symbols[:MAX_BRIEFS]:
                briefs = self._research_store.get(user_id, sym)
                if briefs:
                    b = briefs[0]  # most recent
                    out.append(
                        {
                            "ticker": b.ticker,
                            "summary": b.summary[:300],
                            "sentiment": b.sentiment,
                            "catalysts": b.catalysts[:3],
                        }
                    )
            return out
        except Exception as exc:
            logger.debug("digest_compiler: briefs gather failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Helpers — prompt + model

    def _build_prompt(
        self,
        symbols: list[str],
        news_items: list[dict[str, Any]],
        briefs: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = [f"Universe: {', '.join(symbols)}\n"]

        if news_items:
            parts.append("=== RECENT HEADLINES ===")
            for item in news_items[:MAX_NEWS_ITEMS]:
                parts.append(f"[{item['ticker']}] {item['text']} ({item['ts'][:10]})")

        if briefs:
            parts.append("\n=== RESEARCH BRIEFS ===")
            for b in briefs:
                cats = "; ".join(b["catalysts"]) if b["catalysts"] else "none"
                parts.append(
                    f"[{b['ticker']}] sentiment={b['sentiment']:+.2f} — "
                    f"{b['summary']} | catalysts: {cats}"
                )

        if not news_items and not briefs:
            parts.append("(No recent data available — produce a minimal digest.)")

        return "\n".join(parts)

    def _call_model(self, user_id: str, ref: ModelRef, prompt: str) -> Any:
        """Call the model via EndpointRegistry.  Returns a dict with 'content' and 'cost'."""
        endpoint = self._registry.get(user_id, ref.endpoint_id)
        if endpoint is None:
            raise RuntimeError(f"endpoint {ref.endpoint_id!r} not found for user {user_id!r}")

        import httpx

        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"

        payload = {
            "model": ref.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }

        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("model returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        cost = None
        usage = data.get("usage", {})
        if usage:
            # Cheap estimate: 0.15/1M in, 0.60/1M out (Haiku-tier pricing).
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            cost = (inp * 0.15 + out * 0.60) / 1_000_000
        return {"content": content, "cost": cost}

    # ------------------------------------------------------------------
    # Helpers — parsing + materiality

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the model's JSON response; fall back to empty on any error."""
        text = raw.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _render_text(self, parsed: dict[str, Any], symbols: list[str]) -> str:
        """Render the parsed model output into a compact digest string."""
        lines: list[str] = [f"[Analyst Digest — {', '.join(symbols)}]"]

        regime = parsed.get("regime")
        if regime:
            lines.append(f"Regime: {regime}")

        headlines = parsed.get("headlines", [])
        if headlines:
            lines.append("Top signals:")
            for h in headlines[:10]:
                lines.append(f"  • {h}")

        key_points = parsed.get("key_points", [])
        if key_points:
            lines.append("Key points:")
            for kp in key_points[:5]:
                lines.append(f"  • {kp}")

        if parsed.get("material_event"):
            lines.append("[MATERIAL EVENT DETECTED]")

        return "\n".join(lines)

    @staticmethod
    def _detect_material(headlines: list[str]) -> bool:
        """True if any headline contains a materiality keyword."""
        combined = " ".join(str(h) for h in headlines).lower()
        return any(kw in combined for kw in MATERIAL_KEYWORDS)
