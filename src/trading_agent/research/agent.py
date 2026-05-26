"""``ResearchAgent`` — the shared, cost-gated research pass.

One pass digests the raw items WS-B has ingested into per-ticker briefs every
trader can read (``CONTRACTS.md §WS-C``). The flow:

1. :meth:`IngestStore.drain` from the user's saved cursor → the new backlog.
2. Group items by ticker (items with no ticker fold into a synthetic
   :data:`MARKET_TICKER` macro bucket).
3. **One batched cheap-model call** through the :class:`EndpointRegistry` (model
   named by the caller's :class:`ModelRef`, normally the user's
   ``research_model`` setting) asking for a brief per ticker.
4. Parse the JSON into :class:`Brief` objects (sources = the items' URLs) and
   :meth:`ResearchStore.put` each.

**Cost-gating** (``CONTRACTS.md §Cost-gating``): the agent reuses WS-D's
:class:`~trading_agent.memory.reflect.CostGate` — it ``check``s the per-user
daily $ ceiling *before* the paid call and ``record``s actual (or estimated)
spend after. There is no loop: a ``run`` is one explicit pass (triggered by
``/api/research/run`` or a cadence) and a no-op — zero spend — when the backlog
is empty. A targeted ``tickers`` subset is treated as a non-consuming peek (the
drain cursor only advances on a full pass), so scheduled full passes never lose
items a manual subset run happened to skip.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict

from ..config.endpoints import EndpointRegistry, ModelRef
from ..config.settings_store import SettingsStore
from ..ingest.fetchers.base import RawItem, now_iso
from ..ingest.store import IngestStore
from ..memory.reflect import CostGate
from .store import Brief, ResearchStore

logger = logging.getLogger(__name__)

# Settings key holding the drain watermark (a fetched_at epoch). Distinct from
# WS-D's spend ledger key; both live in user_settings.
CURSOR_KEY = "research_cursor"

# Items a source could not tag with a symbol become one macro brief (the mock's
# "The big picture" card).
MARKET_TICKER = "MARKET"

# Cap on how many items per ticker we put in the prompt — keeps the single
# batched call bounded so cost stays predictable regardless of backlog size.
MAX_ITEMS_PER_TICKER = 12

# Default spend estimate for one pass when the caller gives none. The cockpit's
# Research card models ~6k in / ~1.5k out tokens per run; at cheap-model prices
# that is well under a cent, so this is a deliberately conservative ceiling
# check. Actual cost (when the provider reports it) is what gets recorded.
DEFAULT_RUN_USD = 0.02

_SYSTEM = (
    "You are a markets research analyst. You are given recent social/news items "
    "grouped by ticker. For EACH ticker, write one concise brief for a trading "
    "agent: a 1-3 sentence summary of what matters, a sentiment score, and the "
    "key catalysts. sentiment is a float in [-1, 1] (negative = bearish, 0 = "
    "neutral, positive = bullish). Be specific and skip filler. Reply ONLY as "
    'JSON: {"briefs": [{"ticker": "AAPL", "summary": "...", "sentiment": 0.3, '
    '"catalysts": ["..."]}]}.'
)


class ResearchAgent:
    """Turns ingested items into stored per-ticker briefs, one gated pass."""

    def __init__(
        self,
        ingest: IngestStore,
        store: ResearchStore,
        registry: EndpointRegistry,
        settings: SettingsStore,
    ) -> None:
        self._ingest = ingest
        self._store = store
        self._registry = registry
        self._settings = settings
        self._gate = CostGate(settings)

    # --- the pass ------------------------------------------------------------

    def run(
        self,
        user_id: str,
        tickers: list[str] | None,
        ref: ModelRef,
        *,
        estimated_usd: float = DEFAULT_RUN_USD,
    ) -> list[Brief]:
        """Digest the new ingest backlog into briefs. Cost-gated; explicit.

        ``tickers`` restricts which tickers to brief. ``None`` (the normal
        cadence/trigger) briefs *every* ticker in the backlog and advances the
        drain cursor — a consuming pass. A non-empty subset briefs only those
        and leaves the cursor untouched, so it is a re-runnable peek.

        Raises :class:`~trading_agent.memory.reflect.CostGateError` if the daily
        ceiling would be exceeded. Returns the briefs written ([] on no backlog).
        """
        since = float(self._settings.get(user_id, CURSOR_KEY, 0.0) or 0.0)
        items = self._ingest.drain(user_id, since)
        if not items:
            return []  # nothing new → no model call, no spend

        grouped = self._group_by_ticker(items)
        wanted = {t.upper() for t in tickers} if tickers else None
        if wanted is not None:
            grouped = OrderedDict((t, v) for t, v in grouped.items() if t in wanted)
        if not grouped:
            return []

        # Gate BEFORE the paid call; charge actual cost (or the estimate) after.
        self._gate.check(user_id, estimated_usd)
        result = self._registry.chat(
            user_id,
            ref,
            self._messages(grouped),
            json_mode=True,
            temperature=0.2,
            max_tokens=1200,
        )
        spent = result.cost if result.cost is not None else estimated_usd
        self._gate.record(user_id, spent)

        briefs = self._parse(result.content, grouped)
        for brief in briefs:
            self._store.put(user_id, brief)

        # Only a full pass consumes the backlog (advances the cursor).
        if tickers is None:
            self._settings.set(user_id, CURSOR_KEY, self._ingest.latest_fetched_at(user_id))
        return briefs

    # --- grouping / prompt ---------------------------------------------------

    @staticmethod
    def _group_by_ticker(items: list[RawItem]) -> OrderedDict[str, list[RawItem]]:
        grouped: OrderedDict[str, list[RawItem]] = OrderedDict()
        for it in items:
            key = (it.ticker or MARKET_TICKER).upper()
            grouped.setdefault(key, []).append(it)
        return grouped

    @staticmethod
    def _messages(grouped: OrderedDict[str, list[RawItem]]) -> list[dict[str, str]]:
        blocks: list[str] = []
        for ticker, items in grouped.items():
            lines = [f"## {ticker}"]
            for it in items[:MAX_ITEMS_PER_TICKER]:
                text = it.text.strip().replace("\n", " ")
                lines.append(f"- {text} ({it.url})")
            blocks.append("\n".join(lines))
        user = (
            "Recent items grouped by ticker:\n\n"
            + "\n\n".join(blocks)
            + "\n\nWrite one brief for each of these tickers: "
            + ", ".join(grouped.keys())
            + "."
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]

    # --- parsing -------------------------------------------------------------

    def _parse(
        self, content: str, grouped: OrderedDict[str, list[RawItem]]
    ) -> list[Brief]:
        """Turn the model's JSON into briefs, one per requested ticker present.

        Sources for a brief are the (deduped) URLs of that ticker's items, taken
        from what we sent — never from the model, so they can't be fabricated.
        """
        parsed = self._extract_briefs(content)
        ts = now_iso()
        out: list[Brief] = []
        for raw in parsed:
            ticker = str(raw.get("ticker", "")).strip().upper()
            if ticker not in grouped:
                continue  # model invented / drifted a ticker we didn't ask about
            summary = str(raw.get("summary", "")).strip()
            if not summary:
                continue
            out.append(
                Brief(
                    ticker=ticker,
                    summary=summary,
                    sentiment=_clamp_sentiment(raw.get("sentiment")),
                    catalysts=_string_list(raw.get("catalysts")),
                    sources=_dedupe([it.url for it in grouped[ticker] if it.url]),
                    ts=ts,
                )
            )
        return out

    @staticmethod
    def _extract_briefs(content: str) -> list[dict[str, object]]:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("research model returned non-JSON; no briefs parsed")
            return []
        items = data.get("briefs", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [x for x in items if isinstance(x, dict)]


def _clamp_sentiment(value: object) -> float:
    try:
        s = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, s))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
