"""Gated reflection: turn a trading round into a *few* durable lessons.

Memory is not a journal. Reflection exists to keep only decision-changing
lessons, and to keep the store from flooding (the Artoo failure mode). Two
guards, always applied before anything is written:

- **Cap** — at most ``max_writes`` new lessons per reflection.
- **Dedup** — a candidate is dropped if it is near-identical (cosine ≥
  ``dedup_threshold``) to an existing active lesson *or* to one already accepted
  in the same batch.

:meth:`Reflector.distill` is the optional LLM step that produces candidate
lessons from a round/decision log. It calls a paid model, so it is **cost-gated**
(:class:`CostGate`, per ``CONTRACTS.md §Cost-gating``): an explicit trigger that
refuses to run once the user's daily $ ceiling is reached. The pure
:meth:`Reflector.reflect` path (cap + dedup over caller-supplied candidates) is
free and is what tests exercise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .store import Lesson, MemoryStore
from .vector.base import cosine_similarity

if TYPE_CHECKING:
    from ..config.endpoints import EndpointRegistry, ModelRef
    from ..config.settings_store import SettingsStore

DEFAULT_MAX_WRITES = 3
DEFAULT_DEDUP_THRESHOLD = 0.90
SPEND_KEY = "__daily_spend__"  # per-user settings key: {YYYY-MM-DD: usd_spent}


class CostGateError(RuntimeError):
    """A paid-model call was refused because it would exceed the daily ceiling."""


@dataclass
class Skipped:
    text: str
    reason: str


@dataclass
class ReflectionResult:
    written: list[Lesson] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)


class CostGate:
    """Per-user daily $ budget, persisted in the settings store.

    Spend is tracked under :data:`SPEND_KEY` keyed by UTC date so the ledger
    survives restarts and rolls over at midnight. The ceiling comes from the
    ``daily_usd_ceiling`` setting (WS-0 default 5.0).
    """

    def __init__(self, settings: SettingsStore) -> None:
        self._settings = settings

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def spent_today(self, user_id: str) -> float:
        rec = self._settings.get(user_id, SPEND_KEY, {}) or {}
        return float(rec.get(self._today(), 0.0))

    def remaining(self, user_id: str) -> float:
        ceiling = float(self._settings.get(user_id, "daily_usd_ceiling", 5.0))
        return max(0.0, ceiling - self.spent_today(user_id))

    def check(self, user_id: str, estimated_usd: float) -> None:
        if estimated_usd > self.remaining(user_id):
            raise CostGateError(
                f"daily budget reached: est ${estimated_usd:.4f} > "
                f"${self.remaining(user_id):.4f} remaining"
            )

    def record(self, user_id: str, usd: float) -> None:
        today = self._today()
        rec = dict(self._settings.get(user_id, SPEND_KEY, {}) or {})
        rec[today] = float(rec.get(today, 0.0)) + max(0.0, usd)
        # Keep only the last few days so the ledger never grows unbounded.
        rec = {d: v for d, v in rec.items() if d >= self._cutoff()}
        self._settings.set(user_id, SPEND_KEY, rec)

    @staticmethod
    def _cutoff() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7 * 86400))


_DISTILL_SYSTEM = (
    "You distill a trading agent's round into at most {n} durable, "
    "decision-changing lessons it should remember next time. Skip play-by-play "
    "and one-off noise. Reply ONLY as JSON: {{\"lessons\": [\"...\"]}}."
)


class Reflector:
    """Gates candidate lessons into a trader's memory; optional LLM distill."""

    def __init__(
        self,
        memory: MemoryStore,
        *,
        settings: SettingsStore | None = None,
        registry: EndpointRegistry | None = None,
        max_writes: int = DEFAULT_MAX_WRITES,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        self.memory = memory
        self.settings = settings
        self.registry = registry
        self.max_writes = max_writes
        self.dedup_threshold = dedup_threshold
        self.cost_gate = CostGate(settings) if settings is not None else None

    # --- the gate ------------------------------------------------------------

    def reflect(
        self,
        user_id: str,
        trader_id: str,
        candidates: list[str],
        *,
        tags: list[str] | None = None,
        max_writes: int | None = None,
        dedup_threshold: float | None = None,
    ) -> ReflectionResult:
        """Cap + dedup ``candidates`` and write the survivors. Always free."""
        cap = self.max_writes if max_writes is None else max_writes
        thr = self.dedup_threshold if dedup_threshold is None else dedup_threshold
        result = ReflectionResult()
        accepted_vectors: list[list[float]] = []

        for raw in candidates:
            text = raw.strip()
            if not text:
                result.skipped.append(Skipped(raw, "empty"))
                continue
            if len(result.written) >= cap:
                result.skipped.append(Skipped(raw, "over write cap"))
                continue
            # Near-duplicate of something already stored for this exact pair?
            existing = self.memory.recall(user_id, trader_id, text, k=1)
            if existing and existing[0].score is not None and existing[0].score >= thr:
                result.skipped.append(Skipped(raw, f"duplicate of {existing[0].id}"))
                continue
            # Near-duplicate of a sibling accepted earlier in this batch?
            vec = self.memory.embedder.embed(text)
            if any(cosine_similarity(vec, av) >= thr for av in accepted_vectors):
                result.skipped.append(Skipped(raw, "duplicate within batch"))
                continue
            lesson = self.memory.remember(user_id, trader_id, text, tags=tags)
            result.written.append(lesson)
            accepted_vectors.append(vec)
        return result

    # --- optional cost-gated LLM distill -------------------------------------

    def distill(
        self,
        user_id: str,
        trader_id: str,
        context: str,
        ref: ModelRef,
        *,
        max_candidates: int = 5,
        estimated_usd: float = 0.01,
    ) -> list[str]:
        """LLM step: extract candidate lessons from a round log. Cost-gated.

        Returns raw candidate strings (feed them to :meth:`reflect`). Raises
        :class:`CostGateError` if the daily ceiling would be exceeded, and
        requires a registry + settings (so spend is tracked and gated).
        """
        if self.registry is None or self.cost_gate is None:
            raise RuntimeError("distill needs a registry + settings for cost-gating")
        self.cost_gate.check(user_id, estimated_usd)
        messages = [
            {"role": "system", "content": _DISTILL_SYSTEM.format(n=max_candidates)},
            {"role": "user", "content": context},
        ]
        res = self.registry.chat(user_id, ref, messages, json_mode=True, temperature=0.2)
        # Charge actual cost if the provider reported it, else the estimate.
        self.cost_gate.record(user_id, res.cost if res.cost is not None else estimated_usd)
        return self._parse_lessons(res.content, max_candidates)

    def reflect_from_context(
        self,
        user_id: str,
        trader_id: str,
        context: str,
        ref: ModelRef,
        *,
        max_candidates: int = 5,
        estimated_usd: float = 0.01,
        tags: list[str] | None = None,
    ) -> ReflectionResult:
        """distill (paid, gated) → reflect (cap + dedup) in one call."""
        candidates = self.distill(
            user_id, trader_id, context, ref,
            max_candidates=max_candidates, estimated_usd=estimated_usd,
        )
        return self.reflect(user_id, trader_id, candidates, tags=tags)

    @staticmethod
    def _parse_lessons(content: str, limit: int) -> list[str]:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Model didn't return clean JSON: treat non-empty lines as lessons.
            lines = [ln.strip("-*• \t") for ln in content.splitlines() if ln.strip()]
            return lines[:limit]
        items = data.get("lessons", []) if isinstance(data, dict) else data
        out = [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else []
        return out[:limit]
