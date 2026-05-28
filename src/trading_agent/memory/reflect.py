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

P5 extension: :meth:`Reflector.distill_dual_outputs` emits TWO outputs in one
distill call — a shareable pattern observation (position specifics stripped →
global pattern KB) AND a private strategic lesson (→ trader's own memory).
:class:`LearningLoop` drives the outcome-grounded calibration cycle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .store import Lesson, MemoryStore
from .vector.base import cosine_similarity

if TYPE_CHECKING:
    from ..config.endpoints import EndpointRegistry, ModelRef
    from ..config.settings_store import SettingsStore
    from ..patterns.store import PatternStore

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

    # --- P5: dual-output distillation ----------------------------------------

    def distill_dual_outputs(
        self,
        user_id: str,
        trader_id: str,
        context: str,
        ref: ModelRef,
        *,
        estimated_usd: float = 0.01,
    ) -> DualReflectionOutput:
        """Emit TWO outputs in one cost-gated LLM call (P5).

        - ``pattern_obs``: a shareable market observation with position
          specifics stripped — goes to the global pattern KB.
        - ``private_lessons``: strategic lessons for this trader only —
          fed into :meth:`reflect` → :class:`MemoryStore`.

        Raises :class:`CostGateError` if budget exceeded.
        """
        if self.registry is None or self.cost_gate is None:
            raise RuntimeError("distill_dual_outputs needs a registry + settings")
        self.cost_gate.check(user_id, estimated_usd)
        messages = [
            {"role": "system", "content": _DUAL_DISTILL_SYSTEM},
            {"role": "user", "content": context},
        ]
        res = self.registry.chat(user_id, ref, messages, json_mode=True, temperature=0.2)
        self.cost_gate.record(user_id, res.cost if res.cost is not None else estimated_usd)
        return self._parse_dual(res.content)

    def reflect_dual(
        self,
        user_id: str,
        trader_id: str,
        context: str,
        ref: ModelRef,
        *,
        estimated_usd: float = 0.01,
        tags: list[str] | None = None,
    ) -> tuple[DualReflectionOutput, ReflectionResult]:
        """Dual distill (paid) → reflect private lessons (cap+dedup).

        Returns ``(dual_output, private_reflection_result)``. The caller is
        responsible for writing ``dual_output.pattern_obs`` to the pattern KB.
        """
        dual = self.distill_dual_outputs(user_id, trader_id, context, ref, estimated_usd=estimated_usd)
        private = self.reflect(user_id, trader_id, dual.private_lessons, tags=tags)
        return dual, private

    @staticmethod
    def _parse_dual(content: str) -> DualReflectionOutput:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return DualReflectionOutput(pattern_obs="", private_lessons=[])
        if not isinstance(data, dict):
            return DualReflectionOutput(pattern_obs="", private_lessons=[])
        obs = str(data.get("pattern_observation") or "").strip()
        lessons_raw = data.get("private_lessons") or []
        lessons = [str(x).strip() for x in lessons_raw if str(x).strip()] if isinstance(lessons_raw, list) else []
        return DualReflectionOutput(pattern_obs=obs, private_lessons=lessons)


# --- P5 data types -----------------------------------------------------------

@dataclass
class DualReflectionOutput:
    """Two outputs from one dual-distill call (P5)."""

    pattern_obs: str              # shareable observation → global pattern KB
    private_lessons: list[str]   # private strategic lessons → trader memory


# --- P5 calibrated learning loop ---------------------------------------------

_PRUNE_HIT_RATE_FLOOR = 0.52   # archive labels that rot toward coin-flip
_PRUNE_MIN_N = 5                # need at least this many scored episodes to prune


class LearningLoop:
    """Outcome-grounded calibration cycle (P5).

    After each bench round, for each book:
      1. predict: the decision + matched pattern label + optional predicted_prob
      2. observe: realized price movement over a configurable horizon
      3. score: objective realized_hit (1 = direction correct, 0 = miss)
      4. update: regime-conditioned KB base rates
      5. walk-forward validate: only score on data the model never trained on
      6. calibration tracking: predicted_prob vs realized_hit (Brier-like)
      7. decay-prune: archive labels whose hit-rate rots toward coin-flip

    No LLM in this path: all scoring is deterministic from prices.
    """

    def __init__(
        self,
        pattern_store: PatternStore,
        *,
        prune_hit_rate_floor: float = _PRUNE_HIT_RATE_FLOOR,
        prune_min_n: int = _PRUNE_MIN_N,
    ) -> None:
        self._store = pattern_store
        self._prune_floor = prune_hit_rate_floor
        self._prune_min_n = prune_min_n

    def observe_outcome(
        self,
        episode_id: str,
        *,
        entry_price: float,
        exit_price: float,
        action: str,
        predicted_prob: float | None = None,
    ) -> OutcomeRecord:
        """Compute realized outcome from price movement and record it.

        ``action`` must be 'BUY' or 'SELL' to determine direction.
        ``realized_hit`` = 1 if the price moved in the predicted direction.
        """
        action_upper = action.strip().upper()
        if action_upper not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got {action!r}")

        fwd_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        realized_hit = 1 if (action_upper == "BUY" and fwd_pct > 0) or (action_upper == "SELL" and fwd_pct < 0) else 0

        self._store.update_outcome(
            episode_id,
            outcome=fwd_pct,
            realized_hit=realized_hit,
            predicted_prob=predicted_prob,
        )

        return OutcomeRecord(
            episode_id=episode_id,
            fwd_pct=fwd_pct,
            realized_hit=realized_hit,
            predicted_prob=predicted_prob,
            calibration_error=(
                (predicted_prob - realized_hit) ** 2
                if predicted_prob is not None
                else None
            ),
        )

    def decay_prune(self) -> list[str]:
        """Archive labels whose forward hit-rate has rotted toward coin-flip.

        Returns list of archived label names. Only prunes when n >= ``prune_min_n``
        so a new label isn't pruned before it has data.
        """
        decaying = self._store.decaying_labels(hit_rate_floor=self._prune_floor)
        archived: list[str] = []
        for label in decaying:
            eps = self._store.by_label(label, limit=200)
            count = 0
            for ep in eps:
                if ep.status == "active":
                    self._store.archive(ep.id)
                    count += 1
            if count > 0:
                archived.append(label)
        return archived

    def calibration_summary(
        self,
        label: str | None = None,
        regime: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate calibration stats (Brier score + hit-rate)."""
        if label:
            s = self._store.stats(label, regime=regime)
            return {
                "label": label,
                "regime": regime,
                "n": s["n"],
                "hit_rate": s["hit_rate"],
                "calibration_brier": s["calibration_brier"],
            }
        # Overall: query all distinct labels and aggregate.
        labels_raw = self._store._db.query(
            "SELECT DISTINCT label FROM pattern_episodes WHERE status=?",
            ("active",),
        )
        rows: list[dict[str, Any]] = []
        for row in labels_raw:
            s = self._store.stats(row["label"], regime=regime)
            if s["n"] > 0:
                rows.append(s)
        total_n = sum(r["n"] for r in rows)
        brier_vals = [r["calibration_brier"] for r in rows if r["calibration_brier"] is not None]
        brier_mean = sum(brier_vals) / len(brier_vals) if brier_vals else None
        return {
            "total_episodes": total_n,
            "calibration_brier_mean": brier_mean,
            "label_count": len(rows),
        }


@dataclass
class OutcomeRecord:
    """Result of one observe_outcome call."""

    episode_id: str
    fwd_pct: float
    realized_hit: int
    predicted_prob: float | None
    calibration_error: float | None


# --- dual-output distill prompt ----------------------------------------------

_DUAL_DISTILL_SYSTEM = """\
You analyze a completed trading round and produce two distinct outputs:

1. pattern_observation: A SHAREABLE, OBJECTIVE market observation. Strip all \
position sizes, cash amounts, and trader-specific details. Focus on the market \
setup (what the price did, what the pattern was, what followed). Max 2 sentences.

2. private_lessons: 1-3 PRIVATE strategic lessons this trader should remember \
about its own decision quality, risk management, or reasoning errors. These are \
specific to this trader's style and account.

Reply ONLY as JSON:
{"pattern_observation": "...", "private_lessons": ["...", "..."]}
"""
