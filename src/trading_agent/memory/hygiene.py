"""Memory hygiene — ported from Artoo's discipline, with namespacing Artoo lacks.

Three scheduler-runnable passes, all operating **strictly within a (user, trader)
scope** so cleanup never bleeds across traders:

- :meth:`Hygiene.dedup` — collapse near-duplicate active lessons. A pair counts
  as duplicate if it is **semantically** close (cosine ≥ ``semantic_threshold``)
  or **lexically** close (BM25 overlap ≥ ``bm25_threshold``) while still
  semantically related. The earliest-learned lesson is kept; the rest are
  archived.
- :meth:`Hygiene.sweep_stale` — archive lessons untouched for more than
  ``max_age_days`` (cold memory).
- :meth:`Hygiene.run` — both passes, returning a combined report.

Everything is a **soft-delete**: ``status`` flips to ``archived`` (recoverable
via :meth:`MemoryStore.restore`). Nothing is ever hard-deleted silently.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

from .embed import _tokenize
from .store import Lesson, MemoryStore
from .vector.base import cosine_similarity

DEFAULT_SEMANTIC_THRESHOLD = 0.93
DEFAULT_BM25_THRESHOLD = 0.6
# Lexical-only matches still need *some* semantic agreement to be a dup.
LEXICAL_SEMANTIC_FLOOR = 0.80
DEFAULT_MAX_AGE_DAYS = 90


@dataclass
class HygieneReport:
    deduped: list[tuple[str, str]] = field(default_factory=list)  # (archived_id, kept_id)
    archived_stale: list[str] = field(default_factory=list)
    scanned: int = 0


class BM25:
    """Compact Okapi BM25 over a fixed corpus of token lists."""

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf = [Counter(doc) for doc in corpus]
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query: list[str], idx: int) -> float:
        if self.avgdl == 0.0:
            return 0.0
        tf = self.tf[idx]
        dl = self.lengths[idx]
        total = 0.0
        for term in query:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return total

    def similarity(self, i: int, j: int, tokens: list[list[str]]) -> float:
        """Normalized, symmetric lexical similarity in roughly [0, 1]."""
        self_i = self.score(tokens[i], i) or 1.0
        self_j = self.score(tokens[j], j) or 1.0
        return max(self.score(tokens[i], j) / self_i, self.score(tokens[j], i) / self_j)


class Hygiene:
    """Dedup + staleness passes over a MemoryStore, per (user, trader)."""

    def __init__(
        self,
        memory: MemoryStore,
        *,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        bm25_threshold: float = DEFAULT_BM25_THRESHOLD,
    ) -> None:
        self.memory = memory
        self.semantic_threshold = semantic_threshold
        self.bm25_threshold = bm25_threshold

    # --- scope helpers -------------------------------------------------------

    def _traders(self, user_id: str, trader_id: str | None) -> list[str]:
        if trader_id is not None:
            return [trader_id]
        seen: dict[str, None] = {}
        for lesson in self.memory.list(user_id):
            seen.setdefault(lesson.trader_id, None)
        return list(seen)

    # --- dedup ---------------------------------------------------------------

    def dedup(self, user_id: str, trader_id: str | None = None) -> HygieneReport:
        report = HygieneReport()
        for tid in self._traders(user_id, trader_id):
            report = self._dedup_one(user_id, tid, report)
        return report

    def _dedup_one(self, user_id: str, trader_id: str, report: HygieneReport) -> HygieneReport:
        # Earliest first: the first occurrence of a duplicate is the one we keep.
        lessons = sorted(
            self.memory.list(user_id, trader_id), key=lambda lesson: lesson.created_at
        )
        report.scanned += len(lessons)
        if len(lessons) < 2:
            return report
        tokens = [_tokenize(lesson.text) for lesson in lessons]
        vectors = [self.memory.embedder.embed(lesson.text) for lesson in lessons]
        bm25 = BM25(tokens)

        kept: list[int] = []
        for j, lesson in enumerate(lessons):
            dup_of = self._duplicate_of(j, kept, vectors, tokens, bm25)
            if dup_of is None:
                kept.append(j)
                continue
            if self.memory.archive(user_id, lesson.id):
                report.deduped.append((lesson.id, lessons[dup_of].id))
        return report

    def _duplicate_of(
        self,
        j: int,
        kept: list[int],
        vectors: list[list[float]],
        tokens: list[list[str]],
        bm25: BM25,
    ) -> int | None:
        for i in kept:
            semantic = cosine_similarity(vectors[i], vectors[j])
            if semantic >= self.semantic_threshold:
                return i
            if (
                bm25.similarity(i, j, tokens) >= self.bm25_threshold
                and semantic >= LEXICAL_SEMANTIC_FLOOR
            ):
                return i
        return None

    # --- staleness -----------------------------------------------------------

    def sweep_stale(
        self,
        user_id: str,
        *,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        trader_id: str | None = None,
        now: float | None = None,
    ) -> HygieneReport:
        report = HygieneReport()
        cutoff = (now if now is not None else time.time()) - max_age_days * 86400
        lessons: list[Lesson] = (
            self.memory.list(user_id, trader_id)
            if trader_id is not None
            else self.memory.list(user_id)
        )
        report.scanned = len(lessons)
        for lesson in lessons:
            if lesson.updated_at < cutoff and self.memory.archive(user_id, lesson.id):
                report.archived_stale.append(lesson.id)
        return report

    # --- combined ------------------------------------------------------------

    def run(
        self,
        user_id: str,
        *,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        trader_id: str | None = None,
        now: float | None = None,
    ) -> HygieneReport:
        """Staleness sweep then dedup. A scheduler can call this on a cadence."""
        report = self.sweep_stale(
            user_id, max_age_days=max_age_days, trader_id=trader_id, now=now
        )
        dd = self.dedup(user_id, trader_id)
        report.deduped.extend(dd.deduped)
        report.scanned += dd.scanned
        return report
