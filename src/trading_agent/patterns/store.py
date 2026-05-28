"""Pattern knowledge base store (P4): global, shared, regime-conditioned.

A pattern episode is stored two ways (mirroring ResearchStore's dual-storage
design):

- **Structured SQLite row** in ``pattern_episodes`` — enables exact-match,
  date-range, and label-filtered queries (the SQL job) without touching vectors.
- **Vector point** in the shared VectorStore under the ``global:patterns``
  collection — enables similarity-based "looks like now" recall.

**Global namespace.** Unlike research briefs (per-user) or lessons (per-trader),
patterns are objective market observations — no privacy boundary. Collection key
is ``global:patterns`` and every trader reads the same store.

**Regime-conditioned stats.** ``stats(label, regime=None)`` returns aggregate
forward hit-rate, calibration, and trade count conditioned on regime. The model
sees stats, NOT raw episodes (avoids flooding + preserves walk-forward validity).

**Calibration tracking.** A ``predicted_prob`` on each episode is compared to
the realized outcome to track how well model-predicted probabilities match
actual frequencies (the Brier-like calibration score).

**Decay-pruning.** Labels with hit_rate < ``prune_threshold`` (decaying toward
coin-flip) are archived — soft-deleted, never hard-deleted. The ``archive()``
call sets ``status='archived'`` and removes the vector point so future recall
doesn't surface it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config.db import Database
from ..memory.embed import Embedder, EmbedError
from ..memory.vector.base import VectorStore

GLOBAL_COLLECTION = "global:patterns"
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pattern_episodes (
    id                TEXT PRIMARY KEY,
    symbol            TEXT NOT NULL,
    label             TEXT NOT NULL,
    event_date        TEXT NOT NULL,            -- YYYY-MM-DD the pattern was observed
    regime            TEXT NOT NULL DEFAULT 'unknown',
    outcome           REAL,                     -- realized outcome (e.g. fwd_pct_5d)
    social_velocity   REAL NOT NULL DEFAULT 0.0,
    sentiment_quartile INTEGER NOT NULL DEFAULT 2, -- 1..4
    predicted_prob    REAL,                     -- model-estimated prob at decision time
    realized_hit      INTEGER,                  -- 1 = predicted direction hit, 0 = miss, NULL = open
    calibration_error REAL,                     -- (predicted_prob - realized_hit)^2 per episode
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        REAL NOT NULL,
    notes             TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pattern_label   ON pattern_episodes(label, regime, status);
CREATE INDEX IF NOT EXISTS idx_pattern_date    ON pattern_episodes(event_date, status);
CREATE INDEX IF NOT EXISTS idx_pattern_symbol  ON pattern_episodes(symbol, status);
"""


@dataclass
class PatternEpisode:
    """One market pattern observation."""

    symbol: str
    label: str
    event_date: str           # YYYY-MM-DD
    regime: str = "unknown"
    outcome: float | None = None
    social_velocity: float = 0.0
    sentiment_quartile: int = 2
    predicted_prob: float | None = None
    realized_hit: int | None = None
    calibration_error: float | None = None
    status: str = STATUS_ACTIVE
    notes: str = ""
    id: str = ""
    created_at: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "label": self.label,
            "event_date": self.event_date,
            "regime": self.regime,
            "outcome": self.outcome,
            "social_velocity": self.social_velocity,
            "sentiment_quartile": self.sentiment_quartile,
            "predicted_prob": self.predicted_prob,
            "realized_hit": self.realized_hit,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> PatternEpisode:
        # Support both sqlite3.Row (key access) and plain dicts.
        def _get(key: str, default: Any = None) -> Any:
            try:
                return row[key]
            except (KeyError, IndexError):
                return default

        return cls(
            id=str(_get("id") or ""),
            symbol=str(_get("symbol") or ""),
            label=str(_get("label") or ""),
            event_date=str(_get("event_date") or ""),
            regime=str(_get("regime") or "unknown"),
            outcome=_f(_get("outcome")),
            social_velocity=float(_get("social_velocity") or 0.0),
            sentiment_quartile=int(_get("sentiment_quartile") or 2),
            predicted_prob=_f(_get("predicted_prob")),
            realized_hit=_get("realized_hit"),
            calibration_error=_f(_get("calibration_error")),
            status=str(_get("status") or STATUS_ACTIVE),
            notes=str(_get("notes") or ""),
            created_at=float(_get("created_at") or 0.0),
        )


@dataclass
class PatternMatch:
    """A recalled pattern with regime-conditioned stats."""

    episode: PatternEpisode
    label: str
    regime: str
    stats: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # vector similarity


class PatternStore:
    """Global shared pattern KB: SQL + vector, regime-conditioned recall."""

    def __init__(
        self,
        db: Database,
        vector: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._db = db
        self._db.connect().executescript(SCHEMA)
        self._vector = vector
        self._embedder = embedder

    # --- write ----------------------------------------------------------------

    def add(self, episode: PatternEpisode) -> PatternEpisode:
        """Store a pattern episode (SQL row + best-effort vector point)."""
        if not episode.id:
            episode.id = uuid.uuid4().hex
        if not episode.created_at:
            episode.created_at = time.time()
        # Auto-compute calibration_error when both inputs are provided.
        if (
            episode.calibration_error is None
            and episode.predicted_prob is not None
            and episode.realized_hit is not None
        ):
            episode.calibration_error = (episode.predicted_prob - episode.realized_hit) ** 2
        self._db.execute(
            "INSERT OR REPLACE INTO pattern_episodes "
            "(id, symbol, label, event_date, regime, outcome, social_velocity, "
            " sentiment_quartile, predicted_prob, realized_hit, calibration_error, "
            " status, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.id,
                episode.symbol.upper(),
                episode.label,
                episode.event_date,
                episode.regime,
                episode.outcome,
                episode.social_velocity,
                episode.sentiment_quartile,
                episode.predicted_prob,
                episode.realized_hit,
                episode.calibration_error,
                episode.status,
                episode.created_at,
                episode.notes,
            ),
        )
        self._index(episode)
        return episode

    def update_outcome(
        self,
        episode_id: str,
        *,
        outcome: float,
        realized_hit: int,
        predicted_prob: float | None = None,
    ) -> None:
        """Record the realized outcome for an episode (P5 learning loop)."""
        calib: float | None = None
        if predicted_prob is not None:
            calib = (predicted_prob - realized_hit) ** 2
        self._db.execute(
            "UPDATE pattern_episodes SET outcome=?, realized_hit=?, "
            "calibration_error=? WHERE id=?",
            (outcome, realized_hit, calib, episode_id),
        )
        # Refresh vector payload with updated outcome.
        row = self._db.query_one(
            "SELECT * FROM pattern_episodes WHERE id=?", (episode_id,)
        )
        if row:
            ep = PatternEpisode.from_row(row)
            self._index(ep)

    def archive(self, episode_id: str) -> bool:
        """Soft-delete: flip status to archived + remove from vector index."""
        self._db.execute(
            "UPDATE pattern_episodes SET status=? WHERE id=?",
            (STATUS_ARCHIVED, episode_id),
        )
        if self._vector is not None:
            self._vector.delete(GLOBAL_COLLECTION, episode_id)
        return True

    # --- read (SQL) -----------------------------------------------------------

    def get(self, episode_id: str) -> PatternEpisode | None:
        row = self._db.query_one(
            "SELECT * FROM pattern_episodes WHERE id=?", (episode_id,)
        )
        return PatternEpisode.from_row(row) if row else None

    def by_label(
        self,
        label: str,
        *,
        regime: str | None = None,
        status: str = STATUS_ACTIVE,
        limit: int = 50,
    ) -> list[PatternEpisode]:
        """All episodes matching ``label`` (+ optionally ``regime``), newest first."""
        if regime:
            rows = self._db.query(
                "SELECT * FROM pattern_episodes WHERE label=? AND regime=? AND status=? "
                "ORDER BY event_date DESC LIMIT ?",
                (label, regime, status, limit),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM pattern_episodes WHERE label=? AND status=? "
                "ORDER BY event_date DESC LIMIT ?",
                (label, status, limit),
            )
        return [PatternEpisode.from_row(r) for r in rows]

    def by_date_range(
        self,
        from_date: str,
        to_date: str,
        *,
        symbol: str | None = None,
        label: str | None = None,
        status: str = STATUS_ACTIVE,
        limit: int = 100,
    ) -> list[PatternEpisode]:
        """Episodes in a date range (YYYY-MM-DD), optionally filtered."""
        clauses = ["event_date BETWEEN ? AND ?", "status = ?"]
        params: list[Any] = [from_date, to_date, status]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if label:
            clauses.append("label = ?")
            params.append(label)
        params.append(limit)
        sql = (
            "SELECT * FROM pattern_episodes WHERE "
            + " AND ".join(clauses)
            + " ORDER BY event_date DESC LIMIT ?"
        )
        return [PatternEpisode.from_row(r) for r in self._db.query(sql, tuple(params))]

    def stats(
        self,
        label: str,
        *,
        regime: str | None = None,
        min_n: int = 1,
    ) -> dict[str, Any]:
        """Regime-conditioned forward stats for ``label``.

        Returns a dict: {label, regime, n, hit_rate, mean_outcome,
        calibration_brier, has_sufficient_data}.
        """
        if regime:
            rows = self._db.query(
                "SELECT realized_hit, outcome, calibration_error "
                "FROM pattern_episodes "
                "WHERE label=? AND regime=? AND status=? AND realized_hit IS NOT NULL",
                (label, regime, STATUS_ACTIVE),
            )
        else:
            rows = self._db.query(
                "SELECT realized_hit, outcome, calibration_error "
                "FROM pattern_episodes "
                "WHERE label=? AND status=? AND realized_hit IS NOT NULL",
                (label, STATUS_ACTIVE),
            )
        rows_list = list(rows)
        hits = [int(r["realized_hit"]) for r in rows_list]
        outcomes = [_f(r["outcome"]) for r in rows_list if r["outcome"] is not None]
        calib_errors = [_f(r["calibration_error"]) for r in rows_list if r["calibration_error"] is not None]

        n = len(hits)
        hit_rate: float | None = (sum(hits) / n) if n > 0 else None
        mean_outcome: float | None = (sum(o for o in outcomes if o is not None) / len(outcomes)) if outcomes else None
        brier: float | None = (sum(e for e in calib_errors if e is not None) / len(calib_errors)) if calib_errors else None

        return {
            "label": label,
            "regime": regime,
            "n": n,
            "hit_rate": hit_rate,
            "mean_outcome": mean_outcome,
            "calibration_brier": brier,
            "has_sufficient_data": n >= min_n,
        }

    def decaying_labels(self, *, hit_rate_floor: float = 0.52) -> list[str]:
        """Labels whose active hit-rate is below ``hit_rate_floor`` (P5 decay)."""
        labels_raw = self._db.query(
            "SELECT DISTINCT label FROM pattern_episodes WHERE status=?",
            (STATUS_ACTIVE,),
        )
        decaying: list[str] = []
        for row in labels_raw:
            s = self.stats(row["label"])
            if s["n"] < 5:
                continue  # not enough data to judge
            if s["hit_rate"] is not None and s["hit_rate"] < hit_rate_floor:
                decaying.append(row["label"])
        return decaying

    def count(self, status: str = STATUS_ACTIVE) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM pattern_episodes WHERE status=?", (status,)
        )
        return int(row["n"]) if row else 0

    # --- read (semantic) -----------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        label: str | None = None,
        regime: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[PatternMatch]:
        """Hybrid recall: vector similarity first, then enrich with SQL stats.

        Falls back to label-filtered SQL if no vector store/embedder is wired.
        """
        if self._vector is not None and self._embedder is not None:
            return self._vector_recall(query, k=k, label=label, regime=regime)
        return self._sql_recall(label=label, regime=regime, limit=k)

    def _vector_recall(
        self,
        query: str,
        *,
        k: int,
        label: str | None,
        regime: str | None,
    ) -> list[PatternMatch]:
        assert self._vector is not None and self._embedder is not None
        try:
            vec = self._embedder.embed(query)
        except EmbedError:
            return self._sql_recall(label=label, regime=regime, limit=k)
        flt: dict[str, Any] = {"status": STATUS_ACTIVE}
        if label:
            flt["label"] = label
        if regime:
            flt["regime"] = regime
        hits = self._vector.search(GLOBAL_COLLECTION, vec, k, flt=flt)
        out: list[PatternMatch] = []
        for h in hits:
            ep = PatternEpisode.from_row(_payload_to_row(h.payload))
            s = self.stats(ep.label, regime=ep.regime)
            out.append(PatternMatch(episode=ep, label=ep.label, regime=ep.regime, stats=s, score=h.score))
        return out

    def _sql_recall(
        self,
        *,
        label: str | None,
        regime: str | None,
        limit: int,
    ) -> list[PatternMatch]:
        eps = self.by_label(label or "", regime=regime, limit=limit) if label else []
        if not eps:
            # fallback: most recent active episodes
            rows = self._db.query(
                "SELECT * FROM pattern_episodes WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?",
                (STATUS_ACTIVE, limit),
            )
            eps = [PatternEpisode.from_row(r) for r in rows]
        out: list[PatternMatch] = []
        for ep in eps:
            s = self.stats(ep.label, regime=ep.regime)
            out.append(PatternMatch(episode=ep, label=ep.label, regime=ep.regime, stats=s))
        return out

    # --- private helpers ------------------------------------------------------

    def _index(self, episode: PatternEpisode) -> None:
        if self._vector is None or self._embedder is None:
            return
        if episode.status != STATUS_ACTIVE:
            return
        text = f"{episode.label} {episode.symbol} {episode.regime} {episode.event_date}"
        try:
            vec = self._embedder.embed(text)
        except EmbedError:
            return
        self._vector.upsert(GLOBAL_COLLECTION, episode.id, vec, episode.to_payload())


def _payload_to_row(p: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a DB-row-like dict from a vector point payload."""
    return {
        "id": p.get("id", ""),
        "symbol": p.get("symbol", ""),
        "label": p.get("label", ""),
        "event_date": p.get("event_date", ""),
        "regime": p.get("regime", "unknown"),
        "outcome": p.get("outcome"),
        "social_velocity": p.get("social_velocity", 0.0),
        "sentiment_quartile": p.get("sentiment_quartile", 2),
        "predicted_prob": p.get("predicted_prob"),
        "realized_hit": p.get("realized_hit"),
        "calibration_error": None,
        "status": p.get("status", STATUS_ACTIVE),
        "notes": "",
        "created_at": p.get("created_at", 0.0),
    }


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
