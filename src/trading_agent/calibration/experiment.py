"""P6 calibration experiment driver: ON vs OFF cohort A/B proof.

Design
------
``ExperimentDriver`` spins up two cohorts against a shared ``BenchController``
(or a minimal synthetic bench in tests):

* **intel_on**  — traders added with ``intelligence_flags={}`` (memory layer
  active, the default).
* **intel_off** — traders added with ``intelligence_flags={"memory": False}``
  (memory layer disabled).

The same model slug is used for both cohorts so the only variable is whether the
private memory layer is wired.  Both cohorts observe the same price sequence
(passed as a list of bar dicts) and each trader fires one ``decide()`` turn per
bar via ``bench.run_decisions()``.

After all rounds the driver harvests per-cohort metrics from the bench
leaderboard (P&L, win-rate, decision count, rough Sharpe-ish) and writes them to
``ExperimentStore``.  The store is queried by the new ``/api/calibration/experiment``
endpoints.

No LLM calls are needed in tests: the driver accepts a ``trader_factory``
callable so tests can inject mock traders that immediately return a HOLD
decision without touching the network.

Schema (stored inside the shared ``config.db``)
----------------------------------------------
``calibration_experiments``
    id, run_id, started_at, finished_at, model, rounds, status

``calibration_cohorts``
    id, run_id, label, intel_on, traders (JSON list of names)

``calibration_cohort_metrics``
    id, run_id, cohort_label, metric_name, metric_value
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bench.controller import BenchController
    from ..config.db import Database

# ── SQLite schema (idempotent) ─────────────────────────────────────────────────

EXPERIMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_experiments (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL UNIQUE,
    started_at  REAL NOT NULL,
    finished_at REAL,
    model       TEXT NOT NULL DEFAULT '',
    rounds      INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending'   -- pending | running | done | failed
);

CREATE TABLE IF NOT EXISTS calibration_cohorts (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    label       TEXT NOT NULL,    -- 'intel_on' or 'intel_off'
    intel_on    INTEGER NOT NULL, -- 1=True, 0=False
    traders     TEXT NOT NULL DEFAULT '[]'  -- JSON list of trader names
);

CREATE TABLE IF NOT EXISTS calibration_cohort_metrics (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    cohort_label TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL
);

CREATE INDEX IF NOT EXISTS idx_exp_run_id     ON calibration_experiments(run_id);
CREATE INDEX IF NOT EXISTS idx_cohort_run_id  ON calibration_cohorts(run_id);
CREATE INDEX IF NOT EXISTS idx_cmetric_run_id ON calibration_cohort_metrics(run_id, cohort_label);
"""


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class CohortMetrics:
    """Per-cohort aggregate from one experiment run."""

    run_id: str
    label: str          # 'intel_on' or 'intel_off'
    intel_on: bool
    trader_names: list[str] = field(default_factory=list)
    # leaderboard-derived
    mean_pnl: float | None = None
    mean_return_pct: float | None = None
    mean_win_rate: float | None = None
    mean_decisions: float | None = None
    sharpe_ish: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "intel_on": self.intel_on,
            "trader_names": self.trader_names,
            "mean_pnl": self.mean_pnl,
            "mean_return_pct": self.mean_return_pct,
            "mean_win_rate": self.mean_win_rate,
            "mean_decisions": self.mean_decisions,
            "sharpe_ish": self.sharpe_ish,
        }


@dataclass
class ExperimentRun:
    """Summary of one experiment."""

    run_id: str
    started_at: float
    finished_at: float | None
    model: str
    rounds: int
    status: str                    # pending | running | done | failed
    cohorts: list[CohortMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.model,
            "rounds": self.rounds,
            "status": self.status,
            "cohorts": [c.as_dict() for c in self.cohorts],
        }


# ── ExperimentStore ────────────────────────────────────────────────────────────


class ExperimentStore:
    """Read/write store for calibration experiment runs backed by ``config.db``."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._db.connect().executescript(EXPERIMENT_SCHEMA)

    # --- write ----------------------------------------------------------------

    def create_run(self, *, model: str) -> str:
        """Insert a new experiment run row; return the run_id."""
        run_id = uuid.uuid4().hex
        now = time.time()
        self._db.execute(
            "INSERT INTO calibration_experiments (id, run_id, started_at, model, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, run_id, now, model, "pending"),
        )
        return run_id

    def set_status(self, run_id: str, status: str) -> None:
        self._db.execute(
            "UPDATE calibration_experiments SET status=? WHERE run_id=?",
            (status, run_id),
        )

    def finish_run(self, run_id: str, rounds: int) -> None:
        now = time.time()
        self._db.execute(
            "UPDATE calibration_experiments SET finished_at=?, rounds=?, status=? "
            "WHERE run_id=?",
            (now, rounds, "done", run_id),
        )

    def fail_run(self, run_id: str) -> None:
        now = time.time()
        self._db.execute(
            "UPDATE calibration_experiments SET finished_at=?, status=? WHERE run_id=?",
            (now, "failed", run_id),
        )

    def record_cohort(
        self,
        run_id: str,
        label: str,
        *,
        intel_on: bool,
        trader_names: list[str],
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO calibration_cohorts (id, run_id, label, intel_on, traders) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, run_id, label, int(intel_on), json.dumps(trader_names)),
        )

    def record_metric(
        self,
        run_id: str,
        cohort_label: str,
        metric_name: str,
        metric_value: float | None,
    ) -> None:
        self._db.execute(
            "INSERT INTO calibration_cohort_metrics "
            "(id, run_id, cohort_label, metric_name, metric_value) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, run_id, cohort_label, metric_name, metric_value),
        )

    # --- read -----------------------------------------------------------------

    def get_run(self, run_id: str) -> ExperimentRun | None:
        row = self._db.query_one(
            "SELECT * FROM calibration_experiments WHERE run_id=?", (run_id,)
        )
        if row is None:
            return None
        return self._hydrate_run(row)

    def list_runs(self, *, limit: int = 50) -> list[ExperimentRun]:
        rows = self._db.query(
            "SELECT * FROM calibration_experiments ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [self._hydrate_run(r) for r in rows]

    # --- helpers --------------------------------------------------------------

    def _hydrate_run(self, row: Any) -> ExperimentRun:
        run_id = str(row["run_id"])
        cohort_rows = self._db.query(
            "SELECT * FROM calibration_cohorts WHERE run_id=?", (run_id,)
        )
        cohorts: list[CohortMetrics] = []
        for cr in cohort_rows:
            label = str(cr["label"])
            intel_on = bool(cr["intel_on"])
            trader_names: list[str] = json.loads(str(cr["traders"] or "[]"))
            metrics_rows = self._db.query(
                "SELECT metric_name, metric_value FROM calibration_cohort_metrics "
                "WHERE run_id=? AND cohort_label=?",
                (run_id, label),
            )
            m: dict[str, float | None] = {str(r["metric_name"]): r["metric_value"] for r in metrics_rows}
            cohorts.append(
                CohortMetrics(
                    run_id=run_id,
                    label=label,
                    intel_on=intel_on,
                    trader_names=trader_names,
                    mean_pnl=m.get("mean_pnl"),
                    mean_return_pct=m.get("mean_return_pct"),
                    mean_win_rate=m.get("mean_win_rate"),
                    mean_decisions=m.get("mean_decisions"),
                    sharpe_ish=m.get("sharpe_ish"),
                )
            )
        return ExperimentRun(
            run_id=run_id,
            started_at=float(row["started_at"]),
            finished_at=row["finished_at"],
            model=str(row["model"]),
            rounds=int(row["rounds"]),
            status=str(row["status"]),
            cohorts=cohorts,
        )


# ── ExperimentDriver ───────────────────────────────────────────────────────────

# Default cohort configuration.
_COHORT_ON_LABEL = "intel_on"
_COHORT_OFF_LABEL = "intel_off"

# Default number of simulated rounds per experiment.
DEFAULT_ROUNDS = 5

# Default number of traders per cohort.
DEFAULT_COHORT_SIZE = 1


class ExperimentDriver:
    """Spin up two cohorts (intelligence ON vs OFF), run them, record metrics.

    Parameters
    ----------
    controller:
        A ``BenchController`` instance with its bench and client pre-wired.
        The driver calls ``controller.add_model()`` twice for each cohort slot,
        sets the intel flags, then drives ``bench.run_decisions()`` for the
        requested number of rounds.
    store:
        Persists experiment runs and per-cohort metrics.
    model:
        OpenRouter model slug to use for all traders in both cohorts.
    rounds:
        Number of ``bench.run_decisions()`` calls (one per round).  Each round
        fans a decision turn to every registered trader.
    cohort_size:
        Number of traders per cohort.  Useful for averaging out random variance
        in tests.  Default 1.
    bars:
        Optional sequence of bar dicts to feed into ``bench.observe_bar()``
        before each round.  If omitted, traders decide on whatever prices were
        already seeded on the bench.
    trader_factory:
        Optional callable ``(model, name, intel_on) -> Trader`` that overrides
        the normal ``add_model()`` path.  Intended for tests that inject mock
        traders without LLM calls.  When provided, the trader is added to the
        bench directly via ``bench.add_competitor()`` and ``bind_execution()``
        is skipped (no broker binding needed in unit tests).
        Signature: ``factory(model: str, name: str, intel_on: bool) -> Trader``
    """

    def __init__(
        self,
        controller: BenchController,
        store: ExperimentStore,
        *,
        model: str = "test/model",
        rounds: int = DEFAULT_ROUNDS,
        cohort_size: int = DEFAULT_COHORT_SIZE,
        bars: list[dict[str, Any]] | None = None,
        trader_factory: Callable[[str, str, bool], Any] | None = None,
    ) -> None:
        self._controller = controller
        self._store = store
        self.model = model
        self.rounds = max(1, rounds)
        self.cohort_size = max(1, cohort_size)
        self.bars = list(bars or [])
        self._trader_factory = trader_factory

    # --- public API -----------------------------------------------------------

    def run(self) -> ExperimentRun:
        """Execute the experiment synchronously; return the completed run."""
        run_id = self._store.create_run(model=self.model)
        self._store.set_status(run_id, "running")
        try:
            result = self._execute(run_id)
        except Exception as exc:
            self._store.fail_run(run_id)
            raise RuntimeError(f"Experiment {run_id} failed: {exc}") from exc
        return result

    # --- private --------------------------------------------------------------

    def _execute(self, run_id: str) -> ExperimentRun:
        bench = self._controller.bench
        on_names: list[str] = []
        off_names: list[str] = []

        # Register traders for both cohorts.
        for i in range(self.cohort_size):
            on_name = self._add_trader(
                run_id=run_id,
                slot=i,
                intel_on=True,
            )
            off_name = self._add_trader(
                run_id=run_id,
                slot=i,
                intel_on=False,
            )
            on_names.append(on_name)
            off_names.append(off_name)

        # Record cohort membership.
        self._store.record_cohort(run_id, _COHORT_ON_LABEL, intel_on=True, trader_names=on_names)
        self._store.record_cohort(run_id, _COHORT_OFF_LABEL, intel_on=False, trader_names=off_names)

        # Run rounds.
        bars_iter = iter(self.bars)
        for _ in range(self.rounds):
            # Optionally feed one bar per round (or the same bar if exhausted).
            bar = next(bars_iter, None)
            if bar is not None:
                bench.observe_bar(bar)
            bench.run_decisions()

        # Harvest metrics and persist.
        leaderboard = {row["name"]: row for row in bench.leaderboard()}
        self._record_cohort_metrics(run_id, _COHORT_ON_LABEL, on_names, leaderboard)
        self._record_cohort_metrics(run_id, _COHORT_OFF_LABEL, off_names, leaderboard)

        self._store.finish_run(run_id, rounds=self.rounds)
        run = self._store.get_run(run_id)
        assert run is not None
        return run

    def _add_trader(self, run_id: str, slot: int, intel_on: bool) -> str:
        label = _COHORT_ON_LABEL if intel_on else _COHORT_OFF_LABEL
        name = f"{run_id[:8]}-{label}-{slot}"

        if self._trader_factory is not None:
            # Test path: inject a mock trader directly into the bench.
            trader = self._trader_factory(self.model, name, intel_on)
            self._controller.bench.add_competitor(name, trader)
            return name

        # Production path: use the controller to add, which honours intel flags.
        flags: dict[str, bool] = {} if intel_on else {"memory": False}
        actual_name = self._controller.add_model(
            self.model,
            name,
            intelligence_flags=flags,
            tutorial_remaining=0,
        )
        return actual_name

    def _record_cohort_metrics(
        self,
        run_id: str,
        cohort_label: str,
        trader_names: list[str],
        leaderboard: dict[str, Any],
    ) -> None:
        """Compute mean metrics over a cohort and persist them."""
        rows = [leaderboard[n] for n in trader_names if n in leaderboard]
        if not rows:
            return

        pnl_vals = [float(r.get("pnl", 0) or 0) for r in rows]
        ret_vals = [float(r.get("return_pct", 0) or 0) for r in rows]
        dec_vals = [int(r.get("decisions", 0) or 0) for r in rows]

        # Win-rate: wins / trades, None when no trades.
        wr_vals: list[float] = []
        for r in rows:
            trades = int(r.get("trades", 0) or 0)
            wins = int(r.get("wins", 0) or 0)
            if trades > 0:
                wr_vals.append(wins / trades)

        mean_pnl = _mean(pnl_vals)
        mean_ret = _mean(ret_vals)
        mean_dec = _mean([float(d) for d in dec_vals])
        mean_wr = _mean(wr_vals) if wr_vals else None

        # Sharpe-ish: mean_return_pct / sqrt(max(total_decisions, 1))
        total_dec = int(sum(dec_vals))
        sharpe = (mean_ret / math.sqrt(max(total_dec, 1))) if mean_ret is not None else None

        for metric_name, val in [
            ("mean_pnl", mean_pnl),
            ("mean_return_pct", mean_ret),
            ("mean_win_rate", mean_wr),
            ("mean_decisions", mean_dec),
            ("sharpe_ish", sharpe),
        ]:
            self._store.record_metric(run_id, cohort_label, metric_name, val)


# ── helpers ───────────────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)
