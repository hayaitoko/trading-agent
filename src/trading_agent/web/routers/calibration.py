"""Calibration router (P6): per-trader calibration + risk-adjusted P&L.

Read-only surface. Auth via the ``current_user`` pattern. Returns:
- Per-trader calibration (predicted_prob vs realized frequency, Brier score).
- Simple risk-adjusted P&L (Sharpe-ish: mean_return / std_return where possible).

The pattern KB is read from ``app.state.pattern_store`` (None → empty response).
The bench leaderboard is read from ``app.state.bench`` (None → empty response).

P6 experiment endpoints (``/api/calibration/experiment/...``) drive the A/B
cohort proof: intel-ON vs intel-OFF over the same inputs.  The
``ExperimentStore`` is resolved from ``app.state.experiment_store``; if absent,
the endpoints return 503.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


def _pattern_store(request: Request) -> Any:
    return getattr(request.app.state, "pattern_store", None)


def _bench(request: Request) -> Any:
    return getattr(request.app.state, "bench", None)


@router.get("/")
def get_calibration(
    request: Request,
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """Return calibration + risk-adjusted P&L for all active traders."""
    bench = _bench(request)
    pattern_store = _pattern_store(request)

    trader_stats = _trader_calibration(bench)
    kb_stats = _kb_calibration(pattern_store)

    return {
        "traders": trader_stats,
        "pattern_kb": kb_stats,
    }


@router.get("/traders/{trader_name}")
def get_trader_calibration(
    trader_name: str,
    request: Request,
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """Calibration + P&L for one specific trader."""
    bench = _bench(request)
    rows = _trader_calibration(bench)
    matched = [r for r in rows if r.get("name") == trader_name]
    if not matched:
        return {"name": trader_name, "error": "trader not found"}
    return matched[0]


@router.get("/labels")
def get_label_calibration(
    request: Request,
    user: str = Depends(current_user),
    label: str | None = None,
    regime: str | None = None,
) -> dict[str, Any]:
    """Regime-conditioned calibration stats for a pattern label (or all labels)."""
    pattern_store = _pattern_store(request)
    return _kb_calibration(pattern_store, label=label, regime=regime)


# --- helpers -----------------------------------------------------------------


def _trader_calibration(bench: Any) -> list[dict[str, Any]]:
    """Per-trader P&L stats with a simple Sharpe-ish ratio."""
    if bench is None:
        return []
    try:
        rows = bench.leaderboard()
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("name", "?")
        pnl = float(row.get("pnl", 0) or 0)
        ret_pct = float(row.get("return_pct", 0) or 0)
        trades = int(row.get("trades", 0) or 0)
        wins = int(row.get("wins", 0) or 0)
        win_rate = wins / trades if trades > 0 else None
        # Simple "Sharpe-ish": we don't have a daily series here, so we use
        # (return_pct) / sqrt(trades) as a rough risk-adjusted metric.
        sharpe_ish = ret_pct / math.sqrt(max(trades, 1))
        out.append({
            "name": name,
            "pnl": pnl,
            "return_pct": ret_pct,
            "trades": trades,
            "wins": wins,
            "win_rate": win_rate,
            "sharpe_ish": round(sharpe_ish, 4),
            "account_value": row.get("account_value"),
        })
    return out


def _kb_calibration(
    pattern_store: Any,
    *,
    label: str | None = None,
    regime: str | None = None,
) -> dict[str, Any]:
    """Calibration stats from the pattern KB."""
    if pattern_store is None:
        return {"available": False}
    try:
        from ...memory.reflect import LearningLoop

        loop = LearningLoop(pattern_store)
        return {"available": True, **loop.calibration_summary(label=label, regime=regime)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# ── Experiment endpoints (P6 A/B driver) ─────────────────────────────────────


def _experiment_store(request: Request) -> Any:
    return getattr(request.app.state, "experiment_store", None)


def _require_experiment_store(request: Request) -> Any:
    store = _experiment_store(request)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="experiment_store not configured on app.state",
        )
    return store


class RunExperimentBody(BaseModel):
    """Request body for POST /api/calibration/experiment/run."""

    model: str = "test/model"
    rounds: int = 5
    cohort_size: int = 1


@router.post("/experiment/run")
def run_experiment(
    body: RunExperimentBody,
    request: Request,
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """Trigger a new ON/OFF cohort experiment and return its result.

    Requires ``app.state.experiment_store`` and ``app.state.bench_controller``
    to be set (the cockpit serve path wires both).  Returns the completed
    :class:`~trading_agent.calibration.experiment.ExperimentRun` as JSON.
    """
    store = _require_experiment_store(request)
    controller = getattr(request.app.state, "bench_controller", None)
    if controller is None:
        raise HTTPException(status_code=503, detail="bench_controller not configured on app.state")

    from ...calibration.experiment import ExperimentDriver

    driver = ExperimentDriver(
        controller,
        store,
        model=body.model,
        rounds=body.rounds,
        cohort_size=body.cohort_size,
    )
    try:
        run = driver.run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return run.as_dict()


@router.get("/experiment/results")
def list_experiment_results(
    request: Request,
    user: str = Depends(current_user),
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most recent experiment runs (newest first)."""
    store = _require_experiment_store(request)
    return [r.as_dict() for r in store.list_runs(limit=limit)]


@router.get("/experiment/{run_id}")
def get_experiment(
    run_id: str,
    request: Request,
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """Return a single experiment run by its ``run_id``."""
    store = _require_experiment_store(request)
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"experiment {run_id!r} not found")
    return run.as_dict()
