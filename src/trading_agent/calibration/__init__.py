"""P6 calibration experiment driver.

:class:`ExperimentStore` persists A/B cohort runs and per-cohort metrics.
:class:`ExperimentDriver` spins up the ON/OFF cohorts, runs them against a
shared set of price observations, and records results into the store.

The store is read-only from the calibration router — the existing
``/api/calibration`` endpoints pull leaderboard stats; the experiment endpoints
added below expose the A/B summary.
"""

from .experiment import CohortMetrics, ExperimentDriver, ExperimentRun, ExperimentStore

__all__ = ["CohortMetrics", "ExperimentDriver", "ExperimentRun", "ExperimentStore"]
