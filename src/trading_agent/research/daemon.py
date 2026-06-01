"""Background research daemon — autonomous ingest + research on an hourly cadence.

Runs as a daemon thread alongside the feed and bench-cadence threads in
``cockpit_main``.  On each cycle:

1. Runs a full ingest pass (all enabled sources for users who have any) via
   :class:`~trading_agent.ingest.worker.IngestWorker`.
2. Runs a research pass via :class:`~trading_agent.research.agent.ResearchAgent`
   — cost-gated via :class:`~trading_agent.memory.reflect.CostGate`.  Returns
   immediately when the ingest backlog is empty (no model call, no spend).

Pre-SoD hydration hook
----------------------
When the bench is running under a :class:`~trading_agent.bench.scheduler.MarketScheduler`
the daemon also installs a *pre-SoD* hook: on every cadence tick it checks
whether a SoD turn is about to fire (i.e. the current time falls inside the
``[sod_utc - PRE_SOD_WINDOW_MINUTES, sod_utc)`` window and a research pass has
not yet been run today).  When true it runs an extra ingest+research cycle so
fresh overnight data is in the brief store before any trader's SoD turn reads it.

Failure isolation
-----------------
Any source-level or research-level exception is caught and logged; the daemon
never dies from a single cycle failure.  Missing API keys (Alpaca, OpenRouter,
etc.) cause the relevant step to degrade gracefully rather than crash.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ..intel.lifecycle import compute_live_window

if TYPE_CHECKING:
    from ..config.db import Database
    from ..config.endpoints import EndpointRegistry
    from ..config.settings_store import SettingsStore
    from ..ingest.registry import SourceRegistry
    from ..ingest.store import IngestStore
    from ..research.agent import ResearchAgent
    from ..research.store import ResearchStore

logger = logging.getLogger(__name__)

# Default cadence: run ingest+research every hour.
DEFAULT_CADENCE_SECONDS: int = 3600

# Pre-SoD window: run a research pass when we are within this many minutes of
# the scheduled SoD turn.  Must be comfortably less than SOD_LEAD_MINUTES (60).
PRE_SOD_WINDOW_MINUTES: int = 15


def _users_with_sources(db: Database) -> list[str]:
    """Return user_ids that have at least one enabled ingest source."""
    try:
        rows = db.query("SELECT DISTINCT user_id FROM sources WHERE enabled = 1")
        return [r["user_id"] for r in rows]
    except Exception:
        return []


def _run_ingest_cycle(
    ingest_store: IngestStore,
    source_registry: SourceRegistry,
    user_id: str,
) -> int:
    """Run one async ingest cycle for *user_id* in a fresh event loop.

    Returns the number of new items written.  Any exception is swallowed and
    logged so the daemon never dies on a source failure.
    """
    from ..ingest.worker import IngestWorker

    worker = IngestWorker(ingest_store, source_registry)
    try:
        return asyncio.run(worker.run_once(user_id))
    except Exception:
        logger.exception("research_daemon: ingest cycle failed for user=%s", user_id)
        return 0


def _run_research_pass(
    research_agent: ResearchAgent,
    user_id: str,
    ref: Any,
) -> int:
    """Run one research pass for *user_id*; return the number of briefs written.

    Swallows :class:`~trading_agent.memory.reflect.CostGateError` (budget
    reached) and any other exception, logging both without crashing the daemon.
    """
    from ..memory.reflect import CostGateError

    try:
        briefs = research_agent.run(user_id, None, ref)
        return len(briefs)
    except CostGateError as e:
        logger.info("research_daemon: cost gate blocked research for user=%s: %s", user_id, e)
        return 0
    except Exception:
        logger.exception("research_daemon: research pass failed for user=%s", user_id)
        return 0


class ResearchDaemon:
    """Background daemon: hourly ingest + research, with optional pre-SoD hydration.

    Parameters
    ----------
    db:
        Shared :class:`~trading_agent.config.db.Database` (same as the app's).
    ingest_store:
        :class:`~trading_agent.ingest.store.IngestStore` backed by *db*.
    source_registry:
        :class:`~trading_agent.ingest.registry.SourceRegistry` backed by *db*.
    research_agent:
        :class:`~trading_agent.research.agent.ResearchAgent` backed by *db*.
    registry:
        :class:`~trading_agent.config.endpoints.EndpointRegistry` used to
        resolve the research model ref per user.
    settings:
        :class:`~trading_agent.config.settings_store.SettingsStore` for
        per-user settings (research model, cost ceiling, etc.).
    cadence_seconds:
        Seconds between full ingest+research cycles.  Default: 3600 (hourly).
    scheduler:
        Optional :class:`~trading_agent.bench.scheduler.MarketScheduler`.
        When provided the daemon installs a pre-SoD hydration hook.
    """

    def __init__(
        self,
        db: Database,
        ingest_store: IngestStore,
        source_registry: SourceRegistry,
        research_agent: ResearchAgent,
        registry: EndpointRegistry,
        settings: SettingsStore,
        *,
        cadence_seconds: int = DEFAULT_CADENCE_SECONDS,
        scheduler: Any = None,
    ) -> None:
        self._db = db
        self._ingest = ingest_store
        self._src_reg = source_registry
        self._agent = research_agent
        self._registry = registry
        self._settings = settings
        self.cadence_seconds = cadence_seconds
        self._scheduler = scheduler
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Pre-SoD tracking: date string of the last day we ran a pre-SoD pass.
        self._pre_sod_ran_date: str | None = None
        self._pre_sod_lock = threading.Lock()

    def start(self) -> None:
        """Start the daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="research-daemon", daemon=True
        )
        self._thread.start()
        logger.info(
            "research_daemon: started (cadence=%ds, pre-SoD hydration=%s)",
            self.cadence_seconds,
            "enabled" if self._scheduler is not None else "disabled",
        )

    def stop(self) -> None:
        """Signal the daemon to stop and return immediately."""
        self._stop.set()

    def _resolve_ref(self, user_id: str) -> Any | None:
        """Resolve the research ModelRef for *user_id*.

        Returns None when no usable endpoint is configured (graceful degrade).
        """
        try:
            from ..manager.agent import resolve_reflection_ref
            return resolve_reflection_ref(self._settings, self._registry, user_id)
        except Exception:
            return None

    def _run_cycle_for_users(self, users: list[str]) -> None:
        """Run one ingest+research cycle for each user in *users*."""
        for user_id in users:
            written = _run_ingest_cycle(self._ingest, self._src_reg, user_id)
            logger.debug(
                "research_daemon: ingest user=%s new_items=%d", user_id, written
            )
            ref = self._resolve_ref(user_id)
            if ref is None:
                logger.debug(
                    "research_daemon: no research model for user=%s, skipping research", user_id
                )
                continue
            n = _run_research_pass(self._agent, user_id, ref)
            logger.info(
                "research_daemon: research user=%s briefs_written=%d", user_id, n
            )

    def _check_pre_sod(self, users: list[str]) -> None:
        """Run a pre-SoD hydration pass if the SoD window is approaching.

        Uses the scheduler's calendar (via ``MarketScheduler._calendar``) to
        determine the SoD time for today.  If we are within
        ``PRE_SOD_WINDOW_MINUTES`` of SoD and haven't run a pre-SoD pass
        today, runs one immediately.
        """
        if self._scheduler is None:
            return
        try:
            now = datetime.now(UTC)
            calendar = self._scheduler._calendar  # type: ignore[attr-defined]
            window = compute_live_window(calendar, now)
            if window is None:
                return  # non-trading day or calendar unavailable

            sod_utc = window.sod_utc
            pre_sod_start = sod_utc - timedelta(minutes=PRE_SOD_WINDOW_MINUTES)
            date_str = window.date_et  # ET date string e.g. "2026-05-31"

            with self._pre_sod_lock:
                already_ran = self._pre_sod_ran_date == date_str

            if already_ran:
                return  # already hydrated today

            if pre_sod_start <= now < sod_utc:
                logger.info(
                    "research_daemon: pre-SoD hydration triggered (SoD at %s UTC)",
                    sod_utc.strftime("%H:%M"),
                )
                self._run_cycle_for_users(users)
                with self._pre_sod_lock:
                    self._pre_sod_ran_date = date_str
        except Exception:
            logger.exception("research_daemon: pre-SoD check failed")

    def _run(self) -> None:
        """Main daemon loop."""
        while not self._stop.is_set():
            users = _users_with_sources(self._db)
            if not users:
                logger.debug("research_daemon: no users with enabled sources; sleeping")
            else:
                try:
                    self._run_cycle_for_users(users)
                except Exception:
                    logger.exception("research_daemon: cycle-level failure")

            # Wait for the cadence interval, checking for pre-SoD every minute.
            elapsed = 0
            interval = min(60, self.cadence_seconds)
            while elapsed < self.cadence_seconds and not self._stop.is_set():
                self._stop.wait(interval)
                elapsed += interval
                # Check pre-SoD on every minute tick (cheap — no network call).
                if users:
                    self._check_pre_sod(users)


def build_research_daemon(
    app_state: Any,
    *,
    cadence_seconds: int = DEFAULT_CADENCE_SECONDS,
    scheduler: Any = None,
) -> ResearchDaemon | None:
    """Build a :class:`ResearchDaemon` from ``app.state`` attributes.

    Returns ``None`` when the required state is unavailable (no db, no ingest
    store, no research store).  The caller can safely ignore a None return —
    the cockpit runs without autonomous research in that case.

    Parameters
    ----------
    app_state:
        The FastAPI ``app.state`` object after ``build_cockpit`` has run.
    cadence_seconds:
        Override the default hourly cadence.
    scheduler:
        The :class:`~trading_agent.bench.scheduler.MarketScheduler` for
        pre-SoD hydration.  None disables the pre-SoD hook.
    """
    db = getattr(app_state, "db", None)
    endpoints = getattr(app_state, "endpoints", None)
    settings = getattr(app_state, "settings", None)
    research: ResearchStore | None = getattr(app_state, "research", None)

    if db is None or endpoints is None or settings is None or research is None:
        logger.info("research_daemon: prerequisites missing; autonomous research disabled")
        return None

    try:
        from ..ingest.registry import SourceRegistry
        from ..ingest.store import IngestStore
        from .agent import ResearchAgent

        ingest_store = IngestStore(db)
        src_reg = SourceRegistry(db)
        agent = ResearchAgent(ingest_store, research, endpoints, settings)

        return ResearchDaemon(
            db,
            ingest_store,
            src_reg,
            agent,
            endpoints,
            settings,
            cadence_seconds=cadence_seconds,
            scheduler=scheduler,
        )
    except Exception:
        logger.exception("research_daemon: failed to build; autonomous research disabled")
        return None
