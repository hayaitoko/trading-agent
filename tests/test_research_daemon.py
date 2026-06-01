"""Tests for the ResearchDaemon: cadence scheduling + pre-SoD hydration.

All tests are purely synchronous / deterministic. No network calls are made.
The IngestWorker's async path is stubbed so tests don't actually call asyncio.run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from trading_agent.config.db import Database
from trading_agent.config.endpoints import EndpointRegistry, ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.ingest.registry import SourceRegistry
from trading_agent.ingest.store import IngestStore
from trading_agent.intel.lifecycle import LiveWindow
from trading_agent.memory.embed import FakeEmbedder
from trading_agent.memory.reflect import CostGateError
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.research.agent import ResearchAgent
from trading_agent.research.daemon import (
    PRE_SOD_WINDOW_MINUTES,
    ResearchDaemon,
    _run_ingest_cycle,
    _run_research_pass,
    _users_with_sources,
    build_research_daemon,
)
from trading_agent.research.store import ResearchStore

USER = "u1"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Any) -> Database:
    return Database(tmp_path / "config.db")


def _make_stores(tmp_path: Any) -> tuple[Database, IngestStore, SourceRegistry, ResearchStore]:
    db = _make_db(tmp_path)
    ingest = IngestStore(db)
    src_reg = SourceRegistry(db)
    vec = SqliteVecStore(tmp_path / "memory.db")
    research = ResearchStore(db, vec, FakeEmbedder())
    return db, ingest, src_reg, research


def _make_agent(
    db: Database, ingest: IngestStore, research: ResearchStore, tmp_path: Any
) -> ResearchAgent:
    from trading_agent.config.endpoints import EndpointRegistry
    from trading_agent.config.settings_store import SettingsStore

    registry = EndpointRegistry(db)
    settings = SettingsStore(db)
    return ResearchAgent(ingest, research, registry, settings)


def _insert_source(db: Database, user_id: str) -> None:
    """Insert a minimal enabled ingest source row for *user_id*."""
    import uuid

    db.connect().execute(
        """
        INSERT OR IGNORE INTO sources (id, user_id, kind, name, config_json, enabled)
        VALUES (?, ?, 'rss', 'test-feed', '{}', 1)
        """,
        (str(uuid.uuid4()), user_id),
    )


# ---------------------------------------------------------------------------
# _users_with_sources
# ---------------------------------------------------------------------------


def test_users_with_sources_empty(tmp_path: Any) -> None:
    db = _make_db(tmp_path)
    assert _users_with_sources(db) == []


def test_users_with_sources_with_enabled(tmp_path: Any) -> None:
    db = _make_db(tmp_path)
    _insert_source(db, USER)
    result = _users_with_sources(db)
    assert USER in result


def test_users_with_sources_disabled_not_returned(tmp_path: Any) -> None:
    db = _make_db(tmp_path)
    db.connect().execute(
        "INSERT OR IGNORE INTO sources (user_id, kind, config_json, enabled) VALUES (?, 'rss', '{}', 0)",
        (USER,),
    )
    assert _users_with_sources(db) == []


# ---------------------------------------------------------------------------
# _run_ingest_cycle — stubbed asyncio.run
# ---------------------------------------------------------------------------


def test_run_ingest_cycle_no_sources_returns_zero(tmp_path: Any) -> None:
    """When no sources are configured, the ingest cycle writes 0 items."""
    db, ingest, src_reg, _ = _make_stores(tmp_path)
    # No sources in the DB → SourceRegistry.build returns [] → 0 items written
    result = _run_ingest_cycle(ingest, src_reg, USER)
    assert result == 0


def test_run_ingest_cycle_exception_swallowed(tmp_path: Any) -> None:
    """An exception in the ingest cycle is swallowed and 0 is returned."""
    db, ingest, src_reg, _ = _make_stores(tmp_path)
    broken_registry = MagicMock()
    broken_registry.build.side_effect = RuntimeError("network down")
    # Should not raise
    with patch("trading_agent.research.daemon.asyncio.run", side_effect=RuntimeError("boom")):
        result = _run_ingest_cycle(ingest, broken_registry, USER)
    assert result == 0


# ---------------------------------------------------------------------------
# _run_research_pass — stubbed agent
# ---------------------------------------------------------------------------


def test_run_research_pass_no_backlog_returns_zero(tmp_path: Any) -> None:
    """When the ingest backlog is empty the agent returns [] without a model call."""
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    ref = ModelRef(endpoint_id="ep1", model="x")
    # No items in ingest store → no model call → 0 briefs
    result = _run_research_pass(agent, USER, ref)
    assert result == 0


def test_run_research_pass_cost_gate_error_swallowed(tmp_path: Any) -> None:
    """CostGateError is caught and 0 is returned — daemon keeps running."""
    broken_agent = MagicMock()
    broken_agent.run.side_effect = CostGateError("budget exceeded")
    result = _run_research_pass(broken_agent, USER, ModelRef("ep1", "x"))
    assert result == 0


def test_run_research_pass_generic_exception_swallowed(tmp_path: Any) -> None:
    broken_agent = MagicMock()
    broken_agent.run.side_effect = RuntimeError("model error")
    result = _run_research_pass(broken_agent, USER, ModelRef("ep", "m"))
    assert result == 0


# ---------------------------------------------------------------------------
# ResearchDaemon — start/stop + no sources path
# ---------------------------------------------------------------------------


def test_research_daemon_start_stop(tmp_path: Any) -> None:
    """Daemon thread starts, can be stopped cleanly."""
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    daemon = ResearchDaemon(
        db, ingest, src_reg, agent, registry, settings, cadence_seconds=9999
    )
    daemon.start()
    assert daemon._thread is not None
    assert daemon._thread.is_alive()
    daemon.stop()
    # Give the thread a moment to notice the stop signal.
    daemon._stop.wait(timeout=1.0)
    # stop() just sets the event; thread may still be sleeping up to cadence.
    # We only verify the event was set.
    assert daemon._stop.is_set()


def test_research_daemon_start_idempotent(tmp_path: Any) -> None:
    """Calling start() twice does not spawn a second thread."""
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    daemon = ResearchDaemon(
        db, ingest, src_reg, agent, registry, settings, cadence_seconds=9999
    )
    daemon.start()
    t1 = daemon._thread
    daemon.start()
    assert daemon._thread is t1  # same thread object


def test_research_daemon_no_sources_does_not_raise(tmp_path: Any) -> None:
    """With no sources the daemon runs but writes nothing."""
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    calls: list[str] = []

    # Patch the cycle runner to track calls without actually running anything.
    def tracked_run(self: ResearchDaemon, users: list[str]) -> None:
        calls.extend(users)

    with patch.object(ResearchDaemon, "_run_cycle_for_users", tracked_run):
        daemon = ResearchDaemon(
            db, ingest, src_reg, agent, registry, settings, cadence_seconds=9999
        )
        daemon.start()
        # Let the thread start; it should immediately see no users and skip.
        import time
        time.sleep(0.1)
        daemon.stop()

    # No users with sources → _run_cycle_for_users never called.
    assert calls == []


# ---------------------------------------------------------------------------
# Pre-SoD hydration
# ---------------------------------------------------------------------------


def _make_live_window(sod_offset_minutes: float) -> LiveWindow:
    """Build a LiveWindow where SoD is ``sod_offset_minutes`` from now.

    Positive = SoD is in the future; negative = SoD is in the past.
    """
    now = datetime.now(UTC)
    sod = now + timedelta(minutes=sod_offset_minutes)
    open_utc = sod + timedelta(minutes=60)  # SoD = open - 60 min
    close_utc = open_utc + timedelta(hours=6, minutes=30)
    return LiveWindow(
        date_et=now.strftime("%Y-%m-%d"),
        sod_utc=sod,
        open_utc=open_utc,
        close_utc=close_utc,
        eod_utc=close_utc + timedelta(minutes=30),
    )


def _make_daemon_with_scheduler(tmp_path: Any) -> ResearchDaemon:
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    scheduler = MagicMock()
    scheduler._calendar = MagicMock()

    return ResearchDaemon(
        db, ingest, src_reg, agent, registry, settings,
        cadence_seconds=9999, scheduler=scheduler,
    )


def test_pre_sod_fires_within_window(tmp_path: Any) -> None:
    """_check_pre_sod runs a cycle when we are within PRE_SOD_WINDOW_MINUTES of SoD."""
    daemon = _make_daemon_with_scheduler(tmp_path)

    # SoD is 10 minutes from now → inside the 15-min pre-SoD window.
    now = datetime.now(UTC)
    sod = now + timedelta(minutes=10)
    window = LiveWindow(
        date_et=now.strftime("%Y-%m-%d"),
        sod_utc=sod,
        open_utc=sod + timedelta(minutes=60),
        close_utc=sod + timedelta(minutes=60 + 390),
        eod_utc=sod + timedelta(minutes=60 + 390 + 30),
    )

    cycles: list[list[str]] = []

    def track_cycle(self: ResearchDaemon, users: list[str]) -> None:
        cycles.append(list(users))

    with patch("trading_agent.research.daemon.compute_live_window", return_value=window):
        with patch.object(ResearchDaemon, "_run_cycle_for_users", track_cycle):
            daemon._check_pre_sod([USER])

    assert len(cycles) == 1, "Expected exactly one pre-SoD hydration cycle"
    assert USER in cycles[0]


def test_pre_sod_does_not_fire_outside_window(tmp_path: Any) -> None:
    """_check_pre_sod does NOT run when we are more than PRE_SOD_WINDOW_MINUTES away."""
    daemon = _make_daemon_with_scheduler(tmp_path)

    now = datetime.now(UTC)
    # SoD is PRE_SOD_WINDOW_MINUTES + 30 minutes away → outside the hydration window.
    sod = now + timedelta(minutes=PRE_SOD_WINDOW_MINUTES + 30)
    window = LiveWindow(
        date_et=now.strftime("%Y-%m-%d"),
        sod_utc=sod,
        open_utc=sod + timedelta(minutes=60),
        close_utc=sod + timedelta(minutes=60 + 390),
        eod_utc=sod + timedelta(minutes=60 + 390 + 30),
    )

    cycles: list[list[str]] = []

    def track_cycle(self: ResearchDaemon, users: list[str]) -> None:
        cycles.append(list(users))

    with patch("trading_agent.research.daemon.compute_live_window", return_value=window):
        with patch.object(ResearchDaemon, "_run_cycle_for_users", track_cycle):
            daemon._check_pre_sod([USER])

    assert cycles == [], "Pre-SoD hydration must not fire outside the window"





def test_pre_sod_does_not_fire_twice_same_day(tmp_path: Any) -> None:
    """_check_pre_sod runs at most once per trading day."""
    daemon = _make_daemon_with_scheduler(tmp_path)

    now = datetime.now(UTC)
    sod = now + timedelta(minutes=10)  # inside the 15-min window
    window = LiveWindow(
        date_et=now.strftime("%Y-%m-%d"),
        sod_utc=sod,
        open_utc=sod + timedelta(minutes=60),
        close_utc=sod + timedelta(minutes=60 + 390),
        eod_utc=sod + timedelta(minutes=60 + 390 + 30),
    )

    cycles: list[list[str]] = []

    def track_cycle(self: ResearchDaemon, users: list[str]) -> None:
        cycles.append(list(users))

    with patch("trading_agent.research.daemon.compute_live_window", return_value=window):
        with patch.object(ResearchDaemon, "_run_cycle_for_users", track_cycle):
            daemon._check_pre_sod([USER])  # first call — should run
            daemon._check_pre_sod([USER])  # second call — should NOT run

    assert len(cycles) == 1, "Pre-SoD hydration must fire at most once per day"


def test_pre_sod_no_scheduler_does_nothing(tmp_path: Any) -> None:
    """When no scheduler is set, _check_pre_sod is a no-op."""
    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    daemon = ResearchDaemon(
        db, ingest, src_reg, agent, registry, settings, cadence_seconds=9999, scheduler=None
    )
    cycles: list[list[str]] = []

    def track_cycle(self: ResearchDaemon, users: list[str]) -> None:
        cycles.append(list(users))

    with patch.object(ResearchDaemon, "_run_cycle_for_users", track_cycle):
        daemon._check_pre_sod([USER])

    assert cycles == []


def test_pre_sod_non_trading_day_does_nothing(tmp_path: Any) -> None:
    """When compute_live_window returns None (holiday/weekend), no hydration fires."""
    daemon = _make_daemon_with_scheduler(tmp_path)
    cycles: list[list[str]] = []

    def track_cycle(self: ResearchDaemon, users: list[str]) -> None:
        cycles.append(list(users))

    with patch("trading_agent.research.daemon.compute_live_window", return_value=None):
        with patch.object(ResearchDaemon, "_run_cycle_for_users", track_cycle):
            daemon._check_pre_sod([USER])

    assert cycles == []


# ---------------------------------------------------------------------------
# build_research_daemon
# ---------------------------------------------------------------------------


def test_build_research_daemon_missing_state_returns_none(tmp_path: Any) -> None:
    """build_research_daemon returns None when app.state is incomplete."""
    state = MagicMock()
    state.db = None
    result = build_research_daemon(state)
    assert result is None


def test_build_research_daemon_with_full_state(tmp_path: Any) -> None:
    """build_research_daemon returns a ResearchDaemon when state is complete."""
    db = _make_db(tmp_path)
    vec = SqliteVecStore(tmp_path / "memory.db")
    research = ResearchStore(db, vec, FakeEmbedder())
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    state = MagicMock()
    state.db = db
    state.endpoints = registry
    state.settings = settings
    state.research = research

    daemon = build_research_daemon(state)
    assert daemon is not None
    assert isinstance(daemon, ResearchDaemon)


def test_due_users_honors_research_cadence(tmp_path: Any) -> None:
    """ResearchDaemon._due_users gates each user by their research_cadence setting."""
    import time as _time

    db, ingest, src_reg, research = _make_stores(tmp_path)
    agent = _make_agent(db, ingest, research, tmp_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)
    daemon = ResearchDaemon(
        db, ingest, src_reg, agent, registry, settings, cadence_seconds=3600
    )

    # "off" → never due
    settings.set(USER, "research_cadence", "off")
    assert daemon._due_users([USER]) == []

    # "15m" → due on first evaluation (last_run defaults to 0)
    settings.set(USER, "research_cadence", "15m")
    assert daemon._due_users([USER]) == [USER]

    # after a run, not due again until the 15-min interval elapses
    daemon._last_run[USER] = _time.time()
    assert daemon._due_users([USER]) == []

    # pretend 16 minutes passed → due again
    daemon._last_run[USER] = _time.time() - 16 * 60
    assert daemon._due_users([USER]) == [USER]
