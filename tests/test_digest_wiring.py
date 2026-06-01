"""WS-Digest wiring tests (wave5/digest-wiring).

Covers:
  - build_digest_daemon factory: returns a DigestDaemon from a populated
    app.state; returns None when prerequisites are absent.
  - build_cockpit: attaches app.state.digest_store as a DigestStore and passes
    it to BenchController (controller._digest_store is non-None).
  - POST /api/accounts: digest_mode=true => trader's _digest_mode is True and
    digest_store is wired; false/absent => pull-mode unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
from fastapi.testclient import TestClient

from trading_agent.bench.bench import Bench
from trading_agent.bench.controller import BenchController
from trading_agent.config.db import Database
from trading_agent.digest.daemon import DigestDaemon, build_digest_daemon
from trading_agent.digest.store import DigestStore
from trading_agent.llm.openrouter import OpenRouterClient
from trading_agent.scripts.serve import build_cockpit
from trading_agent.web.app import create_cockpit_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "x", "choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    return httpx.MockTransport(handler)


def _make_app_state(tmp_path: Path) -> Any:
    """Minimal app.state-like namespace with all digest prerequisites."""
    from trading_agent.config.endpoints import EndpointRegistry
    from trading_agent.config.settings_store import SettingsStore

    db = Database(tmp_path / "cfg.db")
    ds = DigestStore(db)

    state = MagicMock()
    state.db = db
    state.endpoints = EndpointRegistry(db)
    state.settings = SettingsStore(db)
    state.digest_store = ds
    state.research = None  # optional; compiler accepts None
    return state


# ---------------------------------------------------------------------------
# 1. build_digest_daemon factory
# ---------------------------------------------------------------------------


class TestBuildDigestDaemon:
    def test_returns_daemon_with_full_state(self, tmp_path: Path) -> None:
        """Factory returns a DigestDaemon when all prerequisites are present."""
        state = _make_app_state(tmp_path)
        daemon = build_digest_daemon(state)
        assert daemon is not None
        assert isinstance(daemon, DigestDaemon)

    def test_returns_none_when_digest_store_absent(self, tmp_path: Path) -> None:
        """Factory returns None when app.state.digest_store is not set."""
        state = _make_app_state(tmp_path)
        state.digest_store = None
        result = build_digest_daemon(state)
        assert result is None

    def test_returns_none_when_db_absent(self, tmp_path: Path) -> None:
        """Factory returns None when db is absent."""
        state = _make_app_state(tmp_path)
        state.db = None
        result = build_digest_daemon(state)
        assert result is None

    def test_returns_none_when_endpoints_absent(self, tmp_path: Path) -> None:
        """Factory returns None when endpoint registry is absent."""
        state = _make_app_state(tmp_path)
        state.endpoints = None
        result = build_digest_daemon(state)
        assert result is None

    def test_returns_none_when_settings_absent(self, tmp_path: Path) -> None:
        """Factory returns None when settings store is absent."""
        state = _make_app_state(tmp_path)
        state.settings = None
        result = build_digest_daemon(state)
        assert result is None

    def test_daemon_uses_custom_cadence(self, tmp_path: Path) -> None:
        """Factory passes cadence_seconds through to the daemon."""
        state = _make_app_state(tmp_path)
        daemon = build_digest_daemon(state, cadence_seconds=60)
        assert daemon is not None
        assert daemon.cadence_seconds == 60

    def test_daemon_can_set_bombshell_callback(self, tmp_path: Path) -> None:
        """DigestDaemon.set_bombshell_callback wires a callback without raising."""
        state = _make_app_state(tmp_path)
        daemon = build_digest_daemon(state)
        assert daemon is not None
        cb = MagicMock()
        daemon.set_bombshell_callback(cb)  # should not raise


# ---------------------------------------------------------------------------
# 2. build_cockpit attaches digest_store and wires BenchController
# ---------------------------------------------------------------------------


class TestBuildCockpitDigestWiring:
    def test_attaches_digest_store(self, tmp_path: Path) -> None:
        """build_cockpit sets app.state.digest_store to a DigestStore."""
        app = build_cockpit(
            db=Database(tmp_path / "c.db"),
            data_dir=tmp_path,
            transport=_mock_transport(),
        )
        try:
            assert hasattr(app.state, "digest_store")
            assert isinstance(app.state.digest_store, DigestStore)
        finally:
            app.state.approvals.close()

    def test_controller_has_digest_store(self, tmp_path: Path) -> None:
        """When a controller is built, it receives the shared digest_store."""
        app = build_cockpit(
            db=Database(tmp_path / "c.db"),
            data_dir=tmp_path,
            transport=_mock_transport(),
            openrouter_client=OpenRouterClient(api_key="k", transport=_mock_transport()),
        )
        try:
            ctrl = app.state.bench_controller
            assert isinstance(ctrl, BenchController)
            # The controller's internal _digest_store should be the same object
            # that was attached to app.state.
            assert ctrl._digest_store is app.state.digest_store
            assert isinstance(ctrl._digest_store, DigestStore)
        finally:
            app.state.approvals.close()

    def test_vstore_and_embedder_on_state(self, tmp_path: Path) -> None:
        """build_cockpit exposes vstore and embedder on app.state (used by DigestStore)."""
        app = build_cockpit(
            db=Database(tmp_path / "c.db"),
            data_dir=tmp_path,
            transport=_mock_transport(),
        )
        try:
            # vstore is always set (even without an owner)
            assert hasattr(app.state, "vstore")
            assert app.state.vstore is not None
            # embedder may be None when no owner resolves (no signed-up user)
            assert hasattr(app.state, "embedder")
        finally:
            app.state.approvals.close()


# ---------------------------------------------------------------------------
# 3. POST /api/accounts: digest_mode routing
# ---------------------------------------------------------------------------


def _cockpit_client_with_controller(tmp_path: Path) -> TestClient:
    """Authed cockpit with a live BenchController that has a digest_store."""
    db = Database(tmp_path / "c.db")
    ds = DigestStore(db)
    bench = Bench(["AAPL"], initial_balance=50_000.0)
    oc = OpenRouterClient(api_key="k", transport=_mock_transport())
    ctrl = BenchController(bench, oc, symbols=["AAPL"], digest_store=ds)

    app = create_cockpit_app(db, transport=_mock_transport())
    app.state.bench = bench
    app.state.bench_controller = ctrl

    from trading_agent.approval_queue import ApprovalQueue
    from trading_agent.risk_manager import RiskManager

    app.state.risk = RiskManager(kill_switch_file=tmp_path / ".kill")
    app.state.approvals = ApprovalQueue(
        db_path=tmp_path / "approvals.db", executor=lambda sig: {"filled": sig}
    )

    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "user1", "password": "pw"})
    return c


class TestCreateTraderDigestMode:
    def test_digest_mode_true_creates_digest_trader(self, tmp_path: Path) -> None:
        """POST /api/accounts with digest_mode=true => trader._digest_mode is True."""
        client = _cockpit_client_with_controller(tmp_path)
        r = client.post(
            "/api/accounts",
            json={"model": "z-ai/glm-5.1", "name": "digest-trader", "digest_mode": True},
        )
        assert r.status_code == 200
        assert r.json()["created"] == "digest-trader"

        bench = client.app.state.bench
        comp = bench._competitors.get("digest-trader")
        assert comp is not None, "trader was not registered in the bench"
        assert getattr(comp.trader, "_digest_mode", False) is True, (
            "trader._digest_mode should be True when digest_mode=true"
        )

    def test_digest_mode_false_is_pull_mode(self, tmp_path: Path) -> None:
        """POST /api/accounts with digest_mode=false => pull mode (digest off)."""
        client = _cockpit_client_with_controller(tmp_path)
        r = client.post(
            "/api/accounts",
            json={"model": "z-ai/glm-5.1", "name": "pull-trader", "digest_mode": False},
        )
        assert r.status_code == 200
        bench = client.app.state.bench
        comp = bench._competitors.get("pull-trader")
        assert comp is not None
        assert getattr(comp.trader, "_digest_mode", False) is False, (
            "trader._digest_mode should be False when digest_mode=false"
        )

    def test_digest_mode_absent_is_pull_mode(self, tmp_path: Path) -> None:
        """POST /api/accounts without digest_mode field => pull mode (default)."""
        client = _cockpit_client_with_controller(tmp_path)
        r = client.post(
            "/api/accounts",
            json={"model": "z-ai/glm-5.1", "name": "default-trader"},
        )
        assert r.status_code == 200
        bench = client.app.state.bench
        comp = bench._competitors.get("default-trader")
        assert comp is not None
        assert getattr(comp.trader, "_digest_mode", False) is False, (
            "trader._digest_mode should be False (default) when digest_mode absent"
        )

    def test_digest_mode_true_wires_digest_store(self, tmp_path: Path) -> None:
        """When digest_mode=true and controller has a digest_store, the trader
        gets a non-None digest_store (so search_context can function)."""
        client = _cockpit_client_with_controller(tmp_path)
        r = client.post(
            "/api/accounts",
            json={"model": "z-ai/glm-5.1", "name": "wired-digest", "digest_mode": True},
        )
        assert r.status_code == 200
        bench = client.app.state.bench
        comp = bench._competitors.get("wired-digest")
        assert comp is not None
        trader = comp.trader
        assert getattr(trader, "_digest_mode", False) is True
        # The trader's _digest_store should be set (not None) because the
        # controller holds a digest_store and digest_mode is True.
        assert getattr(trader, "_digest_store", None) is not None, (
            "digest_store should be wired into the trader when digest_mode=True"
        )

    def test_pull_mode_no_digest_store_on_trader(self, tmp_path: Path) -> None:
        """A pull-mode trader must NOT have a digest_store wired (mode isolation)."""
        client = _cockpit_client_with_controller(tmp_path)
        client.post(
            "/api/accounts",
            json={"model": "z-ai/glm-5.1", "name": "pull-iso"},
        )
        bench = client.app.state.bench
        comp = bench._competitors.get("pull-iso")
        assert comp is not None
        trader = comp.trader
        assert getattr(trader, "_digest_mode", False) is False
        # _digest_store is None because effective_digest_store is gated on digest_mode
        assert getattr(trader, "_digest_store", None) is None, (
            "Pull-mode trader should have _digest_store=None"
        )
