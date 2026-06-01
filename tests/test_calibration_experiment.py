"""Tests for the P6 calibration experiment driver (ON/OFF cohort A/B proof).

All tests use mocked traders — no LLM calls, no OpenRouter key required.

Coverage:
- ExperimentStore CRUD (create / record / read).
- ExperimentDriver cohort split: intel_on traders carry memory, intel_off do not.
- ExperimentDriver metric recording: metrics written per cohort, values sensible.
- End-to-end via run(): store records a 'done' ExperimentRun with two cohorts.
- HTTP endpoints: GET results, GET single run, POST run (injected mock).
- Edge cases: missing experiment_store → 503, missing bench_controller → 503.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path: Any) -> Database:
    return Database(str(tmp_path / "test.db"))


@pytest.fixture()
def store(db: Database) -> Any:
    from trading_agent.calibration.experiment import ExperimentStore
    return ExperimentStore(db)


def _make_bench_and_controller(symbols: list[str] | None = None) -> tuple[Any, Any]:
    """Return (Bench, BenchController) with a MagicMock OpenRouter client."""
    from trading_agent.bench.bench import Bench
    from trading_agent.bench.controller import BenchController

    syms = symbols or ["AAPL"]
    bench = Bench(syms)
    client = MagicMock()
    controller = BenchController(bench, client, symbols=syms, memory=MagicMock())
    return bench, controller


def _mock_trader(model: str, name: str, intel_on: bool) -> Any:
    """Produce a mock Trader that always returns a HOLD decision."""
    from trading_agent.llm.trader import DecisionResult

    trader = MagicMock()
    trader.name = name
    # memory is set/not-set to reflect intel_on flag in factory-created traders.
    trader.memory = MagicMock() if intel_on else None
    trader.decide.return_value = DecisionResult(decisions=[], comment="hold", error=None)
    trader.observe = MagicMock()
    return trader


# ── ExperimentStore tests ─────────────────────────────────────────────────────


class TestExperimentStore:
    def test_create_and_get_run(self, store: Any) -> None:
        run_id = store.create_run(model="test/model")
        assert isinstance(run_id, str) and len(run_id) == 32

        run = store.get_run(run_id)
        assert run is not None
        assert run.run_id == run_id
        assert run.model == "test/model"
        assert run.status == "pending"
        assert run.cohorts == []

    def test_set_status(self, store: Any) -> None:
        run_id = store.create_run(model="m")
        store.set_status(run_id, "running")
        run = store.get_run(run_id)
        assert run is not None
        assert run.status == "running"

    def test_finish_run(self, store: Any) -> None:
        run_id = store.create_run(model="m")
        store.finish_run(run_id, rounds=5)
        run = store.get_run(run_id)
        assert run is not None
        assert run.status == "done"
        assert run.rounds == 5
        assert run.finished_at is not None

    def test_fail_run(self, store: Any) -> None:
        run_id = store.create_run(model="m")
        store.fail_run(run_id)
        run = store.get_run(run_id)
        assert run is not None
        assert run.status == "failed"

    def test_record_cohort_and_metrics(self, store: Any) -> None:
        run_id = store.create_run(model="m")
        store.record_cohort(
            run_id, "intel_on", intel_on=True, trader_names=["t1", "t2"]
        )
        store.record_metric(run_id, "intel_on", "mean_pnl", 42.0)
        store.record_metric(run_id, "intel_on", "mean_return_pct", 0.04)

        run = store.get_run(run_id)
        assert run is not None
        assert len(run.cohorts) == 1
        c = run.cohorts[0]
        assert c.label == "intel_on"
        assert c.intel_on is True
        assert c.trader_names == ["t1", "t2"]
        assert c.mean_pnl == pytest.approx(42.0)
        assert c.mean_return_pct == pytest.approx(0.04)

    def test_list_runs(self, store: Any) -> None:
        for i in range(3):
            store.create_run(model=f"m{i}")
        runs = store.list_runs(limit=10)
        assert len(runs) == 3

    def test_list_runs_limit(self, store: Any) -> None:
        for _i in range(5):
            store.create_run(model="m")
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_get_missing_run(self, store: Any) -> None:
        assert store.get_run("nonexistent") is None

    def test_run_as_dict(self, store: Any) -> None:
        run_id = store.create_run(model="m")
        run = store.get_run(run_id)
        assert run is not None
        d = run.as_dict()
        assert d["run_id"] == run_id
        assert d["status"] == "pending"
        assert isinstance(d["cohorts"], list)


# ── ExperimentDriver tests ────────────────────────────────────────────────────


class TestExperimentDriver:
    def test_cohort_split_memory_flag(self, db: Any) -> None:
        """intel_on traders get memory; intel_off traders do not — via add_model."""
        from trading_agent.calibration.experiment import ExperimentDriver, ExperimentStore

        bench, controller = _make_bench_and_controller()
        store = ExperimentStore(db)

        driver = ExperimentDriver(
            controller,
            store,
            model="test/model",
            rounds=1,
            cohort_size=1,
            trader_factory=_mock_trader,
        )
        run = driver.run()
        assert run.status == "done"

        # There should be 2 traders: one on, one off.
        on_names = [n for n in bench.names() if "intel_on" in n]
        off_names = [n for n in bench.names() if "intel_off" in n]
        assert len(on_names) == 1
        assert len(off_names) == 1

        # Check memory flag through the mock trader.
        on_comp = bench._competitors[on_names[0]]
        off_comp = bench._competitors[off_names[0]]
        assert on_comp.trader.memory is not None
        assert off_comp.trader.memory is None

    def test_two_cohorts_recorded(self, store: Any) -> None:
        """Two cohorts (intel_on + intel_off) are recorded in the store."""
        from trading_agent.calibration.experiment import ExperimentDriver

        bench, controller = _make_bench_and_controller()
        driver = ExperimentDriver(
            controller,
            store,
            model="test/model",
            rounds=2,
            cohort_size=1,
            trader_factory=_mock_trader,
        )
        run = driver.run()
        assert run.status == "done"
        assert len(run.cohorts) == 2
        labels = {c.label for c in run.cohorts}
        assert labels == {"intel_on", "intel_off"}

    def test_metrics_recorded_for_both_cohorts(self, store: Any) -> None:
        """Each cohort has metrics persisted (even if all zeros from HOLD)."""
        from trading_agent.calibration.experiment import ExperimentDriver

        bench, controller = _make_bench_and_controller()
        driver = ExperimentDriver(
            controller,
            store,
            model="test/model",
            rounds=1,
            cohort_size=1,
            trader_factory=_mock_trader,
        )
        run = driver.run()

        for cohort in run.cohorts:
            # mean_pnl must be present (0.0 from a no-trade HOLD run).
            assert cohort.mean_pnl is not None

    def test_rounds_count_stored(self, store: Any) -> None:
        """The requested round count is stored on the ExperimentRun."""
        from trading_agent.calibration.experiment import ExperimentDriver

        _, controller = _make_bench_and_controller()
        driver = ExperimentDriver(
            controller,
            store,
            rounds=3,
            trader_factory=_mock_trader,
        )
        run = driver.run()
        assert run.rounds == 3

    def test_bars_fed_to_bench(self, store: Any) -> None:
        """Bars passed to the driver are observed by the bench before each round."""
        from trading_agent.calibration.experiment import ExperimentDriver

        bench, controller = _make_bench_and_controller(["AAPL"])
        bars = [{"symbol": "AAPL", "close": 150.0 + i} for i in range(3)]

        driver = ExperimentDriver(
            controller,
            store,
            rounds=3,
            bars=bars,
            trader_factory=_mock_trader,
        )
        driver.run()
        # After the last bar, the bench should have the final price.
        assert bench._last_prices.get("AAPL") == pytest.approx(152.0)

    def test_cohort_size_two(self, store: Any) -> None:
        """cohort_size=2 registers 4 traders total (2 per cohort)."""
        from trading_agent.calibration.experiment import ExperimentDriver

        bench, controller = _make_bench_and_controller()
        driver = ExperimentDriver(
            controller,
            store,
            rounds=1,
            cohort_size=2,
            trader_factory=_mock_trader,
        )
        run = driver.run()
        assert run.status == "done"
        assert sum(len(c.trader_names) for c in run.cohorts) == 4

    def test_run_returns_done_status(self, store: Any) -> None:
        from trading_agent.calibration.experiment import ExperimentDriver

        _, controller = _make_bench_and_controller()
        driver = ExperimentDriver(
            controller,
            store,
            rounds=1,
            trader_factory=_mock_trader,
        )
        run = driver.run()
        assert run.status == "done"
        assert run.finished_at is not None

    def test_intel_on_flag_via_controller(self, db: Any) -> None:
        """Without trader_factory, add_model is used: intel_off traders get None memory.

        This test exercises the real add_model() path but does NOT call run() since
        that would invoke AgentTrader.decide() over the mock LLM client, which is
        complex to configure correctly.  We verify only the memory-flag wiring (the
        part that's specific to this test) and leave the end-to-end execution proof
        to the other tests that use trader_factory.
        """
        from trading_agent.bench.bench import Bench
        from trading_agent.bench.controller import BenchController

        syms = ["NVDA"]
        bench = Bench(syms)
        client = MagicMock()
        memory = MagicMock()
        controller = BenchController(bench, client, symbols=syms, memory=memory)

        # Directly use add_model() (the same call ExperimentDriver makes) to check
        # that intelligence_flags are wired correctly — without running any rounds.
        on_name = controller.add_model(
            "test/model",
            "probe-intel_on-0",
            intelligence_flags={},
            tutorial_remaining=0,
        )
        off_name = controller.add_model(
            "test/model",
            "probe-intel_off-0",
            intelligence_flags={"memory": False},
            tutorial_remaining=0,
        )

        on_comp = bench._competitors[on_name]
        off_comp = bench._competitors[off_name]
        assert on_comp.trader.memory is memory      # controller memory threaded in
        assert off_comp.trader.memory is None       # memory=False flag applied


# ── HTTP endpoint tests ───────────────────────────────────────────────────────


def _authed_client(app: Any, username: str = "u1", password: str = "pw") -> Any:
    """Return a TestClient logged in with a session cookie."""
    client = TestClient(app, raise_server_exceptions=False)
    client.post("/api/auth/signup", json={"username": username, "password": password})
    client.post("/api/auth/login", json={"username": username, "password": password})
    return client


class TestExperimentEndpoints:
    def _app_with_store(self, tmp_path: Any, *, with_controller: bool = False) -> tuple[Any, Any]:
        from trading_agent.calibration.experiment import ExperimentStore
        from trading_agent.web.app import create_cockpit_app

        db = Database(str(tmp_path / "web.db"))
        app = create_cockpit_app(db=db)
        store = ExperimentStore(db)
        app.state.experiment_store = store

        if with_controller:
            bench, controller = _make_bench_and_controller()
            app.state.bench_controller = controller
            # Expose bench for the calibration router (optional).
            app.state.bench = bench
        return app, store

    def test_list_results_empty(self, tmp_path: Any) -> None:
        app, _ = self._app_with_store(tmp_path)
        client = _authed_client(app)
        resp = client.get("/api/calibration/experiment/results")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_missing_run_404(self, tmp_path: Any) -> None:
        app, _ = self._app_with_store(tmp_path)
        client = _authed_client(app)
        resp = client.get("/api/calibration/experiment/does-not-exist")
        assert resp.status_code == 404

    def test_missing_experiment_store_503(self, tmp_path: Any) -> None:
        from trading_agent.web.app import create_cockpit_app

        db = Database(str(tmp_path / "no_store.db"))
        app = create_cockpit_app(db=db)
        client = _authed_client(app)
        resp = client.get("/api/calibration/experiment/results")
        assert resp.status_code == 503

    def test_missing_bench_controller_503(self, tmp_path: Any) -> None:
        app, _ = self._app_with_store(tmp_path)
        client = _authed_client(app)
        resp = client.post(
            "/api/calibration/experiment/run",
            json={"model": "test/m", "rounds": 1},
        )
        assert resp.status_code == 503

    def test_post_run_and_list(self, tmp_path: Any) -> None:
        """POST run with mock-trader-patched controller; then list it."""
        from unittest.mock import patch

        from trading_agent.calibration import ExperimentDriver

        app, store = self._app_with_store(tmp_path, with_controller=True)
        client = _authed_client(app)

        # Patch ExperimentDriver to inject mock traders.
        original_init = ExperimentDriver.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("trader_factory", _mock_trader)
            original_init(self, *args, **kwargs)

        with patch.object(ExperimentDriver, "__init__", patched_init):
            resp = client.post(
                "/api/calibration/experiment/run",
                json={"model": "test/model", "rounds": 2, "cohort_size": 1},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["rounds"] == 2
        assert len(data["cohorts"]) == 2

        run_id = data["run_id"]
        # The run should show up in the list.
        resp2 = client.get("/api/calibration/experiment/results")
        assert resp2.status_code == 200
        ids = [r["run_id"] for r in resp2.json()]
        assert run_id in ids

        # And be fetchable by ID.
        resp3 = client.get(f"/api/calibration/experiment/{run_id}")
        assert resp3.status_code == 200
        assert resp3.json()["run_id"] == run_id

    def test_get_run_has_cohorts(self, tmp_path: Any) -> None:
        """Fetched run includes both cohort metrics."""
        from trading_agent.calibration import ExperimentStore

        db = Database(str(tmp_path / "e2.db"))
        store = ExperimentStore(db)
        run_id = store.create_run(model="m")
        store.record_cohort(run_id, "intel_on", intel_on=True, trader_names=["t1"])
        store.record_cohort(run_id, "intel_off", intel_on=False, trader_names=["t2"])
        store.record_metric(run_id, "intel_on", "mean_pnl", 10.0)
        store.record_metric(run_id, "intel_off", "mean_pnl", -2.0)
        store.finish_run(run_id, rounds=3)

        from trading_agent.web.app import create_cockpit_app

        app = create_cockpit_app(db=db)
        app.state.experiment_store = store
        client = _authed_client(app)

        resp = client.get(f"/api/calibration/experiment/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        labels = {c["label"] for c in body["cohorts"]}
        assert labels == {"intel_on", "intel_off"}
        pnls = {c["label"]: c["mean_pnl"] for c in body["cohorts"]}
        assert pnls["intel_on"] == pytest.approx(10.0)
        assert pnls["intel_off"] == pytest.approx(-2.0)
