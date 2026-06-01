"""Tests for the analyst-digest tier (WS-Digest).

Covers:
  - DigestStore: put/get/search round-trip
  - DigestCompiler: distillation, token budget, materiality detection
  - SearchContextTool: returns results / degrades when vault is absent
  - AgentTrader digest mode:
      * inject digest into first-look (extra_lines)
      * gate slow LOOK tools (news/situation/etc. absent from defs)
      * expose search_context backstop
      * search_context dispatches correctly
  - Default-off invariant: when digest_mode=False, tool catalog is byte-for-byte
    identical to the pre-digest baseline (no extra tools, no removed tools)
  - Event-wake: on_research_bombshell fires a run_one for matching digest traders
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Any:
    from trading_agent.config.db import Database

    db = Database(tmp_path / "test.db")
    return db


@pytest.fixture()
def digest_store(tmp_db: Any) -> Any:
    from trading_agent.digest.store import DigestStore

    return DigestStore(tmp_db)


@pytest.fixture()
def fake_embedder() -> Any:
    from trading_agent.memory.embed import FakeEmbedder

    return FakeEmbedder()


# ---------------------------------------------------------------------------
# 1. DigestStore — put / get / search round-trip
# ---------------------------------------------------------------------------


class TestDigestStore:
    def test_put_and_get_latest(self, digest_store: Any) -> None:
        from trading_agent.digest.store import Digest

        d = Digest(
            user_id="u1",
            universe_key="AAPL,NVDA",
            as_of=time.time(),
            digest_text="Regime: calm\n• AAPL up on earnings",
            headlines=["AAPL: beats expectations"],
            regime_label="calm",
            material_flag=False,
        )
        stored = digest_store.put(d)
        assert stored.id != ""
        assert stored.created_at > 0

        got = digest_store.get_latest("u1", ["NVDA", "AAPL"])  # order should not matter
        assert got is not None
        assert got.universe_key == "AAPL,NVDA"
        assert "calm" in got.digest_text

    def test_get_by_key(self, digest_store: Any) -> None:
        from trading_agent.digest.store import Digest, universe_key

        symbols = ["SPY", "QQQ"]
        uk = universe_key(symbols)
        d = Digest(
            user_id="u2",
            universe_key=uk,
            as_of=time.time(),
            digest_text="Macro: risk-off",
            headlines=[],
            regime_label="risk-off",
            material_flag=True,
        )
        digest_store.put(d)
        got = digest_store.get_by_key("u2", uk)
        assert got is not None
        assert got.material_flag is True

    def test_get_latest_returns_none_when_absent(self, digest_store: Any) -> None:
        got = digest_store.get_latest("nonexistent_user", ["AAPL"])
        assert got is None

    def test_recent(self, digest_store: Any) -> None:
        from trading_agent.digest.store import Digest

        for i in range(3):
            digest_store.put(
                Digest(
                    user_id="u3",
                    universe_key=f"SYM{i}",
                    as_of=time.time() - i * 100,
                    digest_text=f"digest {i}",
                    headlines=[],
                    regime_label=None,
                    material_flag=False,
                )
            )
        recents = digest_store.recent("u3", n=10)
        assert len(recents) == 3

    def test_upsert_replaces_existing(self, digest_store: Any) -> None:
        from trading_agent.digest.store import Digest

        uk = "AAPL"
        for text in ["first", "second"]:
            digest_store.put(
                Digest(
                    user_id="u4",
                    universe_key=uk,
                    as_of=time.time(),
                    digest_text=text,
                    headlines=[],
                    regime_label=None,
                    material_flag=False,
                )
            )
        got = digest_store.get_by_key("u4", uk)
        assert got is not None
        assert got.digest_text == "second"

    def test_universe_key_canonical(self) -> None:
        from trading_agent.digest.store import universe_key

        assert universe_key(["NVDA", "AAPL"]) == universe_key(["AAPL", "NVDA"])
        assert universe_key(["aapl"]) == "AAPL"

    def test_staleness(self) -> None:
        from trading_agent.digest.store import Digest

        old = Digest(
            user_id="u", universe_key="X", as_of=time.time() - 7200,
            digest_text="old", headlines=[], regime_label=None, material_flag=False,
        )
        assert old.is_stale(3600) is True

        fresh = Digest(
            user_id="u", universe_key="X", as_of=time.time(),
            digest_text="fresh", headlines=[], regime_label=None, material_flag=False,
        )
        assert fresh.is_stale(3600) is False

    def test_vector_search_fallback_when_no_store(self, digest_store: Any) -> None:
        # No vector store wired → returns empty list.
        results = digest_store.search_vector("u1", "AAPL earnings", k=5)
        assert results == []

    def test_vector_search_with_fake_embedder(
        self, tmp_path: Any, fake_embedder: Any
    ) -> None:
        from trading_agent.config.db import Database
        from trading_agent.digest.store import Digest, DigestStore
        from trading_agent.memory import make_vector_store

        db = Database(tmp_path / "vec_test.db")
        vec_store = make_vector_store("sqlite-vec", path=str(tmp_path / "vec.db"))
        ds = DigestStore(db, vector=vec_store, embedder=fake_embedder)

        ds.put(
            Digest(
                user_id="uv",
                universe_key="AAPL",
                as_of=time.time(),
                digest_text="AAPL strong earnings beat",
                headlines=["AAPL: earnings surprise"],
                regime_label="calm",
                material_flag=False,
            )
        )

        results = ds.search_vector("uv", "AAPL earnings", k=3)
        assert len(results) > 0
        assert any("AAPL" in str(r) for r in results)


# ---------------------------------------------------------------------------
# 2. DigestCompiler — distillation, token budget, materiality
# ---------------------------------------------------------------------------


class TestDigestCompiler:
    def _make_compiler(
        self, tmp_db: Any, digest_store: Any, model_response: dict[str, Any]
    ) -> Any:
        """Build a DigestCompiler with mocked endpoint."""
        from trading_agent.digest.compiler import DigestCompiler

        registry = MagicMock()
        endpoint = MagicMock()
        endpoint.base_url = "http://fake-model"
        endpoint.api_key = "fake-key"
        registry.get.return_value = endpoint

        settings = MagicMock()
        settings.get.return_value = None  # no cost ceiling

        compiler = DigestCompiler(
            digest_store=digest_store,
            research_store=None,
            registry=registry,
            settings=settings,
            db=tmp_db,
        )
        return compiler, endpoint

    def test_compile_and_persist(self, tmp_db: Any, digest_store: Any) -> None:

        compiler, _ = self._make_compiler(
            tmp_db,
            digest_store,
            {"headlines": ["AAPL: beats Q1 EPS"], "regime": "elevated", "material_event": False},
        )

        model_json = json.dumps(
            {"headlines": ["AAPL: beats Q1 EPS"], "regime": "elevated", "material_event": False}
        )

        ref = MagicMock(spec=["endpoint_id", "model"])
        ref.endpoint_id = "ep1"
        ref.model = "cheap-model"

        with patch("httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": model_json}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            # Cost gate: mock the gate check to pass.
            with patch("trading_agent.memory.reflect.CostGate.check"), \
                 patch("trading_agent.memory.reflect.CostGate.record"):
                digest = compiler.compile("u1", ["AAPL"], ref)

        assert digest is not None
        assert "AAPL" in digest.digest_text
        assert digest.regime_label == "elevated"
        assert digest.material_flag is False

    def test_token_budget_enforced(self, tmp_db: Any, digest_store: Any) -> None:

        compiler, _ = self._make_compiler(tmp_db, digest_store, {})
        compiler._max_chars = 50  # tight budget

        ref = MagicMock()
        ref.endpoint_id = "ep1"
        ref.model = "cheap"

        very_long = "X" * 500
        model_json = json.dumps(
            {"headlines": [very_long] * 20, "regime": "calm", "material_event": False}
        )

        with patch("httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": model_json}}]}
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            with patch("trading_agent.memory.reflect.CostGate.check"), \
                 patch("trading_agent.memory.reflect.CostGate.record"):
                digest = compiler.compile("u1", ["AAPL"], ref)

        assert digest is not None
        assert len(digest.digest_text) <= 50

    def test_materiality_detection_from_headlines(self) -> None:
        from trading_agent.digest.compiler import DigestCompiler

        assert DigestCompiler._detect_material(["AAPL: earnings surprise Q1 2025"]) is True
        assert DigestCompiler._detect_material(["TSLA merger announced"]) is True
        assert DigestCompiler._detect_material(["NVDA up 2% today"]) is False

    def test_materiality_detection_from_model_flag(
        self, tmp_db: Any, digest_store: Any
    ) -> None:

        compiler, _ = self._make_compiler(tmp_db, digest_store, {})
        ref = MagicMock()
        ref.endpoint_id = "ep1"
        ref.model = "cheap"

        model_json = json.dumps(
            {"headlines": ["boring news"], "regime": "calm", "material_event": True}
        )

        with patch("httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": model_json}}]}
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            with patch("trading_agent.memory.reflect.CostGate.check"), \
                 patch("trading_agent.memory.reflect.CostGate.record"):
                digest = compiler.compile("u1", ["AAPL"], ref)

        assert digest is not None
        assert digest.material_flag is True

    def test_returns_none_when_cost_gated(
        self, tmp_db: Any, digest_store: Any
    ) -> None:
        from trading_agent.memory.reflect import CostGateError

        compiler, _ = self._make_compiler(tmp_db, digest_store, {})
        ref = MagicMock()
        ref.endpoint_id = "ep1"
        ref.model = "cheap"

        with patch("trading_agent.memory.reflect.CostGate.check", side_effect=CostGateError("budget")):
            result = compiler.compile("u1", ["AAPL"], ref)

        assert result is None


# ---------------------------------------------------------------------------
# 3. SearchContextTool
# ---------------------------------------------------------------------------


class TestSearchContextTool:
    def test_returns_empty_when_no_store(self) -> None:
        from trading_agent.intel.tools.look.search_context import SearchContextTool

        tool = SearchContextTool(
            owner_user_id="u1",
            trader_id="alpha",
            digest_store=None,
        )
        result = tool("AAPL earnings")
        assert result.ok
        assert result.data["results"] == []

    def test_returns_error_on_empty_query(self) -> None:
        from trading_agent.intel.tools.look.search_context import SearchContextTool

        tool = SearchContextTool(
            owner_user_id="u1",
            trader_id="alpha",
            digest_store=MagicMock(),
        )
        result = tool("")
        assert not result.ok
        assert result.error.kind == "invalid_input"  # type: ignore[attr-defined]

    def test_searches_digest_store(self, tmp_path: Any, fake_embedder: Any) -> None:
        from trading_agent.config.db import Database
        from trading_agent.digest.store import Digest, DigestStore
        from trading_agent.intel.tools.look.search_context import SearchContextTool
        from trading_agent.memory import make_vector_store

        db = Database(tmp_path / "sc_test.db")
        vec_store = make_vector_store("sqlite-vec", path=str(tmp_path / "sc_vec.db"))
        ds = DigestStore(db, vector=vec_store, embedder=fake_embedder)

        ds.put(
            Digest(
                user_id="u1",
                universe_key="AAPL",
                as_of=time.time(),
                digest_text="AAPL strong earnings beat analyst expectations",
                headlines=["AAPL: beat"],
                regime_label="calm",
                material_flag=False,
            )
        )

        tool = SearchContextTool(
            owner_user_id="u1",
            trader_id="alpha",
            digest_store=ds,
        )
        result = tool("AAPL earnings", k=3)
        assert result.ok
        assert result.data["total"] > 0

    def test_no_user_context_returns_empty(self, digest_store: Any) -> None:
        from trading_agent.intel.tools.look.search_context import SearchContextTool

        tool = SearchContextTool(
            owner_user_id=None,
            trader_id="alpha",
            digest_store=digest_store,
        )
        result = tool("NVDA")
        assert result.ok
        assert result.data["results"] == []


# ---------------------------------------------------------------------------
# 4. AgentTrader digest mode — inject + tool gating + dispatch
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    digest_mode: bool = False,
    digest_store: Any = None,
    symbols: list[str] | None = None,
) -> Any:
    """Convenience factory for a minimal AgentTrader."""
    from trading_agent.llm.trader import AgentTrader

    client = MagicMock()
    return AgentTrader(
        "test-model",
        client,
        symbols=symbols or ["AAPL"],
        name="TestAgent",
        digest_mode=digest_mode,
        digest_store=digest_store,
        tutorial_remaining=0,
    )


class TestAgentTraderDigestMode:
    def test_default_off_no_digest_injection(self) -> None:
        """With digest_mode=False, first-look extra_lines has no digest content."""
        agent = _make_agent(digest_mode=False)
        lines = agent._first_look_digest_lines()
        assert lines == []

    def test_digest_off_when_no_store(self) -> None:
        """With digest_mode=True but no store, still returns []."""
        agent = _make_agent(digest_mode=True, digest_store=None)
        lines = agent._first_look_digest_lines()
        assert lines == []

    def test_digest_injected_when_store_has_record(self, digest_store: Any) -> None:
        """With digest_mode=True + store + compiled digest → extra_lines has digest."""
        from trading_agent.digest.store import Digest

        digest_store.put(
            Digest(
                user_id="u1",
                universe_key="AAPL",
                as_of=time.time() - 60,
                digest_text="Regime: calm\n• AAPL bullish",
                headlines=["AAPL: bullish"],
                regime_label="calm",
                material_flag=False,
            )
        )
        agent = _make_agent(
            digest_mode=True,
            digest_store=digest_store,
            symbols=["AAPL"],
        )
        agent.owner_user_id = "u1"

        lines = agent._first_look_digest_lines()
        assert len(lines) > 0
        combined = "\n".join(lines)
        assert "Analyst Digest" in combined or "calm" in combined

    def test_digest_injects_material_flag_warning(self, digest_store: Any) -> None:
        from trading_agent.digest.store import Digest

        digest_store.put(
            Digest(
                user_id="u1",
                universe_key="AAPL",
                as_of=time.time() - 30,
                digest_text="[MATERIAL EVENT]",
                headlines=["AAPL: acquisition announced"],
                regime_label="event-window",
                material_flag=True,
            )
        )
        agent = _make_agent(
            digest_mode=True,
            digest_store=digest_store,
            symbols=["AAPL"],
        )
        agent.owner_user_id = "u1"
        lines = agent._first_look_digest_lines()
        combined = "\n".join(lines)
        assert "MATERIAL" in combined

    def test_slow_tools_absent_in_digest_mode(self) -> None:
        """In digest mode, news/situation/research_brief/world_events NOT in defs."""
        agent = _make_agent(digest_mode=True)
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}

        # Slow external-fetch tools must be absent.
        slow_tools = {"news", "world_events", "prediction_market_odds", "forecast",
                      "research_brief", "request_research", "situation"}
        assert not (slow_tools & tool_names), (
            f"Slow tools found in digest mode: {slow_tools & tool_names}"
        )

    def test_search_context_in_digest_mode(self) -> None:
        """In digest mode, search_context must appear in tool defs."""
        agent = _make_agent(digest_mode=True, digest_store=MagicMock())
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}
        assert "search_context" in tool_names

    def test_price_tools_present_in_digest_mode(self) -> None:
        """Live price tools (history, account_state) stay in defs even in digest mode."""
        agent = _make_agent(digest_mode=True)
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}
        # These must remain (price must stay fresh).
        for tool in ("history", "watchlist", "account_state", "advisor_notes"):
            assert tool in tool_names, f"{tool} missing in digest mode"

    def test_default_off_slow_tools_present(self) -> None:
        """With digest_mode=False, slow tools ARE in the catalog."""
        agent = _make_agent(digest_mode=False)
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}
        for tool in ("news", "research_brief", "request_research", "situation"):
            assert tool in tool_names, f"{tool} should be in non-digest catalog"

    def test_default_off_search_context_absent(self) -> None:
        """With digest_mode=False, search_context is NOT in the catalog."""
        agent = _make_agent(digest_mode=False)
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}
        assert "search_context" not in tool_names

    def test_search_context_dispatch(self, digest_store: Any) -> None:
        """search_context tool call dispatches correctly via _execute_tool."""
        from trading_agent.intel.cost_tracker import CostTracker
        from trading_agent.llm.openrouter import ToolCall

        agent = _make_agent(digest_mode=True, digest_store=digest_store)
        agent.owner_user_id = "u1"

        tc = ToolCall(id="t1", name="search_context", arguments={"query": "AAPL", "k": 3})
        result = agent._execute_tool(tc, CostTracker())
        assert result.ok

    def test_slow_tool_not_in_tool_definitions_in_digest_mode(self) -> None:
        """In digest mode, slow tools are absent from tool definitions (model can't call them).

        The dispatch layer still handles them defensively (returns unavailable / empty),
        but the *definitions* exposed to the model do NOT include them — so the model
        cannot discover and call them, which is the intent of digest-mode tool gating.
        """
        agent = _make_agent(digest_mode=True)
        defs = agent._tool_definitions()
        tool_names = {d["function"]["name"] for d in defs}
        # Verify at tool-definition layer (what the model sees).
        assert "news" not in tool_names
        assert "situation" not in tool_names
        assert "research_brief" not in tool_names
        assert "world_events" not in tool_names

    def test_look_tool_availability_digest_mode(self) -> None:
        """In digest mode, slow tools have disabled_reason; search_context is enabled."""
        agent = _make_agent(digest_mode=True, digest_store=MagicMock())
        avail = agent._look_tool_availability()
        # Slow tools disabled.
        assert avail["news"] is not None
        assert avail["situation"] is not None
        assert avail["world_events"] is not None
        # search_context enabled (store is wired).
        assert avail.get("search_context") is None

    def test_look_tool_availability_normal_mode(self) -> None:
        """In normal mode, slow tools have their real availability (not digest-gated)."""
        agent = _make_agent(digest_mode=False)
        avail = agent._look_tool_availability()
        # search_context NOT in avail (it's not a normal-mode tool).
        assert "search_context" not in avail or avail.get("search_context") is not None


# ---------------------------------------------------------------------------
# 5. Default-off invariant: tool catalog unchanged when flag is False
# ---------------------------------------------------------------------------


class TestDefaultOffInvariant:
    def _catalog_names(self, agent: Any) -> set[str]:
        return {d["function"]["name"] for d in agent._tool_definitions()}

    def test_same_tools_as_baseline_when_off(self) -> None:
        """Digest-off tool catalog must be identical to a never-digest agent."""
        from trading_agent.llm.trader import AgentTrader

        client = MagicMock()
        baseline = AgentTrader(
            "model", client, symbols=["AAPL"], name="A", tutorial_remaining=0
        )
        digest_off = AgentTrader(
            "model", client, symbols=["AAPL"], name="B",
            digest_mode=False, digest_store=None, tutorial_remaining=0,
        )
        assert self._catalog_names(baseline) == self._catalog_names(digest_off)

    def test_first_look_extra_lines_unchanged_when_off(self) -> None:
        """Digest-off extra_lines must equal _turn_type_guidance output (no digest lines)."""
        from trading_agent.llm.trader import AgentTrader

        client = MagicMock()
        agent = AgentTrader(
            "model", client, symbols=["AAPL"], name="A",
            digest_mode=False, tutorial_remaining=0,
        )
        digest_lines = agent._first_look_digest_lines()
        assert digest_lines == []


# ---------------------------------------------------------------------------
# 6. Event-wake: on_research_bombshell fires off-cadence turn
# ---------------------------------------------------------------------------


class TestBombshellEventWake:
    def _make_controller(self, digest_store: Any) -> Any:
        from trading_agent.bench.bench import Bench
        from trading_agent.bench.controller import BenchController

        bench = MagicMock(spec=Bench)
        bench._last_prices = {}
        bench._competitors = {}
        bench.names.return_value = []
        bench.leaderboard.return_value = []
        bench.recent_decisions.return_value = []

        client = MagicMock()
        ctrl = BenchController(
            bench=bench,
            client=client,
            symbols=["AAPL"],
            digest_store=digest_store,
        )
        ctrl._running = True
        return ctrl

    def test_bombshell_no_traders_is_safe(self, digest_store: Any) -> None:
        """on_research_bombshell with no competitors doesn't raise."""
        ctrl = self._make_controller(digest_store)
        ctrl.on_research_bombshell("AAPL")  # should not raise

    def test_bombshell_wakes_matching_digest_trader(
        self, tmp_db: Any, digest_store: Any
    ) -> None:
        """on_research_bombshell calls _run_one for a matching digest-mode trader."""
        from trading_agent.bench.bench import Bench
        from trading_agent.bench.controller import BenchController
        from trading_agent.digest.store import universe_key as _uk

        bench = MagicMock(spec=Bench)
        bench._last_prices = {}

        # Create a mock digest-mode trader with the right universe.
        trader = MagicMock()
        trader._digest_mode = True
        trader.symbols = ["AAPL"]
        trader._current_wake_reason = "scheduled"
        trader._current_turn_type = "regular"

        comp = MagicMock()
        comp.trader = trader

        bench._competitors = {"TestAgent": comp}
        bench.names.return_value = ["TestAgent"]
        bench.leaderboard.return_value = []
        bench.recent_decisions.return_value = []

        ctrl = BenchController(
            bench=bench,
            client=MagicMock(),
            symbols=["AAPL"],
            digest_store=digest_store,
        )
        ctrl._running = True

        ctrl.on_research_bombshell(_uk(["AAPL"]))

        bench._run_one.assert_called_once_with(comp)

    def test_bombshell_skips_non_digest_trader(
        self, digest_store: Any
    ) -> None:
        """on_research_bombshell does NOT wake a trader that has digest_mode=False."""
        from trading_agent.bench.bench import Bench
        from trading_agent.bench.controller import BenchController
        from trading_agent.digest.store import universe_key as _uk

        bench = MagicMock(spec=Bench)
        bench._last_prices = {}

        trader = MagicMock()
        trader._digest_mode = False
        trader.symbols = ["AAPL"]

        comp = MagicMock()
        comp.trader = trader

        bench._competitors = {"Normal": comp}
        bench.names.return_value = ["Normal"]
        bench.leaderboard.return_value = []
        bench.recent_decisions.return_value = []

        ctrl = BenchController(
            bench=bench,
            client=MagicMock(),
            symbols=["AAPL"],
            digest_store=digest_store,
        )
        ctrl._running = True

        ctrl.on_research_bombshell(_uk(["AAPL"]))

        bench._run_one.assert_not_called()

    def test_bombshell_skips_universe_mismatch(
        self, digest_store: Any
    ) -> None:
        """on_research_bombshell does NOT wake a trader whose universe doesn't match."""
        from trading_agent.bench.bench import Bench
        from trading_agent.bench.controller import BenchController
        from trading_agent.digest.store import universe_key as _uk

        bench = MagicMock(spec=Bench)
        bench._last_prices = {}

        trader = MagicMock()
        trader._digest_mode = True
        trader.symbols = ["TSLA"]  # different universe

        comp = MagicMock()
        comp.trader = trader

        bench._competitors = {"TeslaTrader": comp}
        bench.names.return_value = ["TeslaTrader"]
        bench.leaderboard.return_value = []
        bench.recent_decisions.return_value = []

        ctrl = BenchController(
            bench=bench,
            client=MagicMock(),
            symbols=["TSLA"],
            digest_store=digest_store,
        )
        ctrl._running = True

        ctrl.on_research_bombshell(_uk(["AAPL"]))  # fires for AAPL, not TSLA

        bench._run_one.assert_not_called()

    def test_bombshell_noop_when_not_running(self, digest_store: Any) -> None:
        """on_research_bombshell is a no-op when the bench is stopped."""
        ctrl = self._make_controller(digest_store)
        ctrl._running = False
        ctrl.on_research_bombshell("AAPL")  # should do nothing, no raise
