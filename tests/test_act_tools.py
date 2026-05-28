"""Tests for WS-Agent A3: ACT toolkit + approval-callback flow + END terminals.

Coverage:
  - TradeTool: kill switch, idempotency, direct execution, approval-queue path
  - TradeBatchTool: whole-batch kill-switch, per-item results
  - UpdateProtectiveOrderTool: kill switch, broker absent, success path
  - ConfirmTradeTool: pre-approved fill, TTL expiry, wrong status
  - AbandonTradeTool: abandon awaiting + approved, wrong status
  - PendingTradeQueue: propose → set_decision → callback → confirm → abandon → expire_old
  - RiskManager: idempotency check/record, check_batch_blocked, kill-switch integration
  - AgentTrader: ACT terminals recognised, turn_id fresh per call, money-is-real scrub
  - MONEY IS REAL: no paper/sim disclosure in any tool result (carry-over fix verified)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.approval_queue import (
    FillResult,
    PendingTrade,
    PendingTradeQueue,
    TradeIntent,
)
from trading_agent.intel.tools.act._base import _idempotency_key, _scrub_fill
from trading_agent.intel.tools.act.abandon_trade import AbandonTradeTool
from trading_agent.intel.tools.act.confirm_trade import ConfirmTradeTool
from trading_agent.intel.tools.act.trade import TradeTool
from trading_agent.intel.tools.act.trade_batch import TradeBatchTool
from trading_agent.intel.tools.act.update_protective_order import (
    UpdateProtectiveOrderTool,
)
from trading_agent.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_broker(fill_price: float = 150.0, qty: float = 10.0) -> MagicMock:
    broker = MagicMock()
    broker.place_order.return_value = {
        "order_id": "ord-001",
        "symbol": "AAPL",
        "side": "BUY",
        "order_type": "market",
        "quantity": qty,
        "filled_quantity": qty,
        "filled_price": fill_price,
        "status": "filled",
    }
    return broker


def _trade_tool(
    *,
    broker: Any = None,
    risk_manager: Any = None,
    ptq: Any = None,
    requires_approval: bool = False,
    trader_id: str = "TestTrader",
    turn_id: str = "turn-001",
) -> TradeTool:
    return TradeTool(
        broker=broker,
        risk_manager=risk_manager,
        pending_trade_queue=ptq,
        trader_id=trader_id,
        turn_id=turn_id,
        requires_approval=requires_approval,
    )


# ---------------------------------------------------------------------------
# RiskManager A3 additions
# ---------------------------------------------------------------------------


class TestRiskManagerA3:
    def test_idempotency_check_record(self):
        rm = RiskManager()
        key = "abc123"
        assert rm.check_idempotency(key) is False
        rm.record_idempotency(key)
        assert rm.check_idempotency(key) is True

    def test_idempotency_kill_switch_forces_true(self):
        rm = RiskManager()
        rm.activate_kill_switch()
        assert rm.check_idempotency("any-key") is True

    def test_check_batch_blocked_kill_switch(self):
        rm = RiskManager()
        rm.activate_kill_switch()
        intents = [{"qty": 10}, {"qty": 20}]
        assert rm.check_batch_blocked("scope", intents) == [True, True]

    def test_check_batch_blocked_size_limit(self):
        from trading_agent.risk_manager import RiskLimits

        rm = RiskManager(limits=RiskLimits(max_position_size=5.0))
        intents = [{"qty": 3}, {"qty": 10}]
        blocked = rm.check_batch_blocked("scope", intents)
        assert blocked[0] is False  # 3 <= 5
        assert blocked[1] is True   # 10 > 5

    def test_check_batch_blocked_empty(self):
        rm = RiskManager()
        assert rm.check_batch_blocked("scope", []) == []


# ---------------------------------------------------------------------------
# _idempotency_key helper
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    def test_deterministic(self):
        k1 = _idempotency_key("T", "turn1", "AAPL", "BUY", 10.0)
        k2 = _idempotency_key("T", "turn1", "AAPL", "BUY", 10.0)
        assert k1 == k2

    def test_case_insensitive(self):
        k1 = _idempotency_key("T", "t1", "aapl", "buy", 10.0)
        k2 = _idempotency_key("T", "t1", "AAPL", "BUY", 10.0)
        assert k1 == k2

    def test_different_turn_ids_differ(self):
        k1 = _idempotency_key("T", "turn1", "AAPL", "BUY", 10.0)
        k2 = _idempotency_key("T", "turn2", "AAPL", "BUY", 10.0)
        assert k1 != k2


# ---------------------------------------------------------------------------
# _scrub_fill
# ---------------------------------------------------------------------------


class TestScrubFill:
    def test_removes_paper_value(self):
        result = {"status": "paper filled", "order_id": "x1"}
        scrubbed = _scrub_fill(result)
        assert "status" not in scrubbed
        assert scrubbed["order_id"] == "x1"

    def test_keeps_clean_values(self):
        result = {"status": "filled", "order_id": "x2", "filled_price": 100.0}
        assert _scrub_fill(result) == result

    def test_numeric_values_not_affected(self):
        result = {"filled_price": 150.0, "qty": 10}
        assert _scrub_fill(result) == result


# ---------------------------------------------------------------------------
# TradeTool
# ---------------------------------------------------------------------------


class TestTradeTool:
    def test_kill_switch_blocks(self):
        rm = RiskManager()
        rm.activate_kill_switch()
        tool = _trade_tool(broker=_mock_broker(), risk_manager=rm)
        res = tool.run("AAPL", "BUY", 10)
        assert res.ok is False
        assert res.error.kind == "unavailable"
        assert "halted" in res.error.message

    def test_invalid_symbol_empty(self):
        tool = _trade_tool(broker=_mock_broker())
        res = tool.run("", "BUY", 10)
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_invalid_side(self):
        tool = _trade_tool(broker=_mock_broker())
        res = tool.run("AAPL", "LONG", 10)
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_invalid_qty_zero(self):
        tool = _trade_tool(broker=_mock_broker())
        res = tool.run("AAPL", "BUY", 0)
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_invalid_qty_negative(self):
        tool = _trade_tool(broker=_mock_broker())
        res = tool.run("AAPL", "BUY", -5)
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_broker_absent(self):
        tool = _trade_tool(broker=None)
        res = tool.run("AAPL", "BUY", 10)
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_direct_execution(self):
        broker = _mock_broker()
        tool = _trade_tool(broker=broker)
        res = tool.run("AAPL", "BUY", 10)
        assert res.ok is True
        assert "fill" in res.data
        assert res.data["fill"]["order_id"] == "ord-001"
        broker.place_order.assert_called_once()

    def test_fill_scrubbed_of_paper_strings(self):
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "x1",
            "status": "paper trade complete",  # should be stripped
            "filled_price": 100.0,
        }
        tool = _trade_tool(broker=broker)
        res = tool.run("AAPL", "BUY", 5)
        assert res.ok is True
        # "status" field had "paper" in it — scrubbed
        assert "status" not in res.data["fill"]
        assert res.data["fill"]["filled_price"] == 100.0

    def test_idempotency_reject_duplicate(self):
        rm = RiskManager()
        broker = _mock_broker()
        tool1 = _trade_tool(broker=broker, risk_manager=rm, turn_id="t1")
        tool2 = _trade_tool(broker=broker, risk_manager=rm, turn_id="t1")
        res1 = tool1.run("AAPL", "BUY", 10)
        assert res1.ok is True
        # Same turn_id + same symbol/side/qty → duplicate
        res2 = tool2.run("AAPL", "BUY", 10)
        assert res2.ok is False
        assert res2.error.kind == "invalid_input"
        assert "duplicate" in res2.error.message

    def test_different_turn_ids_allowed(self):
        rm = RiskManager()
        broker = _mock_broker()
        t1 = _trade_tool(broker=broker, risk_manager=rm, turn_id="t1")
        t2 = _trade_tool(broker=broker, risk_manager=rm, turn_id="t2")
        assert t1.run("AAPL", "BUY", 10).ok is True
        # Same params but different turn_id → new key → allowed
        assert t2.run("AAPL", "BUY", 10).ok is True

    def test_approval_required_enqueues(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        tool = _trade_tool(ptq=ptq, requires_approval=True)
        res = tool.run("AAPL", "BUY", 10)
        assert res.ok is True
        assert "pending_trade_id" in res.data
        assert res.data["status"] == "awaiting_approval"
        ptq.close()

    def test_approval_required_no_broker_still_enqueues(self, tmp_path):
        """Broker is not needed on the propose path — only on confirm."""
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        tool = _trade_tool(broker=None, ptq=ptq, requires_approval=True)
        res = tool.run("MSFT", "SELL", 5)
        assert res.ok is True
        assert res.data["status"] == "awaiting_approval"
        ptq.close()

    def test_no_paper_string_in_result(self):
        broker = _mock_broker()
        tool = _trade_tool(broker=broker)
        res = tool.run("AAPL", "BUY", 10)
        result_str = str(res.data)
        for word in ("paper", "sim", "demo", "fake"):
            assert word not in result_str.lower(), f"Forbidden word {word!r} found in result"


# ---------------------------------------------------------------------------
# TradeBatchTool
# ---------------------------------------------------------------------------


class TestTradeBatchTool:
    def test_empty_batch_rejected(self):
        tool = TradeBatchTool(broker=_mock_broker(), trader_id="T", turn_id="t")
        res = tool.run([])
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_kill_switch_blocks_whole_batch(self):
        rm = RiskManager()
        rm.activate_kill_switch()
        tool = TradeBatchTool(broker=_mock_broker(), risk_manager=rm, trader_id="T", turn_id="t")
        res = tool.run([{"symbol": "AAPL", "side": "BUY", "qty": 10}])
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_per_item_results(self):
        broker = _mock_broker()
        tool = TradeBatchTool(broker=broker, trader_id="T", turn_id="t")
        trades = [
            {"symbol": "AAPL", "side": "BUY", "qty": 5},
            {"symbol": "MSFT", "side": "SELL", "qty": 3},
        ]
        res = tool.run(trades)
        assert res.ok is True
        assert len(res.data["results"]) == 2
        for item in res.data["results"]:
            assert "result" in item
            assert item["result"]["ok"] is True

    def test_idempotency_catches_duplicates_within_batch(self):
        rm = RiskManager()
        broker = _mock_broker()
        tool = TradeBatchTool(broker=broker, risk_manager=rm, trader_id="T", turn_id="t")
        # Same symbol+side+qty twice in one batch → second is a duplicate
        trades = [
            {"symbol": "AAPL", "side": "BUY", "qty": 10},
            {"symbol": "AAPL", "side": "BUY", "qty": 10},
        ]
        res = tool.run(trades)
        assert res.ok is True
        results = res.data["results"]
        assert results[0]["result"]["ok"] is True
        assert results[1]["result"]["ok"] is False
        assert results[1]["result"]["error"]["kind"] == "invalid_input"


# ---------------------------------------------------------------------------
# UpdateProtectiveOrderTool
# ---------------------------------------------------------------------------


class TestUpdateProtectiveOrderTool:
    def _tool(self, *, broker=None, rm=None) -> UpdateProtectiveOrderTool:
        return UpdateProtectiveOrderTool(
            broker=broker, risk_manager=rm, trader_id="T", turn_id="t"
        )

    def test_kill_switch(self):
        rm = RiskManager()
        rm.activate_kill_switch()
        res = self._tool(broker=_mock_broker(), rm=rm).run("ord-1", new_stop=100.0)
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_broker_absent(self):
        res = self._tool(broker=None).run("ord-1", new_stop=100.0)
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_empty_order_id(self):
        res = self._tool(broker=_mock_broker()).run("", new_stop=100.0)
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_no_params(self):
        res = self._tool(broker=_mock_broker()).run("ord-1")
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_success(self):
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "ord-1", "status": "updated"}
        res = self._tool(broker=broker).run("ord-1", new_stop=140.0, new_tp=160.0)
        assert res.ok is True
        assert res.data["order_id"] == "ord-1"
        call_args = broker.place_order.call_args[0][0]
        assert call_args["stop"] == 140.0
        assert call_args["take_profit"] == 160.0

    def test_broker_returns_none(self):
        broker = MagicMock()
        broker.place_order.return_value = None
        res = self._tool(broker=broker).run("ord-missing", new_stop=100.0)
        assert res.ok is False
        assert res.error.kind == "not_found"


# ---------------------------------------------------------------------------
# PendingTradeQueue lifecycle
# ---------------------------------------------------------------------------


class TestPendingTradeQueue:
    def test_propose_returns_awaiting(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("trader1", intent, "key-001")
        assert pt.status == "awaiting_approval"
        assert pt.trader_id == "trader1"
        assert pt.proposed.symbol == "AAPL"
        ptq.close()

    def test_duplicate_idempotency_key_raises(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        ptq.propose("trader1", intent, "key-dup")
        with pytest.raises(ValueError, match="duplicate"):
            ptq.propose("trader1", intent, "key-dup")
        ptq.close()

    def test_set_decision_approved_fires_callback(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("trader1", intent, "key-cb-1")

        fired: list[PendingTrade] = []
        ptq.register_callback(pt.pending_trade_id, fired.append)

        approved = ptq.set_decision(pt.pending_trade_id, "approved")
        assert approved.status == "approved"
        assert approved.approval_ttl_expires_at is not None
        assert len(fired) == 1
        assert fired[0].status == "approved"
        ptq.close()

    def test_set_decision_denied_fires_callback(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="MSFT", side="SELL", qty=5.0)
        pt = ptq.propose("trader1", intent, "key-deny-1")

        fired: list[PendingTrade] = []
        ptq.register_callback(pt.pending_trade_id, fired.append)

        denied = ptq.set_decision(pt.pending_trade_id, "denied", note="market too choppy")
        assert denied.status == "denied"
        assert denied.note == "market too choppy"
        assert len(fired) == 1
        ptq.close()

    def test_set_decision_invalid_decision_raises(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("t", intent, "k1")
        with pytest.raises(ValueError, match="'approved' or 'denied'"):
            ptq.set_decision(pt.pending_trade_id, "maybe")
        ptq.close()

    def test_confirm_returns_fill(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("trader1", intent, "key-conf-1")
        ptq.set_decision(pt.pending_trade_id, "approved")

        def executor(intent):
            return FillResult(
                order_id="ord-fill-1",
                symbol=intent.symbol,
                side=intent.side,
                qty_filled=intent.qty,
                fill_price=150.0,
                status="filled",
            )

        confirmed_pt, fill = ptq.confirm(pt.pending_trade_id, executor)
        assert confirmed_pt.status == "confirmed"
        assert fill.order_id == "ord-fill-1"
        assert fill.fill_price == 150.0
        ptq.close()

    def test_confirm_ttl_expired(self, tmp_path):

        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        ptq.PREAPPROVAL_TTL_MIN = 0  # force immediate expiry
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("trader1", intent, "key-ttl-1")
        ptq.set_decision(pt.pending_trade_id, "approved")

        # Wait a moment so TTL has elapsed
        time.sleep(0.01)

        with pytest.raises(ValueError, match="TTL expired"):
            ptq.confirm(pt.pending_trade_id, lambda i: None)
        ptq.close()

    def test_confirm_wrong_status_raises(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("trader1", intent, "key-ws-1")
        # Not approved yet — status is awaiting_approval
        with pytest.raises(ValueError, match="awaiting_approval"):
            ptq.confirm(pt.pending_trade_id, lambda i: None)
        ptq.close()

    def test_abandon_awaiting(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("t", intent, "k-ab-1")
        result = ptq.abandon(pt.pending_trade_id)
        assert result.status == "abandoned"
        ptq.close()

    def test_abandon_approved(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("t", intent, "k-ab-2")
        ptq.set_decision(pt.pending_trade_id, "approved")
        result = ptq.abandon(pt.pending_trade_id)
        assert result.status == "abandoned"
        ptq.close()

    def test_abandon_denied_raises(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("t", intent, "k-ab-3")
        ptq.set_decision(pt.pending_trade_id, "denied")
        with pytest.raises(ValueError, match="denied"):
            ptq.abandon(pt.pending_trade_id)
        ptq.close()

    def test_expire_old(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        ptq.PREAPPROVAL_TTL_MIN = 0  # immediate expiry

        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("t", intent, "k-exp-1")
        ptq.set_decision(pt.pending_trade_id, "approved")

        fired: list[PendingTrade] = []
        ptq.register_callback(pt.pending_trade_id, fired.append)

        time.sleep(0.01)
        n = ptq.expire_old()
        assert n >= 1
        refreshed = ptq.get(pt.pending_trade_id)
        assert refreshed.status == "expired"
        assert len(fired) == 1
        ptq.close()

    def test_pending_for_trader(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        for i, sym in enumerate(("AAPL", "MSFT", "GOOG")):
            ptq.propose("trader1", TradeIntent(symbol=sym, side="BUY", qty=1.0), f"k-{i}")
        pts = ptq.pending_for_trader("trader1")
        assert len(pts) == 3
        assert all(p.trader_id == "trader1" for p in pts)
        ptq.close()


# ---------------------------------------------------------------------------
# ConfirmTradeTool
# ---------------------------------------------------------------------------


class TestConfirmTradeTool:
    def test_kill_switch(self, tmp_path):
        rm = RiskManager()
        rm.activate_kill_switch()
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        tool = ConfirmTradeTool(
            broker=_mock_broker(),
            risk_manager=rm,
            pending_trade_queue=ptq,
            trader_id="T",
            turn_id="t",
        )
        res = tool.run("any-id")
        assert res.ok is False
        assert res.error.kind == "unavailable"
        ptq.close()

    def test_empty_id(self):
        tool = ConfirmTradeTool(trader_id="T", turn_id="t")
        res = tool.run("")
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_queue_absent(self):
        tool = ConfirmTradeTool(broker=_mock_broker(), trader_id="T", turn_id="t")
        res = tool.run("some-id")
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_not_found(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        tool = ConfirmTradeTool(
            broker=_mock_broker(), pending_trade_queue=ptq, trader_id="T", turn_id="t"
        )
        res = tool.run("nonexistent-id")
        assert res.ok is False
        assert res.error.kind == "not_found"
        ptq.close()

    def test_full_confirm_flow(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        broker = _mock_broker(fill_price=150.0, qty=10.0)

        # Propose + approve via queue
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=10.0)
        pt = ptq.propose("T", intent, "k-cf-1")
        ptq.set_decision(pt.pending_trade_id, "approved")

        tool = ConfirmTradeTool(
            broker=broker, pending_trade_queue=ptq, trader_id="T", turn_id="t"
        )
        res = tool.run(pt.pending_trade_id)
        assert res.ok is True
        assert res.data["fill"]["qty_filled"] == 10.0
        assert res.data["fill"]["fill_price"] == 150.0
        ptq.close()

    def test_no_paper_in_fill(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "x",
            "filled_quantity": 5.0,
            "filled_price": 100.0,
            "status": "filled",
        }
        intent = TradeIntent(symbol="MSFT", side="SELL", qty=5.0)
        pt = ptq.propose("T", intent, "k-cf-nopaper")
        ptq.set_decision(pt.pending_trade_id, "approved")

        tool = ConfirmTradeTool(
            broker=broker, pending_trade_queue=ptq, trader_id="T", turn_id="t"
        )
        res = tool.run(pt.pending_trade_id)
        assert res.ok is True
        result_str = str(res.data)
        for word in ("paper", "sim", "demo", "fake"):
            assert word not in result_str.lower()
        ptq.close()


# ---------------------------------------------------------------------------
# AbandonTradeTool
# ---------------------------------------------------------------------------


class TestAbandonTradeTool:
    def test_empty_id(self):
        tool = AbandonTradeTool(trader_id="T", turn_id="t")
        res = tool.run("")
        assert res.ok is False
        assert res.error.kind == "invalid_input"

    def test_queue_absent(self):
        tool = AbandonTradeTool(trader_id="T", turn_id="t")
        res = tool.run("some-id")
        assert res.ok is False
        assert res.error.kind == "unavailable"

    def test_not_found(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        tool = AbandonTradeTool(pending_trade_queue=ptq, trader_id="T", turn_id="t")
        res = tool.run("nonexistent")
        assert res.ok is False
        assert res.error.kind == "not_found"
        ptq.close()

    def test_abandon_awaiting(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("T", intent, "k-at-1")

        tool = AbandonTradeTool(pending_trade_queue=ptq, trader_id="T", turn_id="t")
        res = tool.run(pt.pending_trade_id)
        assert res.ok is True
        assert res.data["status"] == "abandoned"
        assert res.data["symbol"] == "AAPL"
        ptq.close()

    def test_abandon_approved(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("T", intent, "k-at-2")
        ptq.set_decision(pt.pending_trade_id, "approved")

        tool = AbandonTradeTool(pending_trade_queue=ptq, trader_id="T", turn_id="t")
        res = tool.run(pt.pending_trade_id)
        assert res.ok is True
        assert res.data["status"] == "abandoned"
        ptq.close()

    def test_abandon_already_denied(self, tmp_path):
        ptq = PendingTradeQueue(db_path=str(tmp_path / "test.db"))
        intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
        pt = ptq.propose("T", intent, "k-at-3")
        ptq.set_decision(pt.pending_trade_id, "denied")

        tool = AbandonTradeTool(pending_trade_queue=ptq, trader_id="T", turn_id="t")
        res = tool.run(pt.pending_trade_id)
        assert res.ok is False
        assert res.error.kind == "invalid_input"
        ptq.close()


# ---------------------------------------------------------------------------
# End-to-end approval-callback smoke (verifies all 5 paths)
# ---------------------------------------------------------------------------


class TestApprovalCallbackSmoke:
    """Live smoke: end-to-end approval-callback simulation (plan §A3 requirement)."""

    def test_full_flow(self, tmp_path):
        """
        1. trade(AAPL, BUY, 10) → pending_trade_id, status=awaiting_approval
        2. set_decision(approved) → callback fires, sees TTL
        3. confirm_trade(id) → fill returned
        """
        ptq = PendingTradeQueue(db_path=str(tmp_path / "smoke.db"))
        broker = _mock_broker(fill_price=153.25, qty=10.0)
        rm = RiskManager()

        # Step 1: trade() → awaiting_approval
        trade_tool = TradeTool(
            broker=None,  # not needed for approval-queue path
            risk_manager=rm,
            pending_trade_queue=ptq,
            trader_id="SmokeTrader",
            turn_id="smoke-turn-001",
            requires_approval=True,
        )
        res = trade_tool.run("AAPL", "BUY", 10)
        assert res.ok is True, f"trade() failed: {res}"
        assert res.data["status"] == "awaiting_approval"
        pending_id = res.data["pending_trade_id"]

        # Step 2: operator approves → callback fires
        callbacks_received: list[PendingTrade] = []
        ptq.register_callback(pending_id, callbacks_received.append)

        approved_pt = ptq.set_decision(pending_id, "approved", note="looks good")
        assert approved_pt.status == "approved"
        assert approved_pt.approval_ttl_expires_at is not None
        assert len(callbacks_received) == 1, "callback must fire on approval"
        cb = callbacks_received[0]
        assert cb.status == "approved"
        assert cb.approval_ttl_expires_at is not None  # TTL in callback

        # Step 3: confirm_trade(id) → fill
        confirm_tool = ConfirmTradeTool(
            broker=broker,
            pending_trade_queue=ptq,
            trader_id="SmokeTrader",
            turn_id="smoke-callback-001",
        )
        conf_res = confirm_tool.run(pending_id)
        assert conf_res.ok is True, f"confirm_trade() failed: {conf_res}"
        fill = conf_res.data["fill"]
        assert fill["qty_filled"] == 10.0
        assert fill["fill_price"] == 153.25

        ptq.close()

    def test_deny_flow(self, tmp_path):
        """deny → callback turn with denial; no fill."""
        ptq = PendingTradeQueue(db_path=str(tmp_path / "deny.db"))

        trade_tool = TradeTool(
            pending_trade_queue=ptq,
            trader_id="DenyTrader",
            turn_id="deny-turn-001",
            requires_approval=True,
        )
        res = trade_tool.run("TSLA", "SELL", 3)
        assert res.ok is True
        pending_id = res.data["pending_trade_id"]

        denied_pts: list[PendingTrade] = []
        ptq.register_callback(pending_id, denied_pts.append)

        ptq.set_decision(pending_id, "denied", note="too risky")
        assert len(denied_pts) == 1
        assert denied_pts[0].status == "denied"
        assert denied_pts[0].note == "too risky"

        # Confirm on denied trade → error
        confirm_tool = ConfirmTradeTool(
            broker=_mock_broker(), pending_trade_queue=ptq, trader_id="DenyTrader", turn_id="cb"
        )
        res2 = confirm_tool.run(pending_id)
        assert res2.ok is False

        ptq.close()

    def test_abandon_flow(self, tmp_path):
        """approve → abandon → no fill."""
        ptq = PendingTradeQueue(db_path=str(tmp_path / "abandon.db"))

        trade_tool = TradeTool(
            pending_trade_queue=ptq,
            trader_id="AbandonTrader",
            turn_id="ab-turn-001",
            requires_approval=True,
        )
        res = trade_tool.run("NVDA", "BUY", 2)
        assert res.ok is True
        pending_id = res.data["pending_trade_id"]

        ptq.set_decision(pending_id, "approved")

        abandon_tool = AbandonTradeTool(
            pending_trade_queue=ptq, trader_id="AbandonTrader", turn_id="cb-ab"
        )
        ab_res = abandon_tool.run(pending_id)
        assert ab_res.ok is True
        assert ab_res.data["status"] == "abandoned"

        ptq.close()

    def test_ttl_expiry_fires_callback(self, tmp_path):
        """approved-but-not-confirmed → expire_old → expiry callback fires."""
        ptq = PendingTradeQueue(db_path=str(tmp_path / "ttl.db"))
        ptq.PREAPPROVAL_TTL_MIN = 0  # expire immediately

        trade_tool = TradeTool(
            pending_trade_queue=ptq,
            trader_id="TTLTrader",
            turn_id="ttl-turn-001",
            requires_approval=True,
        )
        res = trade_tool.run("AMZN", "BUY", 1)
        pending_id = res.data["pending_trade_id"]
        ptq.set_decision(pending_id, "approved")

        expired_pts: list[PendingTrade] = []
        ptq.register_callback(pending_id, expired_pts.append)

        time.sleep(0.01)
        n = ptq.expire_old()
        assert n >= 1
        assert len(expired_pts) == 1
        assert expired_pts[0].status == "expired"

        ptq.close()


# ---------------------------------------------------------------------------
# AgentTrader A3 integration
# ---------------------------------------------------------------------------


class TestAgentTraderA3:
    """Verify AgentTrader handles A3 terminals and turn_id refresh."""

    def _make_trader(self, broker=None, ptq=None):
        from trading_agent.llm.trader import AgentTrader

        return AgentTrader(
            model="test-model",
            client=MagicMock(),
            symbols=["AAPL"],
            name="TestAgent",
            broker=broker,
            pending_trade_queue=ptq,
        )

    def test_turn_id_refreshes_per_decide(self):
        trader = self._make_trader()
        from trading_agent.llm.openrouter import ToolCallChatResult

        trader.client.chat_with_tools.return_value = ToolCallChatResult(
            content="hold", tool_calls=[], model="test-model", usage={}, cost=0.0
        )
        trader.decide({"cash": 10000, "positions": []})
        tid1 = trader._current_turn_id
        trader.decide({"cash": 10000, "positions": []})
        tid2 = trader._current_turn_id
        assert tid1 != tid2, "turn_id must be fresh each decide() call"

    def test_done_for_day_terminal_in_tool_defs(self):
        trader = self._make_trader()
        defs = trader._tool_definitions()
        names = [d["function"]["name"] for d in defs]
        assert "done_for_day" in names

    def test_act_tools_appear_when_broker_wired(self):
        trader = self._make_trader(broker=_mock_broker())
        defs = trader._tool_definitions()
        names = [d["function"]["name"] for d in defs]
        assert "trade" in names
        assert "trade_batch" in names
        assert "confirm_trade" in names
        assert "abandon_trade" in names

    def test_act_tools_absent_when_no_broker(self):
        trader = self._make_trader(broker=None)
        defs = trader._tool_definitions()
        names = [d["function"]["name"] for d in defs]
        for n in ("trade", "trade_batch", "confirm_trade", "abandon_trade"):
            assert n not in names, f"ACT tool {n!r} should not appear without broker"

    def test_trade_terminal_recognised(self):
        from trading_agent.llm.trader import _TERMINALS

        for name in ("trade", "trade_batch", "confirm_trade", "abandon_trade", "done_for_day"):
            assert name in _TERMINALS, f"{name!r} must be in _TERMINALS"

    def test_system_prompt_no_paper_strings(self):
        trader = self._make_trader(broker=_mock_broker())
        prompt = trader._stable_system_content.lower()
        for word in ("paper", "sim", "demo", "fake"):
            assert word not in prompt, f"Forbidden word {word!r} in system prompt"


# ---------------------------------------------------------------------------
# MONEY IS REAL: carry-over fix verification
# ---------------------------------------------------------------------------


class TestMoneyIsRealCarryoverFix:
    """Verify ask_manager TOOL_META no longer leaks 'account simulation status'."""

    def test_list_tools_ask_manager_description_no_simulation_leak(self):
        from trading_agent.intel.tools.look.list_tools import ListToolsTool

        tool = ListToolsTool(trader_id="test")
        result = tool()
        assert result.ok
        tools = result.data["tools"]
        ask_manager_entry = next(
            (t for t in tools if t["name"] == "ask_manager"), None
        )
        assert ask_manager_entry is not None
        desc = ask_manager_entry["description"].lower()
        assert "simulation" not in desc, "ask_manager description must not mention simulation"
        assert "account simulation status" not in desc

    def test_ask_manager_tool_meta_no_simulation_leak(self):
        from trading_agent.intel.tools.look.ask_manager import AskManagerTool

        desc = AskManagerTool.TOOL_META["description"].lower()
        assert "simulation" not in desc
        assert "account simulation status" not in desc
