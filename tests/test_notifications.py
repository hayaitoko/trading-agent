"""Tests for the NotificationCenter aggregator."""

from typing import Any

import pytest

from trading_agent.approval_queue import ApprovalQueue
from trading_agent.audit import AuditLogger
from trading_agent.db import DatabaseManager
from trading_agent.web.market_watch import MarketMoveWatcher
from trading_agent.web.notifications import NotificationCenter


@pytest.fixture
def wiring(tmp_path: Any) -> dict[str, Any]:
    db = DatabaseManager(str(tmp_path / "serve.db"))
    audit = AuditLogger(db, data_dir=tmp_path)
    queue = ApprovalQueue(db_path=tmp_path / "approvals.db", executor=lambda s: {"ok": True})
    watch = MarketMoveWatcher(threshold_pct=2.0)
    center = NotificationCenter(queue, db, watch)
    return {"db": db, "audit": audit, "queue": queue, "watch": watch, "center": center}


def test_pending_approvals_mapped(wiring: dict[str, Any]) -> None:
    wiring["queue"].add({"asset": "AAPL", "side": "LONG", "amount": 3, "price": 190.5})
    notes = wiring["center"].pending_approvals()
    assert len(notes) == 1
    n = notes[0]
    assert n.kind == "approval" and n.actionable and n.proposal_id
    assert n.severity == "action"
    assert "AAPL" in n.title and "LONG 3 AAPL" in n.body


def test_risk_alerts_only_warn_and_error(wiring: dict[str, Any]) -> None:
    audit = wiring["audit"]
    audit.info("just_fyi", module="serve")  # excluded
    audit.warn("hourly_trade_limit_blocked", module="serve")
    audit.error("dispatch_failed", module="serve", details={"error": "boom"})
    notes = wiring["center"].risk_alerts()
    kinds = {n.title for n in notes}
    assert "Order blocked: hourly trade limit reached" in kinds
    assert "Order dispatch failed" in kinds
    assert all(n.kind == "risk" for n in notes)
    # ERROR maps to critical, WARN to warning
    sev = {n.title: n.severity for n in notes}
    assert sev["Order dispatch failed"] == "critical"
    assert sev["Order blocked: hourly trade limit reached"] == "warning"


def test_risk_alerts_newest_first(wiring: dict[str, Any]) -> None:
    for i in range(3):
        wiring["audit"].warn("position_size_blocked", module="serve", details={"i": i})
    notes = wiring["center"].risk_alerts(limit=2)
    assert len(notes) == 2  # limit respected
    # newest row (highest id) first
    assert '"i": 2' in notes[0].body


def test_market_alerts_from_watcher(wiring: dict[str, Any]) -> None:
    wiring["watch"].observe("SYNTH", 100.0)
    wiring["watch"].observe("SYNTH", 105.0)  # +5%
    notes = wiring["center"].market_alerts()
    assert len(notes) == 1
    assert notes[0].kind == "market" and notes[0].severity == "info"
    assert "▲" in notes[0].title and "SYNTH" in notes[0].title


def test_snapshot_shape_and_counts(wiring: dict[str, Any]) -> None:
    wiring["queue"].add({"asset": "X", "side": "LONG", "amount": 1})
    wiring["audit"].warn("kill_switch_active", module="serve")
    wiring["watch"].observe("X", 100.0)
    wiring["watch"].observe("X", 110.0)

    snap = wiring["center"].snapshot()
    assert set(snap) >= {"generated_at", "counts", "approvals", "risk", "market"}
    assert snap["counts"] == {"approvals": 1, "risk": 1, "market": 1}
    assert "account" not in snap  # no provider wired here


def test_snapshot_includes_account_when_provider_set(wiring: dict[str, Any]) -> None:
    center = NotificationCenter(
        wiring["queue"], wiring["db"], wiring["watch"],
        account_provider=lambda: {"cash": 9999.0, "pnl": -1.0},
    )
    snap = center.snapshot()
    assert snap["account"]["cash"] == 9999.0


def test_account_provider_error_is_contained(wiring: dict[str, Any]) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("broker down")

    center = NotificationCenter(wiring["queue"], wiring["db"], wiring["watch"], account_provider=boom)
    snap = center.snapshot()
    assert "error" in snap["account"]
