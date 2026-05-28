"""Tests for DatabaseManager schema/WAL and AuditLogger dual-sink writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from trading_agent.audit import AuditLogger
from trading_agent.db import DatabaseManager

# --- DatabaseManager ---------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "trading.db"))


def test_database_creates_four_tables(db):
    with db.get_connection() as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    expected = {"trades", "signals", "positions", "audit_log"}
    assert expected.issubset(names)


def test_database_uses_wal_journal_mode(db):
    with db.get_connection() as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_database_indexes_exist(db):
    with db.get_connection() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert "idx_trades_symbol_timestamp" in names
    assert "idx_signals_symbol_timestamp" in names
    assert "idx_audit_log_timestamp" in names


# --- AuditLogger -------------------------------------------------------------


@pytest.fixture
def audit_setup(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "audit.db"))
    data_dir = tmp_path / "data"
    logger = AuditLogger(db=db, data_dir=data_dir)
    return logger, db, data_dir


def test_audit_log_writes_jsonl_and_sqlite(audit_setup):
    logger, db, data_dir = audit_setup
    logger.info("hello world", module="testmod", details={"k": "v"})

    # JSONL: one file for today, one line, valid JSON.
    files = list(data_dir.glob("audit.*.jsonl"))
    assert len(files) == 1
    assert files[0].name == f"audit.{datetime.now(UTC).date().isoformat()}.jsonl"
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello world"
    assert parsed["module"] == "testmod"
    assert parsed["details"] == {"k": "v"}

    # SQLite row in audit_log
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT level, message, module, details FROM audit_log ORDER BY id ASC"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "INFO"
    assert rows[0][1] == "hello world"
    assert rows[0][2] == "testmod"
    assert json.loads(rows[0][3]) == {"k": "v"}


def test_audit_level_helpers(audit_setup):
    logger, db, _ = audit_setup
    logger.info("a")
    logger.warn("b")
    logger.error("c")
    with db.get_connection() as conn:
        levels = [r[0] for r in conn.execute("SELECT level FROM audit_log ORDER BY id ASC")]
    assert levels == ["INFO", "WARN", "ERROR"]


def test_audit_trade_records_with_order_details(audit_setup):
    logger, db, data_dir = audit_setup
    order = {"order_id": "ord-1", "symbol": "AAPL", "side": "BUY", "quantity": 1.0}
    logger.trade("placed", order)

    # JSONL captured the order under details.order.
    files = list(data_dir.glob("audit.*.jsonl"))
    line = files[0].read_text(encoding="utf-8").splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["message"] == "trade.placed"
    assert parsed["module"] == "trading"
    assert parsed["details"]["order"] == order

    # SQLite row mirrors it.
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT message, module, details FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row[0] == "trade.placed"
    assert row[1] == "trading"
    assert json.loads(row[2])["order"] == order


def test_audit_data_dir_auto_created(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "audit.db"))
    data_dir = tmp_path / "fresh" / "nested" / "dir"
    # Did not yet exist.
    assert not data_dir.exists()
    AuditLogger(db=db, data_dir=data_dir)
    assert data_dir.exists()
