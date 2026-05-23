"""Tests for ApprovalQueue: add/approve/reject/expirations/lifecycle."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.approval_queue import ApprovalQueue


@pytest.fixture
def signal() -> dict[str, Any]:
    return {"symbol": "AAPL", "side": "BUY", "amount": 1.0, "order_type": "market"}


@pytest.fixture
def queue(tmp_path):
    executor = MagicMock(return_value={"order_id": "ord-1", "status": "FILLED"})
    q = ApprovalQueue(db_path=tmp_path / "approvals.db", executor=executor)
    yield q
    q.close()


# --- add ----------------------------------------------------------------------


def test_add_returns_id_and_sets_pending(queue, signal):
    proposal_id = queue.add(signal)
    assert isinstance(proposal_id, str)
    record = queue.get(proposal_id)
    assert record is not None
    assert record.status == "pending"
    assert record.signal == signal
    assert record.decided_at is None


def test_pending_lists_pending_proposals(queue, signal):
    id1 = queue.add(signal)
    id2 = queue.add({**signal, "symbol": "TSLA"})
    ids = {r.proposal_id for r in queue.pending()}
    assert ids == {id1, id2}


# --- approve ------------------------------------------------------------------


def test_approve_invokes_executor_and_marks_approved(queue, signal):
    proposal_id = queue.add(signal)
    result = queue.approve(proposal_id, note="ok")
    assert result == {"order_id": "ord-1", "status": "FILLED"}
    queue.executor.assert_called_once_with(signal)

    record = queue.get(proposal_id)
    assert record.status == "approved"
    assert record.decided_at is not None
    assert record.note == "ok"
    assert record.execution_result == {"order_id": "ord-1", "status": "FILLED"}


def test_approve_without_executor_raises(tmp_path, signal):
    q = ApprovalQueue(db_path=tmp_path / "noexec.db", executor=None)
    try:
        proposal_id = q.add(signal)
        with pytest.raises(RuntimeError):
            q.approve(proposal_id)
    finally:
        q.close()


def test_approve_unknown_id_raises_key_error(queue):
    with pytest.raises(KeyError):
        queue.approve("nonexistent-id")


def test_double_approve_raises_value_error(queue, signal):
    proposal_id = queue.add(signal)
    queue.approve(proposal_id)
    with pytest.raises(ValueError):
        queue.approve(proposal_id)


# --- reject -------------------------------------------------------------------


def test_reject_marks_rejected_without_executor_call(queue, signal):
    proposal_id = queue.add(signal)
    queue.reject(proposal_id, note="nope")
    record = queue.get(proposal_id)
    assert record.status == "rejected"
    assert record.note == "nope"
    assert record.decided_at is not None
    queue.executor.assert_not_called()


def test_reject_unknown_id_raises_key_error(queue):
    with pytest.raises(KeyError):
        queue.reject("missing")


def test_reject_already_decided_raises(queue, signal):
    proposal_id = queue.add(signal)
    queue.reject(proposal_id)
    with pytest.raises(ValueError):
        queue.reject(proposal_id)


# --- expirations --------------------------------------------------------------


def test_process_expirations_moves_pending_past_deadline(queue, signal):
    proposal_id = queue.add(signal, timeout_seconds=0)
    # Tiny sleep so expires_at <= now.
    time.sleep(0.01)
    moved = queue.process_expirations()
    assert moved == 1
    record = queue.get(proposal_id)
    assert record.status == "expired"


def test_pending_filters_out_expired(queue, signal):
    expiring_id = queue.add(signal, timeout_seconds=0)
    fresh_id = queue.add({**signal, "symbol": "TSLA"}, timeout_seconds=3600)
    time.sleep(0.01)
    pending_ids = {r.proposal_id for r in queue.pending()}
    assert expiring_id not in pending_ids
    assert fresh_id in pending_ids


def test_approve_expired_proposal_raises_value_error(queue, signal):
    # add with 0s timeout, sleep, then approve should fail because
    # approve() runs process_expirations() first which flips the status.
    proposal_id = queue.add(signal, timeout_seconds=0)
    time.sleep(0.01)
    with pytest.raises(ValueError):
        queue.approve(proposal_id)
    queue.executor.assert_not_called()


# --- get ----------------------------------------------------------------------


def test_get_unknown_id_returns_none(queue):
    assert queue.get("not-real") is None
