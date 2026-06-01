"""Tests for PaperBroker durability (fill replay) and direct-trade idempotency.

Scope:
- Fills persisted -> a new PaperBroker instance reconstructs identical
  cash, positions, and realized P&L.
- A fresh DB (no fills) starts at configured initial_balance.
- Idempotency: a duplicate idem_key does NOT re-execute the trade.
- Idempotency survives a simulated restart (new store + broker instance,
  same DB file).
- place_order_idempotent raises when no store is configured.
- Realized P&L tracks correctly across fills.
"""

from __future__ import annotations

import pytest

from trading_agent.enums import OrderSide, OrderType  # noqa: I001
from trading_agent.paper_broker import PaperBroker
from trading_agent.paper_broker_store import PaperBrokerStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> PaperBrokerStore:  # type: ignore[type-arg]
    return PaperBrokerStore(tmp_path / "paper.db")


def _make_broker(
    store: PaperBrokerStore,
    book_id: str = "test_book",
    initial_balance: float = 10_000.0,
    commission_bps: float = 0.0,
) -> PaperBroker:
    b = PaperBroker(
        initial_balance=initial_balance,
        store=store,
        book_id=book_id,
        commission_bps=commission_bps,
    )
    b.connect()
    b.update_market_prices({"AAPL": 100.0, "TSLA": 200.0})
    return b


# ---------------------------------------------------------------------------
# Basic store tests
# ---------------------------------------------------------------------------


def test_store_empty_on_fresh_db(store: PaperBrokerStore) -> None:
    assert store.load_fills("nonexistent") == []
    assert store.fill_count("nonexistent") == 0


def test_store_append_and_load(store: PaperBrokerStore) -> None:
    store.append_fill("bk", "ord-1", "AAPL", "BUY", 5.0, 100.0, 0.0)
    store.append_fill("bk", "ord-2", "AAPL", "SELL", 5.0, 110.0, 0.0)
    fills = store.load_fills("bk")
    assert len(fills) == 2
    assert fills[0].side == "BUY"
    assert fills[1].side == "SELL"
    assert fills[0].fill_seq < fills[1].fill_seq  # ordered ascending


def test_store_book_isolation(store: PaperBrokerStore) -> None:
    store.append_fill("book_a", "o1", "AAPL", "BUY", 1.0, 100.0, 0.0)
    assert store.fill_count("book_a") == 1
    assert store.fill_count("book_b") == 0


# ---------------------------------------------------------------------------
# Fresh DB starts at initial_balance (no fills to replay)
# ---------------------------------------------------------------------------


def test_fresh_db_starts_at_initial_balance(store: PaperBrokerStore) -> None:
    b = _make_broker(store)
    assert b.get_balance()["cash"] == pytest.approx(10_000.0)
    assert b.get_positions() == []
    assert b.realized_pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fill replay reconstructs cash and positions
# ---------------------------------------------------------------------------


def test_replay_single_buy(store: PaperBrokerStore) -> None:
    # First broker: buy 10 AAPL @ 100
    b1 = _make_broker(store)
    result = b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    assert result["status"] == "FILLED"
    cash_after = b1.get_balance()["cash"]
    pos_after = b1.get_positions()

    # Second broker: replays fills from same store/book
    b2 = _make_broker(store)
    assert b2.get_balance()["cash"] == pytest.approx(cash_after)
    assert len(b2.get_positions()) == 1
    assert b2.get_positions()[0]["symbol"] == "AAPL"
    assert b2.get_positions()[0]["quantity"] == pytest.approx(pos_after[0]["quantity"])
    assert b2.get_positions()[0]["avg_price"] == pytest.approx(pos_after[0]["avg_price"])


def test_replay_buy_then_sell_flat(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store)
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    b1.update_market_prices({"AAPL": 110.0})
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    cash_b1 = b1.get_balance()["cash"]
    pnl_b1 = b1.realized_pnl

    b2 = _make_broker(store)
    b2.update_market_prices({"AAPL": 110.0})  # prices not replayed; caller re-seeds
    assert b2.get_balance()["cash"] == pytest.approx(cash_b1)
    assert b2.get_positions() == []
    assert b2.realized_pnl == pytest.approx(pnl_b1)


def test_replay_partial_sell(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store)
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    b1.update_market_prices({"AAPL": 110.0})
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 4.0}
    )
    cash_b1 = b1.get_balance()["cash"]
    pos_b1 = b1.get_position("AAPL")
    pnl_b1 = b1.realized_pnl

    b2 = _make_broker(store)
    b2.update_market_prices({"AAPL": 110.0})
    assert b2.get_balance()["cash"] == pytest.approx(cash_b1)
    pos_b2 = b2.get_position("AAPL")
    assert pos_b2 is not None
    assert pos_b2.quantity == pytest.approx(pos_b1.quantity)  # type: ignore[union-attr]
    assert b2.realized_pnl == pytest.approx(pnl_b1)


def test_replay_multiple_symbols(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store, initial_balance=50_000.0)
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    b1.place_order(
        {"symbol": "TSLA", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    cash_b1 = b1.get_balance()["cash"]
    pos_count_b1 = len(b1.get_positions())

    b2 = _make_broker(store, initial_balance=50_000.0)
    assert b2.get_balance()["cash"] == pytest.approx(cash_b1)
    assert len(b2.get_positions()) == pos_count_b1


def test_replay_with_commission(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store, commission_bps=10.0)  # 0.1% commission
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    cash_b1 = b1.get_balance()["cash"]

    b2 = _make_broker(store, commission_bps=10.0)
    assert b2.get_balance()["cash"] == pytest.approx(cash_b1)


def test_replay_book_isolation_separate_books(store: PaperBrokerStore) -> None:
    b_a = _make_broker(store, book_id="book_a")
    _make_broker(store, book_id="book_b")  # initialize book_b with no fills
    b_a.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0}
    )

    # Replay book_b — should not see book_a's fills
    b_b2 = _make_broker(store, book_id="book_b")
    assert b_b2.get_balance()["cash"] == pytest.approx(10_000.0)
    assert b_b2.get_positions() == []

    # Replay book_a — should reconstruct the buy
    b_a2 = _make_broker(store, book_id="book_a")
    assert b_a2.get_balance()["cash"] == pytest.approx(9_500.0)
    assert len(b_a2.get_positions()) == 1


def test_replay_trade_history_length(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store)
    for _ in range(3):
        b1.place_order(
            {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
        )
    assert len(b1.get_trade_history()) == 3

    b2 = _make_broker(store)
    assert len(b2.get_trade_history()) == 3


# ---------------------------------------------------------------------------
# Realized P&L
# ---------------------------------------------------------------------------


def test_realized_pnl_zero_on_fresh_book(store: PaperBrokerStore) -> None:
    b = _make_broker(store)
    assert b.realized_pnl == pytest.approx(0.0)


def test_realized_pnl_after_sell(store: PaperBrokerStore) -> None:
    b = _make_broker(store)
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    b.update_market_prices({"AAPL": 120.0})
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    # Bought @ 100, sold @ 120: P&L = 10 * 20 = 200
    assert b.realized_pnl == pytest.approx(200.0)


def test_realized_pnl_survives_replay(store: PaperBrokerStore) -> None:
    b1 = _make_broker(store)
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    b1.update_market_prices({"AAPL": 120.0})
    b1.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    pnl_original = b1.realized_pnl

    b2 = _make_broker(store)
    assert b2.realized_pnl == pytest.approx(pnl_original)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_requires_store() -> None:
    b = PaperBroker(initial_balance=1_000.0)
    b.connect()
    b.update_market_prices({"AAPL": 100.0})
    with pytest.raises(RuntimeError, match="PaperBrokerStore"):
        b.place_order_idempotent(
            {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0},
            idem_key="key-1",
        )


def test_idempotent_first_call_executes(store: PaperBrokerStore) -> None:
    b = _make_broker(store)
    result = b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0},
        idem_key="k1",
    )
    assert result is not None
    assert result["status"] == "FILLED"
    assert b.get_balance()["cash"] == pytest.approx(9_500.0)


def test_idempotent_duplicate_does_not_re_execute(store: PaperBrokerStore) -> None:
    b = _make_broker(store)
    r1 = b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0},
        idem_key="k1",
    )
    cash_after_first = b.get_balance()["cash"]

    r2 = b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0},
        idem_key="k1",  # same key
    )
    # Cash must not have changed — no second trade
    assert b.get_balance()["cash"] == pytest.approx(cash_after_first)
    # Returns the cached result
    assert r2 is not None
    assert r2.get("status") == r1.get("status")  # type: ignore[union-attr]
    # Only one fill in the store
    assert store.fill_count("test_book") == 1


def test_idempotent_different_keys_each_execute(store: PaperBrokerStore) -> None:
    b = _make_broker(store, initial_balance=50_000.0)
    b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 2.0},
        idem_key="k-a",
    )
    b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 3.0},
        idem_key="k-b",
    )
    # 2+3 = 5 shares at 100 = 500 spent
    assert b.get_balance()["cash"] == pytest.approx(49_500.0)
    assert store.fill_count("test_book") == 2


def test_idempotent_survives_simulated_restart(tmp_path) -> None:  # type: ignore[type-arg]
    """After a process restart (new store + broker instances), a duplicate
    idem_key must not re-execute the trade."""
    db_path = tmp_path / "paper.db"

    # --- Process 1: execute a trade ---
    store1 = PaperBrokerStore(db_path)
    b1 = PaperBroker(initial_balance=10_000.0, store=store1, book_id="live")
    b1.connect()
    b1.update_market_prices({"AAPL": 100.0})
    b1.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0},
        idem_key="signal-xyz",
    )
    cash_p1 = b1.get_balance()["cash"]  # should be 9000

    # --- Process 2: restart with same DB ---
    store2 = PaperBrokerStore(db_path)
    b2 = PaperBroker(initial_balance=10_000.0, store=store2, book_id="live")
    b2.connect()
    b2.update_market_prices({"AAPL": 100.0})

    # Cash should already be reconstructed via replay
    assert b2.get_balance()["cash"] == pytest.approx(cash_p1)
    assert len(b2.get_positions()) == 1

    # Attempting the same idem_key must NOT re-trade
    r = b2.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0},
        idem_key="signal-xyz",
    )
    assert r is not None
    # Cash must not have dropped further
    assert b2.get_balance()["cash"] == pytest.approx(cash_p1)
    # Still only one fill
    assert store2.fill_count("live") == 1


def test_idempotent_separate_books_do_not_collide(store: PaperBrokerStore) -> None:
    b_a = PaperBroker(initial_balance=10_000.0, store=store, book_id="book_a")
    b_a.connect()
    b_a.update_market_prices({"AAPL": 100.0})

    b_b = PaperBroker(initial_balance=10_000.0, store=store, book_id="book_b")
    b_b.connect()
    b_b.update_market_prices({"AAPL": 100.0})

    # Same idem_key, different book — should execute independently
    b_a.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0},
        idem_key="shared-key",
    )
    b_b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0},
        idem_key="shared-key",
    )
    assert store.fill_count("book_a") == 1
    assert store.fill_count("book_b") == 1
    assert b_a.get_balance()["cash"] == pytest.approx(9_500.0)
    assert b_b.get_balance()["cash"] == pytest.approx(9_500.0)


# ---------------------------------------------------------------------------
# Edge: rejected orders persist their idem_key (no re-fire on the same key)
# ---------------------------------------------------------------------------


def test_idempotent_rejected_order_key_is_stored(store: PaperBrokerStore) -> None:
    b = _make_broker(store, initial_balance=100.0)  # too little cash to buy
    result = b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0},
        idem_key="rej-key",
    )
    assert result is not None
    assert result["status"] == "REJECTED"

    # Replay: key exists, must NOT fire again even if cash were somehow enough
    result2 = b.place_order_idempotent(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0},
        idem_key="rej-key",
    )
    assert result2 is not None
    assert result2["status"] == "REJECTED"
    # No fill was ever stored (order was REJECTED)
    assert store.fill_count("test_book") == 0
