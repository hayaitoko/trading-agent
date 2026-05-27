"""Short-selling tests for PaperBroker (``allow_short`` path).

Covers: long-only default still rejects naked sells, opening/covering a short,
margin rejection (ratio + hard cap), realized P&L on a winning and a losing
short, and the win/loss counts those produce.
"""

from __future__ import annotations

import pytest

from trading_agent.enums import OrderSide, OrderType
from trading_agent.paper_broker import PaperBroker


def _broker(**kwargs) -> PaperBroker:
    b = PaperBroker(initial_balance=100_000.0, **kwargs)
    b.connect()
    return b


def _sell(b: PaperBroker, symbol: str, amount: float) -> dict:
    return b.place_order(
        {"symbol": symbol, "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": amount}
    )


def _buy(b: PaperBroker, symbol: str, amount: float) -> dict:
    return b.place_order(
        {"symbol": symbol, "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": amount}
    )


# --- default is still long-only ----------------------------------------------


def test_naked_sell_rejected_when_shorting_disabled():
    b = _broker()  # allow_short defaults False
    b.update_market_prices({"AAPL": 100.0})
    result = _sell(b, "AAPL", 10)
    assert result["status"] == "REJECTED"
    assert b.get_position("AAPL") is None


# --- opening and covering a short --------------------------------------------


def test_open_short_creates_negative_position_and_credits_cash():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    result = _sell(b, "AAPL", 10)
    assert result["status"] == "FILLED"
    pos = b.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == -10.0
    assert pos.avg_price == 100.0
    # Short proceeds land in cash.
    assert b.get_balance()["cash"] == pytest.approx(101_000.0)


def test_extending_a_short_averages_entry_price():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10 @ 100
    b.update_market_prices({"AAPL": 120.0})
    _sell(b, "AAPL", 10)  # short another 10 @ 120
    pos = b.get_position("AAPL")
    assert pos.quantity == -20.0
    assert pos.avg_price == pytest.approx(110.0)  # (10*100 + 10*120) / 20


def test_covering_reduces_short_and_can_flatten():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10
    cover = _buy(b, "AAPL", 4)  # cover 4 → short 6
    assert cover["status"] == "FILLED"
    assert b.get_position("AAPL").quantity == -6.0
    _buy(b, "AAPL", 6)  # cover the rest → flat
    assert b.get_position("AAPL") is None


def test_buy_can_flip_short_to_long():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10 @ 100
    flip = _buy(b, "AAPL", 15)  # cover 10, open long 5
    assert flip["status"] == "FILLED"
    pos = b.get_position("AAPL")
    assert pos.quantity == 5.0
    assert pos.avg_price == 100.0  # new long basis = fill price


# --- margin / exposure rejection ---------------------------------------------


def test_short_rejected_when_exceeding_margin_ratio():
    # equity = 100k cash; 1.5x margin means ~66.6k max short notional.
    b = _broker(allow_short=True, short_margin_ratio=1.5)
    b.update_market_prices({"AAPL": 1_000.0})
    # 100 sh @ 1000 = 100k notional → 1.5x = 150k > 100k equity → reject.
    result = _sell(b, "AAPL", 100)
    assert result["status"] == "REJECTED"
    assert b.get_position("AAPL") is None
    # A smaller short within margin is accepted.
    ok = _sell(b, "AAPL", 50)  # 50k notional → 75k margin ≤ 100k
    assert ok["status"] == "FILLED"


def test_short_rejected_when_exceeding_hard_cap():
    b = _broker(allow_short=True, short_margin_ratio=1.0, max_short_notional=5_000.0)
    b.update_market_prices({"AAPL": 100.0})
    # 100 @ 100 = 10k > 5k cap → reject even though margin would allow it.
    assert _sell(b, "AAPL", 100)["status"] == "REJECTED"
    # 40 @ 100 = 4k ≤ 5k cap.
    assert _sell(b, "AAPL", 40)["status"] == "FILLED"


# --- realized P&L on shorts --------------------------------------------------


def test_winning_short_books_positive_realized_pnl():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10 @ 100
    b.update_market_prices({"AAPL": 90.0})
    _buy(b, "AAPL", 10)  # cover @ 90 → profit 10 * (100 - 90)
    assert b.get_position("AAPL") is None
    assert b.get_realized_pnl() == pytest.approx(100.0)
    assert b.get_win_loss() == (1, 0)
    # Cash: +1000 (short) -900 (cover) = +100 over the initial balance.
    assert b.get_balance()["cash"] == pytest.approx(100_100.0)


def test_losing_short_books_negative_realized_pnl():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10 @ 100
    b.update_market_prices({"AAPL": 115.0})
    _buy(b, "AAPL", 10)  # cover @ 115 → loss 10 * (100 - 115)
    assert b.get_realized_pnl() == pytest.approx(-150.0)
    assert b.get_win_loss() == (0, 1)


def test_partial_cover_realizes_proportional_pnl_and_one_event():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # short 10 @ 100
    b.update_market_prices({"AAPL": 90.0})
    _buy(b, "AAPL", 4)  # cover 4 @ 90 → 4 * (100 - 90) = 40
    assert b.get_realized_pnl() == pytest.approx(40.0)
    assert b.get_win_loss() == (1, 0)  # one closing event, not four
    assert b.get_position("AAPL").quantity == -6.0


def test_winning_and_losing_shorts_accumulate_win_loss_counts():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0, "TSLA": 200.0})
    # Winning short on AAPL.
    _sell(b, "AAPL", 10)
    b.update_market_prices({"AAPL": 90.0})
    _buy(b, "AAPL", 10)
    # Losing short on TSLA.
    _sell(b, "TSLA", 5)
    b.update_market_prices({"TSLA": 220.0})
    _buy(b, "TSLA", 5)
    wins, losses = b.get_win_loss()
    assert (wins, losses) == (1, 1)
    # Net realized: +100 (AAPL) + 5*(200-220)=-100 (TSLA) = 0.
    assert b.get_realized_pnl() == pytest.approx(0.0)


def test_short_then_adverse_move_lowers_account_value():
    b = _broker(allow_short=True)
    b.update_market_prices({"AAPL": 100.0})
    _sell(b, "AAPL", 10)  # cash 101k, position -10 @ 100
    # Mark up 10 → the short is underwater; account value falls below 100k.
    val = b.get_account_value({"AAPL": 110.0})
    assert val == pytest.approx(100_000.0 - 100.0)  # 101000 + (-10 * 110)
