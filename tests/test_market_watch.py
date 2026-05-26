"""Tests for the market-move threshold watcher."""

from trading_agent.web.market_watch import MarketMoveWatcher


def test_no_move_inside_threshold() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    assert w.observe("X", 100.0) is None  # first sets reference
    assert w.observe("X", 101.0) is None  # +1% < 2%
    assert w.recent() == []


def test_fires_on_crossing_band_up() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    w.observe("X", 100.0)
    move = w.observe("X", 103.0)  # +3% -> band 1
    assert move is not None
    assert move.direction == "up"
    assert round(move.pct_change, 4) == 0.03
    assert len(w.recent()) == 1


def test_fires_on_crossing_band_down() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    w.observe("X", 100.0)
    move = w.observe("X", 96.0)  # -4% -> band -2
    assert move is not None
    assert move.direction == "down"


def test_each_new_band_fires_once() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    w.observe("X", 100.0)
    assert w.observe("X", 102.5) is not None  # band 1
    assert w.observe("X", 103.0) is None  # still band 1 — no repeat
    assert w.observe("X", 104.5) is not None  # band 2
    assert len(w.recent()) == 2


def test_reversal_fires() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    w.observe("X", 100.0)
    assert w.observe("X", 103.0) is not None  # band 1
    assert w.observe("X", 96.0) is not None  # crossed to band -2


def test_handle_bar_extracts_close() -> None:
    w = MarketMoveWatcher(threshold_pct=1.0)
    w.handle_bar({"symbol": "Y", "close": 50.0})
    move = w.handle_bar({"symbol": "Y", "close": 51.0})  # +2% -> band 2
    assert move is not None and move.symbol == "Y"


def test_ignores_bad_input() -> None:
    w = MarketMoveWatcher(threshold_pct=1.0)
    assert w.observe(None, 100.0) is None
    assert w.observe("X", None) is None
    assert w.observe("X", 0.0) is None
    assert w.observe("X", -5.0) is None


def test_recent_is_newest_first_and_capped() -> None:
    w = MarketMoveWatcher(threshold_pct=1.0, max_events=3)
    w.observe("X", 100.0)
    for px in (102.0, 104.0, 106.0, 108.0):  # successive new bands
        w.observe("X", px)
    recent = w.recent()
    assert len(recent) == 3  # capped
    assert recent[0].current_price == 108.0  # newest first


def test_per_symbol_references_independent() -> None:
    w = MarketMoveWatcher(threshold_pct=2.0)
    w.observe("A", 100.0)
    w.observe("B", 200.0)
    assert w.observe("A", 103.0) is not None
    assert w.observe("B", 202.0) is None  # +1% on its own reference
