"""Tests for MessageBus and CsvReplayFeed."""

from __future__ import annotations

import csv

import pytest

from trading_agent.data_feed import MessageBus
from trading_agent.feeds.csv_replay import CsvReplayFeed, synthetic_mean_reverting_bars

# --- MessageBus --------------------------------------------------------------


def test_subscribe_publish_unsubscribe_roundtrip():
    bus = MessageBus()
    received = []
    bus.subscribe("topic.a", received.append)
    bus.publish("topic.a", {"v": 1})
    bus.publish("topic.a", {"v": 2})
    assert received == [{"v": 1}, {"v": 2}]

    bus.unsubscribe("topic.a", received.append)
    bus.publish("topic.a", {"v": 3})
    assert received == [{"v": 1}, {"v": 2}]


def test_publish_no_subscribers_is_noop():
    bus = MessageBus()
    bus.publish("nobody", "hi")  # should not raise


def test_multiple_subscribers_all_receive():
    bus = MessageBus()
    a, b = [], []
    bus.subscribe("topic.x", a.append)
    bus.subscribe("topic.x", b.append)
    bus.publish("topic.x", "msg")
    assert a == ["msg"]
    assert b == ["msg"]


def test_handler_exception_does_not_break_other_handlers():
    bus = MessageBus()
    other = []

    def bad(_msg):
        raise RuntimeError("boom")

    bus.subscribe("topic.y", bad)
    bus.subscribe("topic.y", other.append)
    bus.publish("topic.y", "msg")
    # Other handler still gets it; publish() swallows the bad handler's error.
    assert other == ["msg"]


def test_unsubscribe_unknown_topic_is_noop():
    bus = MessageBus()
    bus.unsubscribe("never-subscribed", lambda m: None)  # should not raise


# --- CsvReplayFeed -----------------------------------------------------------


def test_csv_replay_publishes_inline_bars():
    bus = MessageBus()
    received = []
    bus.subscribe("bar.AAPL", received.append)
    bars = [
        {"symbol": "AAPL", "close": 100.0},
        {"symbol": "AAPL", "close": 101.0},
        {"symbol": "TSLA", "close": 200.0},
    ]
    feed = CsvReplayFeed(message_bus=bus, bars=bars)
    published = feed.replay()
    assert published == 3
    # Only AAPL bars were routed to the subscriber.
    assert len(received) == 2
    assert received[0]["close"] == 100.0
    assert received[1]["close"] == 101.0


def test_csv_replay_subscriptions_filter_publish():
    import asyncio

    bus = MessageBus()
    feed = CsvReplayFeed(
        message_bus=bus,
        bars=[{"symbol": "AAPL", "close": 1.0}, {"symbol": "TSLA", "close": 2.0}],
    )
    # Tell the feed it only cares about AAPL.
    asyncio.run(feed.subscribe_symbols(["AAPL"]))
    received = []
    bus.subscribe("bar.AAPL", received.append)
    bus.subscribe("bar.TSLA", received.append)
    count = feed.replay()
    assert count == 1
    assert received == [{"symbol": "AAPL", "close": 1.0}]


def test_csv_replay_reads_csv_file(tmp_path):
    path = tmp_path / "bars.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-01-01T00:00:00",
                "symbol": "AAPL",
                "open": "100",
                "high": "101",
                "low": "99",
                "close": "100.5",
                "volume": "1000",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-01-01T00:01:00",
                "symbol": "AAPL",
                "open": "100.5",
                "high": "102",
                "low": "100",
                "close": "101.0",
                "volume": "1100",
            }
        )

    bus = MessageBus()
    received = []
    bus.subscribe("bar.AAPL", received.append)
    feed = CsvReplayFeed(message_bus=bus, csv_path=path)
    count = feed.replay()
    assert count == 2
    assert len(received) == 2
    # Numeric coercion happened during read.
    assert received[0]["close"] == 100.5
    assert received[0]["volume"] == 1000.0


def test_csv_replay_requires_exactly_one_source():
    bus = MessageBus()
    with pytest.raises(ValueError):
        CsvReplayFeed(message_bus=bus)
    with pytest.raises(ValueError):
        CsvReplayFeed(message_bus=bus, bars=[], csv_path="x.csv")


def test_csv_replay_default_symbol(tmp_path):
    bus = MessageBus()
    received = []
    bus.subscribe("bar.FOO", received.append)
    feed = CsvReplayFeed(
        message_bus=bus, bars=[{"close": 1.0}, {"close": 2.0}], default_symbol="FOO"
    )
    feed.replay()
    assert len(received) == 2
    assert received[0]["symbol"] == "FOO"


# --- synthetic_mean_reverting_bars -------------------------------------------


def test_synthetic_bars_count_and_keys():
    bars = synthetic_mean_reverting_bars("SYNTH", n=50)
    assert len(bars) == 50
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    assert required.issubset(bars[0].keys())
    assert all(b["symbol"] == "SYNTH" for b in bars)


def test_synthetic_bars_have_consistent_ohlc():
    bars = synthetic_mean_reverting_bars("SYNTH", n=30)
    for b in bars:
        assert b["low"] <= b["open"] <= b["high"]
        assert b["low"] <= b["close"] <= b["high"]
