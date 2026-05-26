"""IngestStore: dedup, drain cursor, per-user isolation. No network."""

from __future__ import annotations

import time

import pytest

from trading_agent.config.db import Database
from trading_agent.ingest.fetchers.base import RawItem
from trading_agent.ingest.store import IngestStore


@pytest.fixture
def store(tmp_path) -> IngestStore:
    return IngestStore(Database(tmp_path / "config.db"))


def _item(source_id: str, url: str, ticker: str | None = None) -> RawItem:
    return RawItem(source_id=source_id, text=f"text for {url}", url=url, ts="2023-11-15T12:00:00+00:00", ticker=ticker)


def test_append_returns_new_row_count(store: IngestStore) -> None:
    n = store.append("u1", [_item("s1", "http://a"), _item("s1", "http://b")])
    assert n == 2
    assert store.count("u1") == 2


def test_append_dedups_by_source_and_url(store: IngestStore) -> None:
    store.append("u1", [_item("s1", "http://a")])
    # same (source_id, url) again across a second cycle -> ignored
    written = store.append("u1", [_item("s1", "http://a"), _item("s1", "http://c")])
    assert written == 1
    assert store.count("u1") == 2


def test_dedup_is_scoped_per_source(store: IngestStore) -> None:
    # same url from a different source_id is a distinct item
    store.append("u1", [_item("s1", "http://shared")])
    written = store.append("u1", [_item("s2", "http://shared")])
    assert written == 1
    assert store.count("u1") == 2


def test_append_empty_is_noop(store: IngestStore) -> None:
    assert store.append("u1", []) == 0
    assert store.count("u1") == 0


def test_drain_returns_everything_from_zero(store: IngestStore) -> None:
    store.append("u1", [_item("s1", "http://a", ticker="AAPL"), _item("s1", "http://b")])
    items = store.drain("u1")
    assert {it.url for it in items} == {"http://a", "http://b"}
    assert items[0].ticker == "AAPL"  # round-trips, including None for the second


def test_drain_respects_since_cursor(store: IngestStore) -> None:
    store.append("u1", [_item("s1", "http://old")])
    watermark = store.latest_fetched_at("u1")
    time.sleep(0.005)  # ensure a strictly later fetched_at
    store.append("u1", [_item("s1", "http://new")])

    fresh = store.drain("u1", since=watermark)
    assert [it.url for it in fresh] == ["http://new"]


def test_latest_fetched_at_empty_is_zero(store: IngestStore) -> None:
    assert store.latest_fetched_at("nobody") == 0.0


def test_per_user_isolation(store: IngestStore) -> None:
    store.append("u1", [_item("s1", "http://a")])
    store.append("u2", [_item("s1", "http://a")])  # same item, different user
    assert store.count("u1") == 1
    assert store.count("u2") == 1
    assert [it.url for it in store.drain("u2")] == ["http://a"]
    assert store.latest_fetched_at("u3") == 0.0
