"""WS-G1 cockpit-wiring tests.

The served SPA must be the *live-wired* copy (fetch() against the CONTRACTS
routes), not the localStorage-only mock. We assert the static copy exists, is
served at ``/`` by the cockpit app, and carries the fetch-wiring markers; the
``design/`` mock stays untouched as the visual spec. JS behaviour itself is
verified out-of-band via ``node --check``; here we pin the server contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.web.app import create_cockpit_app

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "src" / "trading_agent" / "web" / "static" / "cockpit.html"
_DESIGN = _ROOT / "design" / "cockpit.html"


@pytest.fixture
def client(tmp_path: Any, monkeypatch: Any) -> TestClient:
    # No env key → signup won't auto-seed, keeping the page deterministic.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return TestClient(create_cockpit_app(Database(tmp_path / "config.db")))


def test_static_cockpit_copied() -> None:
    assert _STATIC.is_file(), "WS-G1 must copy design/cockpit.html into web/static/"


def test_root_serves_cockpit_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Helm" in r.text  # the cockpit brand mark


def test_served_page_is_wired_not_mock(client: TestClient) -> None:
    html = client.get("/").text
    # live wiring is present (frontend plumbing against the CONTRACTS routes)
    for marker in (
        "function api(",       # the fetch helper
        "loadSurface(",        # surface loaders with mock fallback
        "/api/auth/",          # login / signup / logout
        "/api/me",             # session validation on boot
        "/api/settings",       # per-user settings (not localStorage)
        "/api/endpoints",      # server-backed endpoint registry
        "/api/accounts",
        "/api/positions",
        "/api/leaderboard",
        "/api/approvals",
        "/api/risk",
        "initSession(",        # boot resolves a real session
    ):
        assert marker in html, f"missing wiring marker: {marker}"
    # the mock's fake-auth disclaimer is gone now that auth is real
    assert "any credentials work" not in html
    assert "no real auth yet" not in html


def test_endpoint_field_mapping_present(client: TestClient) -> None:
    # WS-G must map the server endpoint shape onto the cockpit's field names.
    html = client.get("/").text
    assert "key_preview" in html and "has_key" in html  # masked-key display
    assert "base_url" in html  # url<->base_url mapping in add/toggle


def test_design_mock_untouched() -> None:
    # The design/ copy stays the localStorage mock — the visual spec, not wired.
    text = _DESIGN.read_text()
    assert "any credentials work" in text
    assert "function api(" not in text
