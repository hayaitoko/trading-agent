"""WS-G1 cockpit-wiring tests.

The served SPA must be the *live-wired* copy (fetch() against the CONTRACTS
routes), not the localStorage-only mock. We assert the static copy exists, is
served at ``/`` by the cockpit app, and carries the fetch-wiring markers; the
``design/`` mock stays untouched as the visual spec. JS behaviour itself is
verified out-of-band via ``node --check``; here we pin the server contract.
"""

from __future__ import annotations

import re
import shutil
import subprocess
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


def test_phase2_agent_surfaces_wired(client: TestClient) -> None:
    # WS-G2: research / manager-chat / notifications+requests / notes / wizard
    # call exactly the CONTRACTS routes (no invented ones besides the flagged
    # bench create), through the same api() helper.
    html = client.get("/").text
    for marker in (
        "/api/research",  # research feed
        "/api/research/run",  # gated, explicit paid trigger
        "runResearch(",
        "loadResearch(",
        "/api/chat",  # manager reply
        "/api/chats",  # saved-chat list/save/delete
        "loadChats(",
        "currentConversationId",  # server-side conversation continuity
        "/api/notifications",
        "/api/notifications/read",
        "loadNotifications(",
        "/api/requests/",  # request allow/decline
        "/api/notes",  # advisor notes get/put
        "loadNotesIn(",
    ):
        assert marker in html, f"missing Phase-2 wiring marker: {marker}"


def test_phase2_keeps_mock_fallbacks(client: TestClient) -> None:
    # Each surface must keep its mock array/seed as the 501/offline fallback so an
    # unfinished WS-C/E/H upstream still renders without errors (same rule as P1).
    html = client.get("/").text
    for marker in ("MGR_SEED", "function seedChats(", "let RESEARCH=", "let MEMORY=", "let NOTIFS="):
        assert marker in html, f"missing mock fallback: {marker}"


def test_cockpit_js_syntax_valid(tmp_path: Any) -> None:
    # JS behaviour is verified out-of-band via `node --check`: pin that the wired
    # SPA's single <script> block parses, so a bad edit can't ship silently.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for JS syntax check")
    html = _STATIC.read_text()
    match = re.search(r"<script>(.*)</script>", html, re.S)
    assert match, "cockpit.html must contain a <script> block"
    js = tmp_path / "cockpit.js"
    js.write_text(match.group(1))
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [node, "--check", str(js)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
