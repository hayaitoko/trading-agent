"""Smoke tests for the Pull/Digest intelligence-mode selector in the add-trader wizard.

Asserts that:
- cockpit.html contains both intelligence-mode option cards (Pull + Digest)
- The create-trader payload includes the ``digest_mode`` field (contract: POST /api/accounts)
- Honest copy is present: Pull is marked validated, Digest is marked experimental
- ``digest_mode`` defaults to false (Pull) in the WIZ initialiser
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_COCKPIT = _ROOT / "src" / "trading_agent" / "web" / "static" / "cockpit.html"


def _html() -> str:
    return _COCKPIT.read_text()


# ---------------------------------------------------------------------------
# Markup presence
# ---------------------------------------------------------------------------


def test_pull_mode_option_present() -> None:
    """The Pull mode card must be in the wizard markup."""
    html = _html()
    assert "Pull" in html, "wizard must contain Pull mode option"
    # Check for the validated tag copy
    assert "validated" in html.lower(), "Pull mode must be tagged as validated"


def test_digest_mode_option_present() -> None:
    """The Digest mode card must be in the wizard markup."""
    html = _html()
    assert "Digest" in html, "wizard must contain Digest mode option"
    # Check for the experimental tag copy
    assert "experimental" in html.lower(), "Digest mode must be tagged as experimental"


def test_digest_mode_honest_copy() -> None:
    """Helper text must not overclaim digest is better — must note it is experimental."""
    html = _html()
    # Must not assert digest trades better without qualification
    assert "not yet proven to trade better" in html, (
        "Digest copy must honestly state it is not yet proven to trade better"
    )
    # Must flag it as cheaper (honest value prop)
    assert "cheaper" in html.lower(), "Digest copy must state it is cheaper"


def test_pull_mode_honest_copy() -> None:
    """Pull mode must describe its thoroughness and cost trade-off honestly."""
    html = _html()
    assert "tokens" in html.lower(), (
        "Pull mode copy must mention token cost (e.g. 'uses the most tokens')"
    )


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


def test_digest_mode_defaults_to_false() -> None:
    """WIZ initialiser must set digest_mode:false (Pull is the default)."""
    html = _html()
    # The openWizard function must initialise digest_mode to false
    assert "digest_mode:false" in html, (
        "openWizard() must initialise WIZ.digest_mode to false (Pull is default)"
    )


# ---------------------------------------------------------------------------
# API contract wiring
# ---------------------------------------------------------------------------


def test_digest_mode_included_in_create_payload() -> None:
    """POST /api/accounts body must include digest_mode field."""
    html = _html()
    # The api() call in wizCreate must pass digest_mode
    assert "digest_mode:WIZ.digest_mode" in html, (
        "wizCreate must include digest_mode:WIZ.digest_mode in the POST /api/accounts body"
    )


def test_digest_mode_sent_to_correct_endpoint() -> None:
    """The digest_mode field must be sent to POST /api/accounts (not any other route)."""
    html = _html()
    # Find the wizCreate function block; confirm digest_mode is present alongside /api/accounts
    match = re.search(r"async function wizCreate\(\)\{(.+?)\n\}", html, re.S)
    assert match, "wizCreate function must exist"
    fn_body = match.group(1)
    assert "/api/accounts" in fn_body, "wizCreate must POST to /api/accounts"
    assert "digest_mode" in fn_body, "wizCreate must pass digest_mode in its POST body"


# ---------------------------------------------------------------------------
# Confirm-step review shows intelligence mode
# ---------------------------------------------------------------------------


def test_review_step_shows_intelligence_mode() -> None:
    """The confirm/review step (step 3) must display the chosen intelligence mode."""
    html = _html()
    # The review row for intelligence should be rendered
    assert "Intelligence" in html, (
        "step-3 review must include an Intelligence row showing the chosen mode"
    )
    assert "Digest (cheaper" in html or "Pull (default" in html, (
        "review step must show the mode label with its tag (cheaper/default)"
    )


# ---------------------------------------------------------------------------
# JS syntax
# ---------------------------------------------------------------------------


def test_cockpit_js_syntax_valid_after_wizard_changes(tmp_path: Any) -> None:
    """The cockpit <script> block must still parse cleanly after the wizard edits."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for JS syntax check")
    html = _html()
    match = re.search(r"<script>(.*)</script>", html, re.S)
    assert match, "cockpit.html must contain a <script> block"
    js = tmp_path / "cockpit_wizard.js"
    js.write_text(match.group(1))
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [node, "--check", str(js)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"JS syntax error after wizard changes:\n{result.stderr}"
