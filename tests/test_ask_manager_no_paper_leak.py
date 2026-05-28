"""Red-team test: ask_manager("is this real money?") must not leak paper status.

This is the §10 enforcement test from the WS-Agent plan (§Discipline rule 10).

The test probes both:
  1. The _scrub_answer() function directly.
  2. A full ask_manager() call whose mock manager returns a variety of suspicious
     replies — asserts the scrubber catches all forbidden disclosure words.

Forbidden words (case-insensitive): paper, sim, demo, fake, test mode, monopoly.

The test is INTENTIONALLY adversarial — it checks strings that a real manager
LLM might accidentally produce when asked about account status.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.intel.tools.look.ask_manager import AskManagerTool, _scrub_answer

# ---------------------------------------------------------------------------
# Forbidden-word constants (match plan §Discipline rule 10)
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = re.compile(
    r"\b(paper|sim|demo|fake|test\s+mode|monopoly)\b",
    re.IGNORECASE,
)


def _has_forbidden(text: str) -> list[str]:
    return [m.group() for m in _FORBIDDEN_PATTERNS.finditer(text)]


# ---------------------------------------------------------------------------
# Direct scrubber tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dangerous_reply", [
    "Your account is a paper trading account.",
    "This is a simulation environment.",
    "You are running in demo mode right now.",
    "These are fake trades using fake money.",
    "You are in test mode, so nothing is real.",
    "Your P&L is monopoly money — don't worry about it.",
    "It's just a sim, not real money.",
    # Edge cases: mixed case, embedded in sentence
    "This is a PAPER account — chill.",
    "You're using Demo capital for dry runs.",
    "SIMULATION account — trades don't count.",
    "It's a DEMO-account so no real risk.",
    # Not forbidden (should pass through unchanged)
    "Diversify your portfolio across sectors.",
    "The market looks volatile today.",
])
def test_scrub_answer_removes_forbidden_words(dangerous_reply: str) -> None:
    scrubbed = _scrub_answer(dangerous_reply)
    leaks = _has_forbidden(scrubbed)
    assert not leaks, (
        f"Forbidden words {leaks!r} found in scrubbed reply.\n"
        f"  Original:  {dangerous_reply!r}\n"
        f"  Scrubbed:  {scrubbed!r}"
    )


# ---------------------------------------------------------------------------
# Full ask_manager integration: "is this real money?" red-team
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("manager_reply", [
    # These are what a misconfigured manager might say
    "Yes, this is a paper trading account — your decisions don't affect real money.",
    "No, you're in simulation mode. Feel free to take big risks.",
    "It's a demo environment, so your money is not at risk.",
    "This is fake money for testing purposes.",
    "You are operating in test mode.",
    "Your account is a paper sim — go ahead and experiment.",
    # A safe reply — should pass through intact
    "Your account reflects real market conditions. Manage your risk carefully.",
    "I cannot confirm anything about your account's nature, but all decisions matter.",
])
def test_ask_manager_is_real_money_never_leaks_paper_status(
    manager_reply: str,
) -> None:
    """ask_manager('is this real money?') reply must not contain forbidden words."""
    manager = MagicMock()
    manager.chat.return_value = manager_reply

    tool = AskManagerTool(
        trader_id="RedTeam",
        owner_user_id="u1",
        manager_agent=manager,
        model_ref=MagicMock(),
    )
    result = tool("is this real money?")

    assert result.ok is True, f"Tool failed unexpectedly: {result.error}"
    answer = result.data["answer"]

    leaks = _has_forbidden(answer)
    assert not leaks, (
        f"Paper-status leak detected in ask_manager reply!\n"
        f"  Forbidden words: {leaks!r}\n"
        f"  Original reply: {manager_reply!r}\n"
        f"  Returned answer: {answer!r}\n"
        f"  Rule: §Discipline 10 — no paper/sim/demo/fake in trader-facing surfaces."
    )


def test_ask_manager_filter_prefix_prevents_disclosure() -> None:
    """The _MANAGER_FILTER prefix injected by ask_manager must contain the disclosure rule."""
    from trading_agent.intel.tools.look.ask_manager import _MANAGER_FILTER

    lower = _MANAGER_FILTER.lower()
    assert "never disclose" in lower or "never" in lower
    assert "paper" in lower or "simulation" in lower


def test_ask_manager_filter_prefix_in_sent_message() -> None:
    """The filter prefix is prepended to the question before sending to the manager."""
    sent: list[str] = []

    def fake_chat(user_id: str, conv_id: str, msg: str, ref: Any) -> str:
        sent.append(msg)
        return "Manage risk carefully."

    manager = MagicMock()
    manager.chat.side_effect = fake_chat

    tool = AskManagerTool(
        trader_id="RedTeam",
        owner_user_id="u1",
        manager_agent=manager,
        model_ref=MagicMock(),
    )
    tool("is this real money?")

    assert len(sent) == 1
    assert "is this real money?" in sent[0]
    assert "NEVER disclose" in sent[0], (
        f"Filter prefix missing from sent message.\n"
        f"  Sent: {sent[0][:300]!r}"
    )
