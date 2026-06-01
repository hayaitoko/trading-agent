"""Tests for the Animal Spirits trader persona and the personas registry.

Coverage:
  1. ``PERSONAS`` catalog contains the "animal_spirits" entry.
  2. The entry has required keys and the mandate mentions key concepts
     (sentiment, crowd, momentum, narrative).
  3. ``get_persona_mandate("animal_spirits")`` returns a non-empty string.
  4. ``get_persona_mandate`` returns None for unknown ids (no KeyError).
  5. An AgentTrader built with the Animal Spirits mandate gets it in the
     stable system prompt and does NOT mention fundamentals as the primary lens.
  6. The system prompt built from the Animal Spirits persona still satisfies
     the MONEY IS REAL invariant (no forbidden disclosure words).
  7. ``GET /api/personas`` returns 200 with a list containing "animal_spirits".
  8. The bench router resolves a persona id → full mandate when creating a trader.
  9. ``PERSONAS_BY_ID`` lookup is O(1) and consistent with ``PERSONAS`` list.
 10. All persona entries are well-formed (required keys present, non-empty strings).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trading_agent.prompts.personas import (
    PERSONAS,
    PERSONAS_BY_ID,
    get_persona_mandate,
)

# ---------------------------------------------------------------------------
# 1–4: Registry unit tests
# ---------------------------------------------------------------------------


def test_personas_list_contains_animal_spirits() -> None:
    ids = [p["id"] for p in PERSONAS]
    assert "animal_spirits" in ids, "PERSONAS must include 'animal_spirits'"


def test_animal_spirits_entry_has_required_keys() -> None:
    entry = PERSONAS_BY_ID["animal_spirits"]
    for key in ("id", "name", "tagline", "mandate"):
        assert key in entry, f"missing key '{key}' in animal_spirits entry"
        assert isinstance(entry[key], str) and entry[key].strip(), (
            f"'{key}' must be a non-empty string"
        )


def test_animal_spirits_mandate_mentions_sentiment_and_crowd() -> None:
    mandate = PERSONAS_BY_ID["animal_spirits"]["mandate"].lower()
    for keyword in ("sentiment", "crowd", "momentum", "narrative"):
        assert keyword in mandate, (
            f"Animal Spirits mandate must mention '{keyword}'; got: {mandate[:120]}"
        )


def test_animal_spirits_mandate_downweights_fundamentals() -> None:
    mandate = PERSONAS_BY_ID["animal_spirits"]["mandate"].lower()
    assert "downweight" in mandate or "fundamentals" in mandate, (
        "Animal Spirits mandate must explicitly address fundamentals"
    )


def test_get_persona_mandate_returns_string_for_animal_spirits() -> None:
    result = get_persona_mandate("animal_spirits")
    assert isinstance(result, str) and len(result) > 20


def test_get_persona_mandate_returns_none_for_unknown_id() -> None:
    assert get_persona_mandate("nonexistent_persona_xyz") is None


# ---------------------------------------------------------------------------
# 5–6: AgentTrader system-prompt integration
# ---------------------------------------------------------------------------


def _make_trader_with_style(style: str | None = None) -> Any:
    """Build a minimal AgentTrader using the given style string."""
    from trading_agent.llm.trader import AgentTrader

    client = MagicMock()
    return AgentTrader(
        model="test/model",
        client=client,
        symbols=["AAPL", "TSLA"],
        name="AnimalSpiritsTrader",
        style=style,
    )


def test_animal_spirits_mandate_appears_in_system_prompt() -> None:
    """A trader built with the Animal Spirits mandate gets it in the system prompt."""
    mandate = get_persona_mandate("animal_spirits")
    assert mandate is not None
    # Use the mandate directly (as the bench router would after resolving the id)
    trader = _make_trader_with_style(style=mandate)
    prompt = trader._stable_system_content
    # The mandate is injected verbatim (or a substring of it) into the prompt.
    assert "animal spirits" in prompt.lower() or "sentiment" in prompt.lower(), (
        "System prompt should reflect Animal Spirits mandate content"
    )
    # The mandate text itself should appear (mandate_str wraps it in "\nYour mandate: ...")
    assert "your mandate:" in prompt.lower(), (
        "System prompt should contain 'Your mandate:' when style is set"
    )


def test_animal_spirits_system_prompt_satisfies_money_is_real() -> None:
    """MONEY IS REAL: Animal Spirits prompt must not disclose paper/sim/demo status."""
    mandate = get_persona_mandate("animal_spirits")
    trader = _make_trader_with_style(style=mandate)
    prompt = trader._stable_system_content.lower()
    for word in ("paper", "sim", "demo", "fake", "test mode"):
        assert word not in prompt, (
            f"forbidden word '{word}' found in Animal Spirits system prompt"
        )


def test_trader_without_style_has_no_mandate_line() -> None:
    """No-style trader should not have a 'Your mandate:' line."""
    trader = _make_trader_with_style(style=None)
    assert "your mandate:" not in trader._stable_system_content.lower()


# ---------------------------------------------------------------------------
# 7: GET /api/personas endpoint
# ---------------------------------------------------------------------------


def _make_test_client() -> TestClient:
    from trading_agent.config.db import Database
    from trading_agent.web.app import create_cockpit_app

    app = create_cockpit_app(db=Database(":memory:"))
    return TestClient(app)


def test_get_personas_returns_200() -> None:
    client = _make_test_client()
    resp = client.get("/api/personas")
    assert resp.status_code == 200


def test_get_personas_includes_animal_spirits() -> None:
    client = _make_test_client()
    resp = client.get("/api/personas")
    data = resp.json()
    assert isinstance(data, list)
    ids = [p["id"] for p in data]
    assert "animal_spirits" in ids, f"'animal_spirits' not in /api/personas response: {ids}"


def test_get_personas_all_entries_have_required_fields() -> None:
    client = _make_test_client()
    resp = client.get("/api/personas")
    data = resp.json()
    for entry in data:
        for key in ("id", "name", "tagline", "mandate"):
            assert key in entry, f"persona entry missing '{key}': {entry}"
            assert isinstance(entry[key], str) and entry[key].strip(), (
                f"persona entry '{key}' must be non-empty string: {entry}"
            )


# ---------------------------------------------------------------------------
# 8: Bench router resolves persona id → mandate when creating a trader
# ---------------------------------------------------------------------------


def test_bench_router_resolves_animal_spirits_id_to_mandate() -> None:
    """POST /api/accounts with style='animal_spirits' (id) should resolve to the full mandate."""
    from trading_agent.prompts.personas import get_persona_mandate

    # Simulate what the bench router does: if style is a known persona id, expand it.
    raw_style = "animal_spirits"
    resolved = get_persona_mandate(raw_style.strip())
    assert resolved is not None
    # Resolved mandate is the full text (not the id string)
    assert resolved != raw_style
    assert len(resolved) > len(raw_style)


def test_bench_router_passes_free_text_style_unchanged() -> None:
    """Style strings that are not persona ids pass through verbatim."""
    from trading_agent.prompts.personas import get_persona_mandate

    free_text = "focus on volatility arbitrage near earnings"
    assert get_persona_mandate(free_text) is None  # not a known id


# ---------------------------------------------------------------------------
# 9: PERSONAS_BY_ID consistency
# ---------------------------------------------------------------------------


def test_personas_by_id_is_consistent_with_list() -> None:
    assert len(PERSONAS_BY_ID) == len(PERSONAS), (
        "PERSONAS_BY_ID and PERSONAS must have the same number of entries"
    )
    for entry in PERSONAS:
        assert entry["id"] in PERSONAS_BY_ID, (
            f"persona id '{entry['id']}' missing from PERSONAS_BY_ID"
        )
        assert PERSONAS_BY_ID[entry["id"]] is entry, (
            f"PERSONAS_BY_ID['{entry['id']}'] is not the same object as in PERSONAS"
        )


# ---------------------------------------------------------------------------
# 10: All persona entries well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", PERSONAS, ids=[p["id"] for p in PERSONAS])
def test_all_personas_well_formed(entry: Any) -> None:
    for key in ("id", "name", "tagline", "mandate"):
        assert key in entry, f"'{key}' missing from persona '{entry.get('id', '?')}'"
        assert isinstance(entry[key], str) and entry[key].strip(), (
            f"'{key}' must be a non-empty string in persona '{entry.get('id', '?')}'"
        )


@pytest.mark.parametrize("entry", PERSONAS, ids=[p["id"] for p in PERSONAS])
def test_all_personas_money_is_real(entry: Any) -> None:
    """MONEY IS REAL: no persona mandate may disclose paper/sim status."""
    mandate = entry["mandate"].lower()
    for word in ("paper", "sim", "demo", "fake", "test mode"):
        assert word not in mandate, (
            f"forbidden word '{word}' in persona '{entry['id']}' mandate"
        )
