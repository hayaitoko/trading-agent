import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.notes import (
    NotesError,
    delete_note,
    ensure_default_structure,
    list_tree,
    read_note,
    search_notes,
    write_note,
)
from trading_agent.notes.consolidator import (
    Consolidator,
    ConsolidatorConfig,
)


@pytest.fixture
def notes_dir(tmp_path):
    d = tmp_path / "notes"
    ensure_default_structure(d)
    return d


def test_ensure_default_structure_creates_dirs(tmp_path):
    d = tmp_path / "notes"
    ensure_default_structure(d)
    assert (d / "companies").is_dir()
    assert (d / "sectors").is_dir()
    assert (d / "macro").is_dir()
    assert (d / "general").is_dir()
    assert (d / "general" / "README.md").exists()


def test_ensure_default_structure_idempotent(tmp_path):
    d = tmp_path / "notes"
    ensure_default_structure(d)
    custom = d / "general" / "README.md"
    custom.write_text("custom readme", encoding="utf-8")
    ensure_default_structure(d)
    assert custom.read_text(encoding="utf-8") == "custom readme"


def test_write_and_read_note(notes_dir):
    write_note(notes_dir, "companies/NVDA.md", "# NVDA\n\nNotes here.\n")
    assert read_note(notes_dir, "companies/NVDA.md") == "# NVDA\n\nNotes here.\n"


def test_write_creates_history_snapshot(notes_dir):
    write_note(notes_dir, "companies/NVDA.md", "first version\n")
    write_note(notes_dir, "companies/NVDA.md", "second version\n")
    history = list((notes_dir / ".history").rglob("NVDA.md"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "first version\n"


def test_delete_note_keeps_history(notes_dir):
    write_note(notes_dir, "companies/NVDA.md", "doomed\n")
    delete_note(notes_dir, "companies/NVDA.md")
    assert not (notes_dir / "companies" / "NVDA.md").exists()
    snapshots = list((notes_dir / ".history").rglob("NVDA.md"))
    assert len(snapshots) == 1
    assert snapshots[0].read_text(encoding="utf-8") == "doomed\n"


def test_list_tree_excludes_reserved(notes_dir):
    write_note(notes_dir, "companies/NVDA.md", "x")
    write_note(notes_dir, "general/foo.md", "y")
    root = list_tree(notes_dir)
    names = {c.name for c in root.children}
    assert "companies" in names
    assert "general" in names
    assert ".history" not in names
    assert ".consolidator" not in names


def test_search_notes_substring(notes_dir):
    write_note(notes_dir, "companies/NVDA.md", "AI chip leader.\nGrowing fast.")
    write_note(notes_dir, "companies/AAPL.md", "Hardware giant.\nServices growing.")
    hits = search_notes(notes_dir, "growing")
    assert len(hits) == 2
    assert all("growing" in h["excerpt"].lower() for h in hits)


def test_search_notes_empty_query_returns_empty(notes_dir):
    hits = search_notes(notes_dir, "")
    assert hits == []


def test_path_traversal_blocked(notes_dir):
    with pytest.raises(NotesError):
        write_note(notes_dir, "../escape.md", "no")
    with pytest.raises(NotesError):
        read_note(notes_dir, "../../etc/passwd")


def test_path_must_be_markdown(notes_dir):
    with pytest.raises(NotesError):
        write_note(notes_dir, "companies/NVDA.txt", "no")


def test_path_reserved_dirs_blocked(notes_dir):
    with pytest.raises(NotesError):
        write_note(notes_dir, ".history/sneaky.md", "no")
    with pytest.raises(NotesError):
        write_note(notes_dir, ".consolidator/log.md", "no")


def test_read_nonexistent_raises(notes_dir):
    with pytest.raises(NotesError):
        read_note(notes_dir, "companies/Missing.md")


def fake_caller(responses):
    iterator = iter(responses)
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return next(iterator)

    return call, calls


@pytest.fixture
def consolidator_setup(notes_dir, tmp_path):
    write_note(notes_dir, "companies/NVDA.md", "Just some notes about NVDA.\n")
    config_path = tmp_path / "consolidator_config.json"
    log_path = notes_dir / ".consolidator" / "log.md"
    return notes_dir, config_path, log_path


async def test_consolidator_config_load_default(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
    )
    config = consolidator.load_config()
    assert config.enabled is False
    assert config.interval_minutes == 60


async def test_consolidator_save_then_load(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
    )
    consolidator.save_config(ConsolidatorConfig(
        enabled=True, interval_minutes=30, model="anthropic/claude-haiku-4.5",
    ))
    again = consolidator.load_config()
    assert again.enabled is True
    assert again.interval_minutes == 30
    assert again.model == "anthropic/claude-haiku-4.5"


async def test_consolidator_run_once_applies_edits(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    new_content = (
        "---\ntitle: NVDA\ncreated: 2026-05-20\nupdated: 2026-05-20\n---\n\n"
        "Notes about NVDA. (as of 2026-05-20)\n"
    )
    response = {
        "content": json.dumps({
            "summary": "Added frontmatter and as-of marker to NVDA",
            "edits": [{"path": "companies/NVDA.md", "new_content": new_content}],
        }),
        "tool_calls": None,
    }
    caller, calls = fake_caller([response])
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
        model_caller=caller,
    )
    result = await consolidator.run_once()
    assert result.error is None
    assert len(result.edits) == 1
    assert read_note(notes_dir, "companies/NVDA.md") == new_content
    assert log_path.exists()
    assert "Added frontmatter" in log_path.read_text(encoding="utf-8")
    assert consolidator.status.last_run_at is not None
    assert consolidator.status.edits_last_run == 1
    assert len(calls) == 1


async def test_consolidator_strips_json_fences(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    response = {
        "content": '```json\n{"summary": "ok", "edits": []}\n```',
        "tool_calls": None,
    }
    caller, _ = fake_caller([response])
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
        model_caller=caller,
    )
    result = await consolidator.run_once()
    assert result.error is None
    assert result.edits == []


async def test_consolidator_empty_new_content_deletes(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    write_note(notes_dir, "companies/AAPL.md", "absorbed into NVDA\n")
    response = {
        "content": json.dumps({
            "summary": "Merged AAPL note into NVDA",
            "edits": [
                {"path": "companies/NVDA.md", "new_content": "merged content\n"},
                {"path": "companies/AAPL.md", "new_content": ""},
            ],
        }),
        "tool_calls": None,
    }
    caller, _ = fake_caller([response])
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
        model_caller=caller,
    )
    await consolidator.run_once()
    assert not (notes_dir / "companies" / "AAPL.md").exists()
    snapshots = list((notes_dir / ".history").rglob("AAPL.md"))
    assert len(snapshots) >= 1


async def test_consolidator_invalid_json_records_error(consolidator_setup):
    notes_dir, config_path, log_path = consolidator_setup
    caller, _ = fake_caller([{"content": "not json at all", "tool_calls": None}])
    consolidator = Consolidator(
        notes_dir=notes_dir,
        config_path=config_path,
        log_path=log_path,
        api_key_getter=lambda: "test-key",
        model_caller=caller,
    )
    result = await consolidator.run_once()
    assert result.error is not None
    assert consolidator.status.last_error is not None


@pytest.fixture
def client(tmp_path):
    from decimal import Decimal

    from trading_agent.web import AppState, create_app

    state = AppState(
        accounts_path=tmp_path / "accounts.json",
        secrets_path=tmp_path / "secrets.json",
        quote_fn=lambda t: Decimal("0"),
    )
    app = create_app(
        state,
        chat_history_path=tmp_path / "chat.json",
        notes_dir=tmp_path / "notes",
        consolidator_config_path=tmp_path / "cc.json",
        start_consolidator=False,
    )
    return TestClient(app)


def test_route_notes_page_renders(client):
    response = client.get("/notes/")
    assert response.status_code == 200
    assert "Notes" in response.text
    assert "consolidator" in response.text


def test_route_notes_tree_returns_default_structure(client):
    response = client.get("/notes/api/tree")
    assert response.status_code == 200
    tree = response.json()
    child_names = {c["name"] for c in tree["children"]}
    assert {"companies", "sectors", "macro", "general"}.issubset(child_names)


def test_route_notes_write_then_read(client):
    write_resp = client.put("/notes/api/write", json={
        "path": "companies/NVDA.md",
        "content": "# NVDA\n",
    })
    assert write_resp.status_code == 200
    read_resp = client.get("/notes/api/read?path=companies/NVDA.md")
    assert read_resp.status_code == 200
    assert read_resp.json()["content"] == "# NVDA\n"


def test_route_notes_delete(client):
    client.put("/notes/api/write", json={
        "path": "general/temp.md", "content": "x",
    })
    response = client.request(
        "DELETE", "/notes/api/delete",
        json={"path": "general/temp.md"},
    )
    assert response.status_code == 200
    follow_up = client.get("/notes/api/read?path=general/temp.md")
    assert follow_up.status_code == 404


def test_route_notes_traversal_blocked(client):
    response = client.put("/notes/api/write", json={
        "path": "../escape.md", "content": "no",
    })
    assert response.status_code == 400


def test_route_consolidator_config_round_trip(client):
    put = client.put("/notes/api/consolidator/config", json={
        "enabled": True,
        "interval_minutes": 15,
        "model": "anthropic/claude-haiku-4.5",
    })
    assert put.status_code == 200

    get = client.get("/notes/api/consolidator")
    data = get.json()
    assert data["config"]["enabled"] is True
    assert data["config"]["interval_minutes"] == 15
    assert data["config"]["model"] == "anthropic/claude-haiku-4.5"


def test_route_consolidator_run_records_error_when_no_api_key(client):
    response = client.post("/notes/api/consolidator/run")
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is not None
    assert "OPENROUTER_API_KEY" in data["error"] or "API" in data["error"]


def test_route_consolidator_run_with_mocked_call(client, tmp_path):
    new_md = "---\ntitle: NVDA\ncreated: 2026-05-20\nupdated: 2026-05-20\n---\nContent\n"

    async def fake(**kwargs):
        return {
            "content": json.dumps({
                "summary": "demo run",
                "edits": [{"path": "companies/NVDA.md", "new_content": new_md}],
            }),
            "tool_calls": None,
        }

    client.app.state.consolidator._model_caller = fake
    client.app.state.app_state.secrets["openrouter_api_key"] = "sk-test"
    client.put("/notes/api/write", json={"path": "companies/NVDA.md", "content": "old"})

    response = client.post("/notes/api/consolidator/run")
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is None
    assert data["edits"] == 1

    read_resp = client.get("/notes/api/read?path=companies/NVDA.md")
    assert read_resp.json()["content"] == new_md


def test_notes_appears_in_nav(client):
    response = client.get("/")
    assert "Notes" in response.text
    assert 'href="/notes/"' in response.text


def test_chat_tools_include_notes(client):
    from trading_agent.chat.tools import TOOL_SCHEMAS
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert {"list_notes", "read_note", "search_notes"}.issubset(names)


async def test_chat_tool_list_notes_returns_paths(tmp_path):
    from decimal import Decimal

    from trading_agent.chat.tools import execute
    from trading_agent.web import AppState
    from trading_agent.web.app import create_app  # noqa: F401, triggers ensure_default

    state = AppState(
        accounts_path=tmp_path / "accounts.json",
        secrets_path=tmp_path / "secrets.json",
        quote_fn=lambda t: Decimal("0"),
    )
    state.notes_dir = tmp_path / "notes"
    ensure_default_structure(state.notes_dir)
    write_note(state.notes_dir, "companies/NVDA.md", "x")

    result = await execute(state, "list_notes", {})
    paths = json.loads(result)
    assert "companies/NVDA.md" in paths
    assert "general/README.md" in paths


async def test_chat_tool_read_note(tmp_path):
    from decimal import Decimal

    from trading_agent.chat.tools import execute
    from trading_agent.web import AppState

    state = AppState(
        accounts_path=tmp_path / "accounts.json",
        secrets_path=tmp_path / "secrets.json",
        quote_fn=lambda t: Decimal("0"),
    )
    state.notes_dir = tmp_path / "notes"
    ensure_default_structure(state.notes_dir)
    write_note(state.notes_dir, "companies/NVDA.md", "hello")

    result = await execute(state, "read_note", {"path": "companies/NVDA.md"})
    data = json.loads(result)
    assert data["content"] == "hello"
    assert data["path"] == "companies/NVDA.md"


def _used_path_arg(tmp_path: Path):
    # Sanity check the pytest fixture exists. Keeps Path import warm.
    assert isinstance(tmp_path, Path)
