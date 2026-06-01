"""Tests for scripts/bootstrap.py — runs without Docker (no network calls).

Validates:
1. Owner user is created on first run.
2. A 'local' embed endpoint is registered with the correct base_url and model.
3. Per-user settings are written (embed_endpoint_id, vstore, qdrant_url).
4. Running the seeder twice does NOT duplicate the user or the endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the bootstrap script is importable from the repo tree.
_repo = Path(__file__).parent.parent
_src = _repo / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_scripts = _repo / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from bootstrap import run  # type: ignore[import]  # noqa: E402

from trading_agent.config.db import Database  # noqa: E402
from trading_agent.config.endpoints import EndpointRegistry  # noqa: E402
from trading_agent.config.settings_store import SettingsStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Fresh temporary data directory for each test."""
    return tmp_path / "data"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _seed(data_dir: Path, **kwargs: object) -> dict[str, object]:
    """Run bootstrap with test-only overrides (no network)."""
    defaults = dict(
        owner="testuser",
        owner_password="testpass123",
        embed_model="bge-small-en-v1.5",
        embed_dim=384,
        embed_base_url="http://embedder:11434/v1",
        qdrant_url="http://qdrant:6333",
        vstore="qdrant",
        data_dir=str(data_dir),
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return run(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBootstrapFirstRun:
    def test_user_created(self, data_dir: Path) -> None:
        result = _seed(data_dir)
        assert result["user_created"] is True

    def test_endpoint_created(self, data_dir: Path) -> None:
        result = _seed(data_dir)
        assert result["endpoint_created"] is True

    def test_endpoint_base_url_matches(self, data_dir: Path) -> None:
        """Registered endpoint must have the correct base_url."""
        _seed(data_dir)
        db = Database(data_dir / "config.db")
        registry = EndpointRegistry(db)
        user_id = db.query_one("SELECT id FROM users WHERE username = 'testuser'")["id"]
        eps = registry.list(str(user_id))
        local_eps = [e for e in eps if e.type == "local"]
        assert len(local_eps) >= 1
        assert local_eps[0].base_url == "http://embedder:11434/v1"

    def test_endpoint_model_setting(self, data_dir: Path) -> None:
        """embed_model setting must match the model we seeded."""
        _seed(data_dir)
        db = Database(data_dir / "config.db")
        settings = SettingsStore(db)
        user_id = str(db.query_one("SELECT id FROM users WHERE username = 'testuser'")["id"])
        assert settings.get(user_id, "embed_model") == "bge-small-en-v1.5"

    def test_embed_endpoint_id_set(self, data_dir: Path) -> None:
        """embed_endpoint_id setting must be set to the registered endpoint's id."""
        result = _seed(data_dir)
        ep_id = str(result["endpoint_id"])
        db = Database(data_dir / "config.db")
        settings = SettingsStore(db)
        user_id = str(result["user_id"])
        assert settings.get(user_id, "embed_endpoint_id") == ep_id

    def test_vstore_setting(self, data_dir: Path) -> None:
        result = _seed(data_dir)
        db = Database(data_dir / "config.db")
        settings = SettingsStore(db)
        user_id = str(result["user_id"])
        assert settings.get(user_id, "vstore") == "qdrant"

    def test_qdrant_url_setting(self, data_dir: Path) -> None:
        result = _seed(data_dir)
        db = Database(data_dir / "config.db")
        settings = SettingsStore(db)
        user_id = str(result["user_id"])
        assert settings.get(user_id, "qdrant_url") == "http://qdrant:6333"

    def test_embed_dim_setting(self, data_dir: Path) -> None:
        result = _seed(data_dir)
        db = Database(data_dir / "config.db")
        settings = SettingsStore(db)
        user_id = str(result["user_id"])
        assert settings.get(user_id, "embed_dim") == 384


class TestBootstrapIdempotent:
    """Running the seeder twice must not duplicate user or endpoint rows."""

    def test_no_duplicate_user(self, data_dir: Path) -> None:
        _seed(data_dir)
        result2 = _seed(data_dir)
        # Second run must NOT claim it created the user
        assert result2["user_created"] is False
        # Only one user row in the database
        db = Database(data_dir / "config.db")
        rows = db.query("SELECT id FROM users WHERE username = 'testuser'")
        assert len(rows) == 1

    def test_no_duplicate_endpoint(self, data_dir: Path) -> None:
        _seed(data_dir)
        result2 = _seed(data_dir)
        # Second run must NOT claim it created the endpoint
        assert result2["endpoint_created"] is False
        # Only one local endpoint row for this user
        db = Database(data_dir / "config.db")
        user_id = str(db.query_one("SELECT id FROM users WHERE username = 'testuser'")["id"])
        registry = EndpointRegistry(db)
        local_eps = [e for e in registry.list(user_id) if e.type == "local"]
        assert len(local_eps) == 1

    def test_settings_stable_on_second_run(self, data_dir: Path) -> None:
        """Settings must be the same after two runs."""
        result1 = _seed(data_dir)
        result2 = _seed(data_dir)
        assert str(result1["user_id"]) == str(result2["user_id"])
        assert str(result1["endpoint_id"]) == str(result2["endpoint_id"])

    def test_data_dir_created(self, tmp_path: Path) -> None:
        """bootstrap creates DATA_DIR/config.db even if data/ doesn't exist."""
        new_dir = tmp_path / "brand_new" / "data"
        assert not new_dir.exists()
        _seed(new_dir)
        assert (new_dir / "config.db").exists()
