"""First-boot idempotent seeder for the trading-agent Docker stack.

Driven by environment variables:
  OWNER             — username for the trading-agent owner account (required)
  OWNER_PASSWORD    — password (default: random dev token)
  EMBED_MODEL       — Ollama model name (default: mxbai-embed-large)
  EMBED_DIM         — embedding dimension (default: 1024)
  EMBED_BASE_URL    — base URL of the local embedder (default: http://embedder:11434/v1)
  QDRANT_URL        — Qdrant service URL (default: http://qdrant:6333)
  VSTORE            — vector store backend: "qdrant" | "sqlite-vec" (default: qdrant)
  DATA_DIR          — path to SQLite data directory (default: /app/data)

Safe to run on every boot — creates the user / endpoint only when absent.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve DATA_DIR and add src/ to the path so the package is importable
# without being installed (the Docker image installs it, but tests run from
# the repo tree with PYTHONPATH set).
# ---------------------------------------------------------------------------
_here = Path(__file__).parent.parent  # repo root (deploy-stack/)
_src = _here / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from trading_agent.config.db import Database  # noqa: E402
from trading_agent.config.endpoints import EndpointRegistry  # noqa: E402
from trading_agent.config.settings_store import SettingsStore  # noqa: E402
from trading_agent.config.users import AuthError, create_user  # noqa: E402


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default or "")
    return val.strip()


def run(  # noqa: PLR0913
    *,
    owner: str | None = None,
    owner_password: str | None = None,
    embed_model: str | None = None,
    embed_dim: int | None = None,
    embed_base_url: str | None = None,
    qdrant_url: str | None = None,
    vstore: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, object]:
    """Seed config.db with owner user + embed endpoint + settings.

    All arguments fall back to environment variables and then to built-in
    defaults.  Returns a dict describing what was seeded (for tests and logs).
    """
    owner = (owner or _env("OWNER", "admin")).strip()
    if not owner:
        print("[bootstrap] ERROR: OWNER must be set", flush=True)
        sys.exit(1)

    password = (owner_password or _env("OWNER_PASSWORD") or "").strip()
    if not password:
        password = secrets.token_urlsafe(20)
        print(f"[bootstrap] OWNER_PASSWORD not set — generated dev password: {password}", flush=True)

    model = embed_model or _env("EMBED_MODEL", "mxbai-embed-large")
    dim = embed_dim if embed_dim is not None else int(_env("EMBED_DIM", "1024"))
    base_url = embed_base_url or _env("EMBED_BASE_URL", "http://embedder:11434/v1")
    q_url = qdrant_url or _env("QDRANT_URL", "http://qdrant:6333")
    vs = vstore or _env("VSTORE", "qdrant")
    d_dir = Path(data_dir or _env("DATA_DIR", "/app/data"))
    d_dir.mkdir(parents=True, exist_ok=True)

    db_path = d_dir / "config.db"
    db = Database(db_path)
    registry = EndpointRegistry(db)
    settings = SettingsStore(db)

    result: dict[str, object] = {
        "data_dir": str(d_dir),
        "db_path": str(db_path),
        "user_created": False,
        "endpoint_created": False,
        "settings_updated": False,
    }

    # ── 1. Create owner user if absent ──────────────────────────────────────
    existing = db.query_one("SELECT id, username FROM users WHERE username = ?", (owner,))
    if existing is not None:
        user_id: str = str(existing["id"])
        print(f"[bootstrap] user already exists: {owner!r} (id={user_id})", flush=True)
    else:
        try:
            user = create_user(db, owner, password)
            user_id = user.id
            result["user_created"] = True
            print(f"[bootstrap] created user: {owner!r} (id={user_id})", flush=True)
        except AuthError as exc:
            # Race condition safety — another process created it between our
            # SELECT and the INSERT.
            row = db.query_one("SELECT id FROM users WHERE username = ?", (owner,))
            if row is None:
                raise RuntimeError(f"create_user failed unexpectedly: {exc}") from exc
            user_id = str(row["id"])
            print(f"[bootstrap] user race — resolved: {owner!r} (id={user_id})", flush=True)

    result["user_id"] = user_id

    # ── 2. Register local embed endpoint if absent ───────────────────────────
    # Idempotent check: look for an existing local endpoint with the same
    # base_url so re-runs don't stack up duplicate rows.
    ep_name = "Ollama Embedder"
    existing_eps = registry.list(user_id)
    existing_ep = next(
        (e for e in existing_eps if e.type == "local" and e.base_url == base_url),
        None,
    )
    if existing_ep is not None:
        ep_id = existing_ep.id
        print(
            f"[bootstrap] embed endpoint already registered: {ep_name!r} (id={ep_id})",
            flush=True,
        )
    else:
        ep = registry.add(
            user_id,
            type="local",
            name=ep_name,
            base_url=base_url,
            api_key="",
            enabled=True,
        )
        ep_id = ep.id
        result["endpoint_created"] = True
        print(
            f"[bootstrap] registered embed endpoint: {ep_name!r} "
            f"base_url={base_url!r} model={model!r} (id={ep_id})",
            flush=True,
        )

    result["endpoint_id"] = ep_id

    # ── 3. Write per-user settings ────────────────────────────────────────────
    settings.set(user_id, "embed_endpoint_id", ep_id)
    settings.set(user_id, "embed_model", model)
    settings.set(user_id, "embed_dim", dim)
    settings.set(user_id, "vstore", vs)
    # The qdrant vector store reads the URL from the "qdrant_url" setting key
    # (see make_vector_store / QdrantVectorStore: `url=` kwarg, keyed by
    # memory/store.py which calls make_vector_store(vstore, url=...) using the
    # setting "qdrant_url").
    settings.set(user_id, "qdrant_url", q_url)
    result["settings_updated"] = True

    print(
        f"[bootstrap] settings: embed_endpoint_id={ep_id!r} embed_model={model!r} "
        f"embed_dim={dim} vstore={vs!r} qdrant_url={q_url!r}",
        flush=True,
    )

    # Seed default OpenRouter endpoint from env (idempotent, no-op if already present)
    registry.seed_defaults(user_id)

    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
