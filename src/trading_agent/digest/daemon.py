"""DigestDaemon — background analyst-digest compile cadence + event-wake.

Runs as a daemon thread alongside the research daemon.  On each cycle:
  1. For each user that has digest-mode traders, compile a fresh digest for each
     unique symbol universe.
  2. If the compiled digest is flagged ``material_flag=True``, fire a
     research-bombshell event-wake so the relevant traders get an off-cadence
     turn immediately (reuses the market-move / attention-queue wake path in the
     bench controller).

Default cadence: 15 minutes (much shorter than the hourly research cadence since
the compiler reads from local stores — no external fetches, just an LLM call).
Between cadence ticks the daemon checks for staleness to decide whether to
compile immediately after a research pass lands new briefs.

Event-wake contract:
  ``bombshell_callback`` is an optional ``callable(universe_key: str) -> None``
  injected by the bench controller (analogous to the market-move callback).  When
  a material event is detected the daemon calls it; the controller then fires an
  off-cadence turn for every trader whose universe matches.  When the callback is
  None the material flag is still persisted (for the next regular turn to see).
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.db import Database
    from ..config.endpoints import EndpointRegistry
    from ..config.settings_store import SettingsStore
    from .compiler import DigestCompiler
    from .store import DigestStore

logger = logging.getLogger(__name__)

# How often (seconds) to run a full compile pass for all users.
DEFAULT_CADENCE_SECONDS: int = 900  # 15 minutes

# Only recompile if the existing digest is older than this (seconds). Prevents
# spurious re-compiles when the cadence fires but nothing has changed.
MIN_AGE_BEFORE_RECOMPILE: int = 300  # 5 minutes


def _users_with_digests(db: Database) -> list[str]:
    """Return user_ids that have at least one enabled ingest source (proxy for active users)."""
    try:
        rows = db.query("SELECT DISTINCT user_id FROM sources WHERE enabled = 1")
        return [r["user_id"] for r in rows]
    except Exception:
        return []


class DigestDaemon:
    """Background daemon: periodic analyst-digest compilation + event-wake.

    Parameters
    ----------
    db:
        Shared :class:`~trading_agent.config.db.Database`.
    digest_store:
        :class:`~.store.DigestStore` for reads/writes.
    compiler:
        :class:`~.compiler.DigestCompiler` to invoke on each cycle.
    registry:
        :class:`~trading_agent.config.endpoints.EndpointRegistry`.
    settings:
        :class:`~trading_agent.config.settings_store.SettingsStore`.
    cadence_seconds:
        Seconds between compile cycles.
    bombshell_callback:
        Optional callable invoked with the ``universe_key`` string when a
        material event is detected.  Injected by BenchController.
    """

    def __init__(
        self,
        db: Database,
        digest_store: DigestStore,
        compiler: DigestCompiler,
        registry: EndpointRegistry,
        settings: SettingsStore,
        *,
        cadence_seconds: int = DEFAULT_CADENCE_SECONDS,
        bombshell_callback: Any = None,
    ) -> None:
        self._db = db
        self._digest_store = digest_store
        self._compiler = compiler
        self._registry = registry
        self._settings = settings
        self.cadence_seconds = cadence_seconds
        self._bombshell_callback = bombshell_callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Track which universe_keys we last compiled to avoid hammering.
        self._last_compile: dict[str, float] = {}

    def start(self) -> None:
        """Start the daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="digest-daemon", daemon=True
        )
        self._thread.start()
        logger.info("digest_daemon: started (cadence=%ds)", self.cadence_seconds)

    def stop(self) -> None:
        """Signal the daemon to stop and return immediately."""
        self._stop.set()

    def set_bombshell_callback(self, callback: Any) -> None:
        """Wire (or replace) the event-wake callback at runtime."""
        self._bombshell_callback = callback

    def compile_now(self, user_id: str, symbols: list[str]) -> None:
        """Force an immediate recompile for a specific user + universe.

        Called by the research daemon's event hook and by the cockpit dev
        endpoint.  Silently no-ops when cost gate blocks or model is absent.
        """
        self._compile_for_user_symbols(user_id, symbols)

    # ------------------------------------------------------------------
    # Main loop

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._cycle()
            except Exception:
                logger.exception("digest_daemon: cycle-level failure")
            self._stop.wait(self.cadence_seconds)

    def _cycle(self) -> None:
        users = _users_with_digests(self._db)
        for user_id in users:
            self._compile_for_user(user_id)

    # ------------------------------------------------------------------
    # Per-user compile

    def _compile_for_user(self, user_id: str) -> None:
        """Compile digests for all tracked universes for a single user."""
        universes = self._universes_for_user(user_id)
        for symbols in universes:
            self._compile_for_user_symbols(user_id, symbols)

    def _universes_for_user(self, user_id: str) -> list[list[str]]:
        """Infer which symbol universes this user has active traders for.

        Reads from ``analyst_digests`` to find previously stored universes
        (set by the bench controller when it wires digest mode) and also
        reads ingest source tickers as a fallback.  Returns deduplicated
        list of symbol lists.
        """

        seen: set[str] = set()
        universes: list[list[str]] = []

        # 1. Already-stored digest universe keys (written by bench controller).
        try:
            rows = self._db.query(
                "SELECT DISTINCT universe_key FROM analyst_digests WHERE user_id = ?",
                (user_id,),
            )
            for r in rows:
                uk = str(r["universe_key"])
                if uk not in seen:
                    seen.add(uk)
                    symbols = [s for s in uk.split(",") if s]
                    if symbols:
                        universes.append(symbols)
        except Exception:
            pass

        return universes

    def _compile_for_user_symbols(self, user_id: str, symbols: list[str]) -> None:
        """Compile (or skip if fresh) a digest for one user + universe."""
        from .store import universe_key as _uk

        uk = _uk(symbols)
        now = datetime.now(UTC).timestamp()

        # Skip if compiled recently enough.
        last = self._last_compile.get(uk, 0.0)
        if (now - last) < MIN_AGE_BEFORE_RECOMPILE:
            return

        # Also check persisted staleness to avoid re-running after a recent
        # successful compile from another path.
        existing = self._digest_store.get_by_key(user_id, uk)
        if existing is not None and not existing.is_stale(MIN_AGE_BEFORE_RECOMPILE):
            self._last_compile[uk] = now
            return

        ref = self._resolve_ref(user_id)
        if ref is None:
            logger.debug("digest_daemon: no model ref for user=%s, skipping", user_id)
            return

        try:
            digest = self._compiler.compile(user_id, symbols, ref)
        except Exception:
            logger.exception("digest_daemon: compile failed for user=%s uk=%s", user_id, uk)
            return

        self._last_compile[uk] = now

        if digest is not None and digest.material_flag:
            self._fire_bombshell(uk, user_id)

    def _resolve_ref(self, user_id: str) -> Any | None:
        """Resolve a cheap model ref for compilation."""
        try:
            from ..manager.agent import resolve_reflection_ref
            return resolve_reflection_ref(self._settings, self._registry, user_id)
        except Exception:
            return None

    def _fire_bombshell(self, uk: str, user_id: str) -> None:
        """Invoke the bombshell callback when a material event is detected."""
        if self._bombshell_callback is None:
            return
        logger.info(
            "digest_daemon: MATERIAL event detected for user=%s uk=%s — firing bombshell wake",
            user_id,
            uk,
        )
        try:
            self._bombshell_callback(uk)
        except Exception:
            logger.exception("digest_daemon: bombshell callback raised")


def build_digest_daemon(
    app_state: Any,
    *,
    cadence_seconds: int = DEFAULT_CADENCE_SECONDS,
) -> DigestDaemon | None:
    """Build a :class:`DigestDaemon` from ``app.state`` attributes.

    Returns ``None`` when the required state is unavailable.  The caller can
    safely ignore a None return — the cockpit runs without the digest tier.
    """
    db = getattr(app_state, "db", None)
    endpoints = getattr(app_state, "endpoints", None)
    settings = getattr(app_state, "settings", None)
    digest_store = getattr(app_state, "digest_store", None)
    research_store = getattr(app_state, "research", None)

    if db is None or endpoints is None or settings is None or digest_store is None:
        logger.info("digest_daemon: prerequisites missing; digest tier disabled")
        return None

    try:
        from .compiler import DigestCompiler

        compiler = DigestCompiler(
            digest_store=digest_store,
            research_store=research_store,
            registry=endpoints,
            settings=settings,
            db=db,
        )
        return DigestDaemon(
            db=db,
            digest_store=digest_store,
            compiler=compiler,
            registry=endpoints,
            settings=settings,
            cadence_seconds=cadence_seconds,
        )
    except Exception:
        logger.exception("digest_daemon: failed to build; digest tier disabled")
        return None
