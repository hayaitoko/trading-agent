"""Runtime control surface for the bench: roster, cadence, and the decision loop.

The web layer drives this — add/remove models, set the cadence (the UI dropdown),
start/stop the autonomous decision loop, and fetch the model menu from OpenRouter.
The loop runs on its own thread, calling :meth:`Bench.run_decisions` every
``cadence_seconds``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from ..llm.trader import LLMTrader

if TYPE_CHECKING:
    from ..config.db import Database
    from ..data.history import HistoryService
    from ..llm.openrouter import OpenRouterClient
    from .bench import Bench

# Known-good slugs surfaced first in the model menu (verified against OpenRouter).
FEATURED_MODELS = [
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.5-flash",
    "z-ai/glm-5.1",
    "x-ai/grok-4.3",
]

CADENCE_OPTIONS = [
    {"label": "Every poll (fast, $$$)", "seconds": 5},
    {"label": "1 min", "seconds": 60},
    {"label": "5 min (recommended)", "seconds": 300},
    {"label": "15 min", "seconds": 900},
    {"label": "30 min", "seconds": 1800},
]

MIN_CADENCE = 5


class BenchController:
    def __init__(
        self,
        bench: Bench,
        client: OpenRouterClient,
        *,
        symbols: list[str],
        cadence_seconds: int = 300,
        history: HistoryService | None = None,
        research: Any = None,
        memory: Any = None,
        owner_user_id: str | None = None,
        db: Database | None = None,
    ) -> None:
        self.bench = bench
        self.client = client
        self.symbols = list(symbols)
        self.cadence_seconds = max(MIN_CADENCE, cadence_seconds)
        # WS-A intelligence wiring threaded into every LLMTrader (all optional;
        # None → that trader degrades exactly as the bench did before WS-A).
        self.history = history
        self.research = research
        self.memory = memory
        self._owner_explicit = owner_user_id
        self.db = db
        self._cached_owner_id: str | None = owner_user_id
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._models_cache: dict[str, Any] | None = None

    @property
    def owner_id(self) -> str | None:
        """The bound owner, resolved lazily and cached once non-None.

        A fresh box has zero users until someone signs up, so resolving once at
        construction would freeze None forever; we re-resolve each access until
        it sticks (explicit value → env → single-user fallback, see
        :func:`config.users.resolve_owner_user_id`)."""
        if self._cached_owner_id is None and self.db is not None:
            from ..config.users import resolve_owner_user_id

            self._cached_owner_id = resolve_owner_user_id(self.db, explicit=self._owner_explicit)
        return self._cached_owner_id

    # --- Roster -------------------------------------------------------------

    def add_model(
        self,
        model: str,
        name: str | None = None,
        *,
        cash: float | None = None,
        style: str | None = None,
    ) -> str:
        trader = LLMTrader(
            model,
            self.client,
            symbols=self.symbols,
            name=name or model,
            style=style,
            history=self.history,
            research=self.research,
            memory=self.memory,
            owner_user_id=self.owner_id,
        )
        self.bench.add_competitor(
            trader.name, trader, initial_balance=cash, style=style
        )
        return trader.name

    def remove(self, name: str) -> None:
        self.bench.remove_competitor(name)

    # --- Cadence + loop -----------------------------------------------------

    def set_cadence(self, seconds: int) -> None:
        self.cadence_seconds = max(MIN_CADENCE, int(seconds))

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop.clear()
            self._running = True
            self.bench.started_at = self.bench.started_at or _now()
            self._thread = threading.Thread(target=self._loop, name="bench-cadence", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Wait first so a freshly-added model gets price context before deciding.
            if self._stop.wait(self.cadence_seconds):
                break
            try:
                self.bench.run_decisions()
            except Exception:  # never let one bad tick kill the loop
                continue

    def tick_now(self) -> None:
        """Run one decision round immediately (manual trigger from the UI)."""
        self.bench.run_decisions()

    # --- Model menu ---------------------------------------------------------

    def available_models(self) -> dict[str, Any]:
        if self._models_cache is not None:
            return self._models_cache
        catalog: list[dict[str, Any]] = []
        try:
            for m in self.client.list_models():
                mid = m.get("id", "")
                if not mid or mid.startswith("~"):  # skip alias entries
                    continue
                pricing = m.get("pricing", {}) or {}
                catalog.append(
                    {
                        "id": mid,
                        "name": m.get("name", mid),
                        "prompt_price": pricing.get("prompt"),
                        "completion_price": pricing.get("completion"),
                        "context_length": m.get("context_length"),
                    }
                )
        except Exception as exc:
            return {"featured": [], "all": [], "error": str(exc)}
        by_id = {c["id"]: c for c in catalog}
        featured = [by_id[mid] for mid in FEATURED_MODELS if mid in by_id]
        catalog.sort(key=lambda c: c["id"])
        self._models_cache = {"featured": featured, "all": catalog, "error": None}
        return self._models_cache

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "cadence_seconds": self.cadence_seconds,
            "cadence_options": CADENCE_OPTIONS,
            "symbols": self.symbols,
            "competitors": self.bench.names(),
        }


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(tzinfo=None).isoformat()
