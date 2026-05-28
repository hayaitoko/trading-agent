"""Runtime control surface for the bench: roster, cadence, and the decision loop.

The web layer drives this — add/remove models, set the cadence (the UI dropdown),
start/stop the autonomous decision loop, and fetch the model menu from OpenRouter.
The loop runs on its own thread, calling :meth:`Bench.run_decisions` every
``cadence_seconds``.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..llm.trader import _MEMORY_RECALL_K, _RESEARCH_K, LLMTrader

if TYPE_CHECKING:
    from ..config.db import Database
    from ..config.settings_store import SettingsStore
    from ..data.history import HistoryService
    from ..llm.openrouter import OpenRouterClient
    from ..web.market_watch import MarketMove, MarketMoveWatcher
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

# P2: event-driven wake-hook defaults.
# De-dup window: if a threshold cross for a given symbol fires, suppress further
# wakes for this many seconds so a sustained move doesn't storm the model.
_WAKE_DEDUP_SECONDS: float = 60.0

# WS-A reflection defaults (overridable per-user in settings).
DEFAULT_REFLECTION_CADENCE = 4  # reflect every N rounds …
DEFAULT_REFLECTION_USD = 0.01  # … conservative per-trader distill cost estimate
_NOTABLE_PNL_DELTA_PCT = 5.0  # … or early when a book's return swings this much
_REFLECTION_DECISION_WINDOW = 10  # cap a book's decision log fed to reflection


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
        reflector: Any = None,
        owner_user_id: str | None = None,
        db: Database | None = None,
        market_watcher: MarketMoveWatcher | None = None,
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
        # The gated reflector drives the post-round write path; None → no learning.
        self.reflector = reflector
        self._owner_explicit = owner_user_id
        self.db = db
        self._settings_store: SettingsStore | None = None
        self._cached_owner_id: str | None = owner_user_id
        self._round = 0  # decision rounds completed (drives the reflection cadence)
        self._last_returns: dict[str, float] = {}  # per-book return_pct at last reflection
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._models_cache: dict[str, Any] | None = None
        # P2: event-driven wake hooks.
        self.market_watcher: MarketMoveWatcher | None = market_watcher
        # Per-symbol timestamp of the last model wake triggered by a market move.
        # Guards against storms: moves that arrive within _WAKE_DEDUP_SECONDS of
        # the last wake for that symbol are silently dropped.
        self._last_wake: dict[str, datetime] = {}

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

    def _settings(self) -> SettingsStore | None:
        """The per-user settings store, built lazily from ``db`` (None if no db)."""
        if self._settings_store is None and self.db is not None:
            from ..config.settings_store import SettingsStore

            self._settings_store = SettingsStore(self.db)
        return self._settings_store

    # --- Roster -------------------------------------------------------------

    def add_model(
        self,
        model: str,
        name: str | None = None,
        *,
        cash: float | None = None,
        style: str | None = None,
        # P6: per-trader intelligence overrides. When provided, these OVERRIDE
        # the owner-level settings for this specific trader instance. Pass the
        # same model twice with different flags to get an A/B on intelligence.
        intelligence_flags: dict[str, bool] | None = None,
        # P3: optional per-trader situation layer objects.
        regime_classifier: Any = None,
        social_aggregator: Any = None,
        calendar_events: list[dict[str, Any]] | None = None,
        # P4: optional pattern KB.
        pattern_store: Any = None,
    ) -> str:
        owner = self.owner_id
        research, research_k, memory_k = self.research, _RESEARCH_K, _MEMORY_RECALL_K
        settings = self._settings()

        # Resolve per-owner settings first, then let per-trader flags override.
        if owner is not None and settings is not None:
            # Per-owner knobs (WS-A): research read toggle + recall breadth.
            if not settings.get(owner, "trader_research_read", True):
                research = None
            research_k = int(settings.get(owner, "trader_research_k", _RESEARCH_K) or _RESEARCH_K)
            memory_k = int(
                settings.get(owner, "trader_memory_recall_k", _MEMORY_RECALL_K) or _MEMORY_RECALL_K
            )

        # P6: per-trader flag overrides disable specific intelligence layers.
        flags = dict(intelligence_flags or {})
        effective_research = research
        effective_memory = self.memory
        if flags.get("research") is False:
            effective_research = None
        if flags.get("memory") is False:
            effective_memory = None

        trader = LLMTrader(
            model,
            self.client,
            symbols=self.symbols,
            name=name or model,
            style=style,
            history=self.history,
            research=effective_research,
            memory=effective_memory,
            owner_user_id=owner,
            research_k=research_k,
            memory_k=memory_k,
            # P3: situation layer
            regime_classifier=regime_classifier,
            social_aggregator=social_aggregator,
            calendar_events=calendar_events or [],
            # P4: pattern KB
            pattern_store=pattern_store,
            # P6: pass the full flags dict so situation/pattern can also be gated
            intelligence_flags=flags,
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
            self._maybe_reflect()   # gated, self-contained — never raises
            self._scan_attention()  # A2: fire due reminders + tripped watchpoints

    def tick_now(self) -> None:
        """Run one decision round immediately (manual trigger from the UI)."""
        self.bench.run_decisions()
        self._maybe_reflect()
        self._scan_attention()

    # --- Reflection (WS-A write path) ---------------------------------------

    def _maybe_reflect(self) -> None:
        """After a round, maybe distill a few durable lessons per book.

        Guarded every which way so it can never kill the cadence loop: needs a
        reflector + a resolved owner, fires only on the cadence (or a notable
        P&L swing), and resolves a cheap model ref. The whole body is wrapped —
        any failure is swallowed; a budget exhaustion stops the round early.
        """
        reflector = self.reflector
        owner = self.owner_id
        if reflector is None or owner is None:
            return
        self._round += 1
        try:
            if not self._reflection_due(owner):
                return
            from ..manager.agent import resolve_reflection_ref

            ref = resolve_reflection_ref(reflector.settings, reflector.registry, owner)
            self._reflect_each_book(owner, ref, reflector)
        except Exception:
            return  # any setup/resolution failure: skip this round, keep looping

    def _reflection_due(self, owner: str) -> bool:
        cadence = int(self.reflector.settings.get(owner, "reflection_cadence_rounds", DEFAULT_REFLECTION_CADENCE) or 0)
        if cadence > 0 and self._round % cadence == 0:
            return True
        return self._notable_pnl_delta()

    def _notable_pnl_delta(self) -> bool:
        """True if any book's return moved ≥ threshold since the last reflection."""
        try:
            rows = self.bench.leaderboard()
        except Exception:
            return False
        for row in rows:
            name, ret = row.get("name"), row.get("return_pct")
            if name is None or not isinstance(ret, (int, float)):
                continue
            if abs(float(ret) - self._last_returns.get(name, 0.0)) >= _NOTABLE_PNL_DELTA_PCT:
                return True
        return False

    def _reflect_each_book(self, owner: str, ref: Any, reflector: Any) -> None:
        from ..memory.reflect import CostGateError

        estimated = float(
            reflector.settings.get(owner, "reflection_estimated_usd", DEFAULT_REFLECTION_USD)
            or DEFAULT_REFLECTION_USD
        )
        # Read the shared, locked views once (never comp.decisions directly).
        leaderboard = {r.get("name"): r for r in self.bench.leaderboard()}
        decisions = self.bench.recent_decisions(limit=200)
        for name in self.bench.names():
            context = self._reflection_context(name, leaderboard.get(name, {}), decisions)
            try:
                reflector.reflect_from_context(
                    owner, name, context, ref,
                    estimated_usd=estimated, tags=["auto-reflection"],
                )
            except CostGateError:
                break  # daily ceiling reached — stop spending this round
            except Exception:
                continue  # one book's failure must not block the rest
        # Re-baseline the P&L-delta gate to the post-reflection state.
        baseline: dict[str, float] = {}
        for row in self.bench.leaderboard():
            nm = row.get("name")
            if nm is not None:
                baseline[str(nm)] = float(row.get("return_pct", 0.0) or 0.0)
        self._last_returns = baseline

    def _reflection_context(self, name: str, row: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
        """A compact round summary for one book: its P&L row + own recent decisions."""
        lines = [f"Trader: {name}"]
        if row:
            lines.append(
                f"P&L {row.get('pnl', 0):+,.0f} ({row.get('return_pct', 0):+.2f}%), "
                f"account value ${row.get('account_value', 0):,.0f}, "
                f"wins {row.get('wins', 0)} / losses {row.get('losses', 0)}, "
                f"{row.get('trades', 0)} trades over {row.get('decisions', 0)} decisions"
            )
        own = [d for d in decisions if d.get("competitor") == name][:_REFLECTION_DECISION_WINDOW]
        lines.append("Recent decisions (newest first):")
        if not own:
            lines.append("  (none this window)")
        for d in own:
            line = (
                f"  {d.get('timestamp', '')} {d.get('action', '?')} "
                f"{d.get('quantity', '')} {d.get('symbol', '')} [{d.get('status', '?')}]"
            )
            if d.get("reason"):
                line += f" — {d['reason']}"
            lines.append(line.rstrip())
        return "\n".join(lines)

    # --- P2: Event-driven wake hooks ----------------------------------------

    def on_market_move(self, move: MarketMove) -> None:
        """Called when MarketMoveWatcher detects a threshold cross.

        This is the **soft-stop** path: the model is woken to decide whether
        the move is a genuine breakout or a spike to fade.  Hard stops (broker-
        level P0 stop orders and the hard floor) fire separately and
        deterministically — they don't come through here.

        A de-dup window (_WAKE_DEDUP_SECONDS) prevents a sustained multi-tick
        move from hammering the model on every tick.
        """
        if not self._running:
            return
        symbol = move.symbol
        now = datetime.now(UTC)
        with self._lock:
            last = self._last_wake.get(symbol)
            if last is not None and (now - last).total_seconds() < _WAKE_DEDUP_SECONDS:
                return  # de-dup: suppress this wake
            self._last_wake[symbol] = now
        # Scoped off-cadence decision: only books with a position in the symbol.
        # The P1 snapshot captures state just before the LLM call, so concurrent
        # hard-floor flattens are caught and the SELL won't fire wrong-way.
        self._tick_for_symbol(symbol)

    def _tick_for_symbol(self, symbol: str) -> None:
        """Run an off-cadence decision round for competitors holding ``symbol``."""
        try:
            self.bench.run_decisions_for_symbol(symbol)
        except Exception:
            return  # never let a wake-hook failure break anything
        self._maybe_reflect()

    # --- A2: Attention-queue scanner ----------------------------------------

    def _scan_attention(self) -> None:
        """Per-tick scan of the pending-attention queue.

        Fires:
          - **Reminders** whose ``payload.when_unix`` has elapsed (UTC).
          - **Watchpoints** whose condition trips against current market data.

        Each fire enqueues an event-driven turn by calling
        :meth:`~trading_agent.bench.bench.Bench.run_decisions_for_symbol` or
        waking the individual competitor (reuses the market-move mechanism so
        the bench doesn't need a new entry point yet — full event-driven wake
        is A4's deliverable).

        Design notes:
          - Runs AFTER the cadence tick so price data is fresh.
          - Never raises: any failure is silently swallowed so it can't kill
            the cadence loop.
          - The actual wake-and-decide is gated by whether the bench is running;
            if stopped, rows still fire (mark_fired) but no turn is triggered.
        """
        try:
            self._do_scan_attention()
        except Exception:
            pass  # never let attention scan kill the loop

    def _do_scan_attention(self) -> None:
        """Concrete attention scan (may raise; wrapped by :meth:`_scan_attention`)."""
        # Resolve the attention queue from any competitor's AgentTrader.
        # All traders on the same bench share the same queue instance (or None).
        aq = self._resolve_attention_queue()
        if aq is None:
            return

        import time as _time

        now = int(_time.time())
        # Clean up expired rows first.
        aq.expire_old()

        # Pull all unfired, non-expired rows.
        rows = aq.poll_all_due(now=now)
        if not rows:
            return

        # Build lightweight market-data snapshots for watchpoint evaluation.
        prices = self._attention_prices()
        approval_syms = self._attention_approval_symbols()

        # Import evaluator lazily to avoid heavy import at module load.
        from ..intel.tools.note.watchpoint import evaluate_condition

        fired_traders: set[str] = set()
        for row in rows:
            if row.kind == "reminder":
                when_unix = row.payload.get("when_unix")
                if when_unix is not None and int(when_unix) <= now:
                    aq.mark_fired(row.id, "elapsed")
                    fired_traders.add(row.trader_id)
            elif row.kind == "watchpoint":
                symbol = str(row.payload.get("symbol", "")).upper()
                tripped, reason = evaluate_condition(
                    row.payload,
                    last_prices=prices,
                    approval_symbols=approval_syms,
                )
                if tripped:
                    aq.mark_fired(row.id, reason)
                    fired_traders.add(row.trader_id)
                    # Trigger an off-cadence turn for traders watching this symbol.
                    if self._running and symbol:
                        try:
                            self.bench.run_decisions_for_symbol(symbol)
                        except Exception:
                            pass

        # For reminder fires (no specific symbol), wake the affected traders.
        if self._running and fired_traders:
            for trader_name in fired_traders:
                try:
                    # Wake traders that had a reminder fire (no symbol context).
                    comp = self.bench._competitors.get(trader_name)
                    if comp is not None:
                        self.bench._run_one(comp)
                except Exception:
                    pass

    def _resolve_attention_queue(self) -> Any:
        """Return the attention_queue from any AgentTrader competitor, or None."""
        for comp in self.bench._competitors.values():
            aq = getattr(comp.trader, "attention_queue", None)
            if aq is not None:
                return aq
        return None

    def _attention_prices(self) -> dict[str, float]:
        """Current last prices for watchpoint evaluation."""
        try:
            return dict(self.bench._last_prices)
        except Exception:
            return {}

    def _attention_approval_symbols(self) -> set[str]:
        """Set of symbols with pending approval entries (best-effort).

        Gathers pending approval records from every competitor's broker,
        scanning for symbols with open approval-queue entries.  Returns an
        empty set if the approval queue is unavailable or the bench has no
        competitors.
        """
        symbols: set[str] = set()
        try:
            for comp in self.bench._competitors.values():
                # AgentTrader may carry an approval_queue reference (A3 wires this).
                # For now, check if broker has any pending orders for the signal model.
                aq = getattr(comp.trader, "_approval_queue", None)
                if aq is None:
                    continue
                for record in aq.pending():
                    sym = str(record.signal.get("asset", "") or "").upper()
                    if sym:
                        symbols.add(sym)
        except Exception:
            pass
        return symbols

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
