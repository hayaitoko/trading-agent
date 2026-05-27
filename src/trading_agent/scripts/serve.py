"""End-to-end smoke server for the web notification center.

Runs the full pipeline in **one process** so the web layer shares the live
``ApprovalQueue`` executor and in-memory ``PaperBroker``:

    feed (thread) -> strategy -> risk gate -> SignalRouter[APPROVAL] -> queue
                                                                          |
    web (uvicorn) -> /api/notifications  <-- NotificationCenter           |
                  -> POST approve --------> queue.approve() -> broker -----+

Two price sources:
    * **synthetic** (default): replays a mean-reverting series, no credentials.
      Exercises all three alert types right now.
    * **--live**: polls real Alpaca quotes into the PaperBroker (real market
      values, simulated fills — "Path B"). Needs ALPACA_API_KEY / ALPACA_SECRET_KEY
      in the environment or .env (data access is free on a paper account).

Run::

    trading-agent-serve                      # synthetic, http://127.0.0.1:8000
    trading-agent-serve --live --symbols AAPL,MSFT
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..approval_queue import ApprovalQueue
from ..audit import AuditLogger
from ..bench.bench import Bench
from ..bench.controller import BenchController
from ..config.db import Database
from ..data_feed import MessageBus
from ..db import DatabaseManager
from ..enums import Mode
from ..feeds import synthetic_mean_reverting_bars
from ..llm.openrouter import OpenRouterClient, OpenRouterError
from ..paper_broker import PaperBroker
from ..risk_manager import RiskLimits, RiskManager
from ..signal_router import SignalRouter, _signal_to_order
from ..strategies.mean_reversion import MeanReversionStrategy
from ..web.app import create_app, create_cockpit_app
from ..web.market_watch import MarketMoveWatcher
from ..web.notifications import NotificationCenter

if TYPE_CHECKING:
    import httpx
    from fastapi import FastAPI

SCOPE = "smoke"
DEFAULT_SYMBOL = "SYNTH-USD"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets vars that aren't already in the environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading-agent web alerts smoke server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-path", type=Path, default=Path("data/serve.db"))
    p.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("strategies/config/mean_reversion.toml"),
    )
    p.add_argument("--bars", type=int, default=400, help="synthetic bars per cycle")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bar-interval", type=float, default=1.2, help="seconds between bars/polls")
    p.add_argument("--no-loop", action="store_true", help="stop after one synthetic cycle")
    p.add_argument(
        "--max-trades-per-hour",
        type=int,
        default=8,
        help="low value makes the hourly-limit risk alert fire during the demo",
    )
    p.add_argument("--threshold-pct", type=float, default=1.5, help="market-move alert band (%)")
    p.add_argument("--live", action="store_true", help="use real Alpaca quotes (Path B)")
    p.add_argument("--symbols", default="AAPL", help="comma-separated symbols for --live")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    argv = sys.argv[1:] if argv is None else argv
    if "--cockpit" in argv:
        return cockpit_main(argv)

    args = _build_args(argv)
    _load_dotenv(Path(".env"))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Shared state -------------------------------------------------------
    db = DatabaseManager(str(args.db_path))
    audit = AuditLogger(db, data_dir=args.data_dir)
    bus = MessageBus()
    broker = PaperBroker(initial_balance=args.initial_balance)
    broker.connect()
    broker_lock = threading.Lock()  # PaperBroker is shared across feed + web threads

    risk = RiskManager(
        limits=RiskLimits(
            max_daily_loss=100_000.0,
            max_position_size=1_000.0,
            max_trades_per_hour=args.max_trades_per_hour,
            max_open_positions=50,
        ),
        kill_switch_file=args.data_dir / ".kill_switch",
    )

    def execute(signal: dict[str, Any]) -> dict[str, Any] | None:
        """Approval executor: map signal -> order, fill, audit, account for risk."""
        with broker_lock:
            result = broker.place_order(_signal_to_order(signal))
        side = str(signal.get("side", "")).upper()
        if result is not None:
            audit.trade(side.lower() or "order", result)
            if side in ("LONG", "BUY"):
                risk.open_position(SCOPE, float(signal.get("amount", 0) or 0))
            elif side in ("SHORT", "SELL"):
                risk.close_position(SCOPE)
        return result

    approvals = ApprovalQueue(db_path=args.data_dir / "approvals.db", executor=execute)
    router = SignalRouter(broker, approval_queue=approvals, global_mode=Mode.APPROVAL)
    strategy = MeanReversionStrategy(config_path=args.strategy_config)
    watch = MarketMoveWatcher(threshold_pct=args.threshold_pct)

    def account_snapshot() -> dict[str, Any]:
        with broker_lock:
            cash = broker.get_balance()["cash"]
            positions = broker.get_positions()
            market_value = broker.get_account_value(dict(broker.market_prices))
        return {
            "cash": cash,
            "market_value": market_value,
            "pnl": market_value - args.initial_balance,
            "positions_count": len(positions),
            "positions": positions,
        }

    center = NotificationCenter(approvals, db, watch, account_provider=account_snapshot)

    # --- Per-bar pipeline (runs on the feed thread) -------------------------
    live = args.live
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if live else [DEFAULT_SYMBOL]

    def on_bar(bar: dict[str, Any]) -> None:
        symbol = bar.get("symbol")
        close = bar.get("close")
        if not live and symbol is not None and close is not None:
            with broker_lock:
                broker.update_market_prices({str(symbol): float(close)})
        watch.observe(symbol, close)

        signal = strategy.on_data(bar)
        side = signal.get("side", "NEUTRAL")
        if side == "NEUTRAL":
            return
        if risk.check_kill_switch():
            audit.warn("kill_switch_active", module="serve")
            return
        if risk.check_position_size(SCOPE, signal["amount"]):
            audit.warn("position_size_blocked", module="serve", details={"signal": signal})
            return
        if risk.increment_hourly_trades(SCOPE):
            audit.warn("hourly_trade_limit_blocked", module="serve", details={"signal": signal})
            return
        try:
            router.dispatch(signal)  # APPROVAL mode -> enqueues a pending approval
        except Exception as exc:
            audit.error("dispatch_failed", module="serve", details={"error": str(exc)})

    for sym in symbols:
        bus.subscribe(f"bar.{sym}", on_bar)

    stop = threading.Event()

    def run_synthetic() -> None:
        seed = args.seed
        while not stop.is_set():
            bars = synthetic_mean_reverting_bars(DEFAULT_SYMBOL, n=args.bars, seed=seed)
            for bar in bars:
                if stop.is_set():
                    return
                bus.publish(f"bar.{DEFAULT_SYMBOL}", bar)
                stop.wait(args.bar_interval)
            if args.no_loop:
                return
            seed += 1

    def run_live() -> None:
        from ..broker_factory import build_alpaca_broker
        from ..feeds import LiveQuoteFeed

        source = build_alpaca_broker()  # paper endpoint by default; data is free
        feed = LiveQuoteFeed(
            bus, quote_source=source, symbols=symbols,
            paper_broker=broker, poll_interval=args.bar_interval, emit_bars=True,
        )
        audit.info("live_feed_started", module="serve", details={"symbols": symbols})
        while not stop.is_set():
            with broker_lock:
                feed.poll_once()
            stop.wait(args.bar_interval)

    feed_thread = threading.Thread(
        target=run_live if live else run_synthetic, name="feed", daemon=True
    )
    feed_thread.start()

    app = create_app(center, approvals)

    mode = f"LIVE Alpaca quotes ({', '.join(symbols)})" if live else "synthetic mean-reversion"
    print(f"=== trading-agent alerts — {mode} ===")
    print(f"open  http://{args.host}:{args.port}/   (Ctrl-C to stop)")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()
        approvals.close()
    return 0


# =============================================================================
# Cockpit (multi-user) serve entrypoint — WS-I engine wiring.
#
# `create_cockpit_app` deliberately leaves app.state.bench (etc.) unset so unit
# tests get graceful-empty routers. The *serve* process attaches the live engine
# here, per CONTRACTS §"Runtime wiring via app.state". `build_cockpit` is the
# testable factory (no threads, import-safe); `cockpit_main` is the CLI that adds
# a price feed + uvicorn. Run with::
#
#     python -m trading_agent.scripts.serve --cockpit            # via __main__
#     uvicorn trading_agent.scripts.serve:build_cockpit --factory --host 0.0.0.0
# =============================================================================

DEFAULT_COCKPIT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META"]


def _echo_executor(signal: dict[str, Any]) -> dict[str, Any]:
    """Minimal approval-queue executor for the cockpit.

    The bench auto-trades and never enqueues approvals, so this is effectively
    unused; it exists only so a manually-enqueued proposal can still be approved
    rather than raising 'no executor configured'.
    """
    return signal


def build_cockpit(
    *,
    db: Database | None = None,
    symbols: list[str] | None = None,
    initial_balance: float = 100_000.0,
    max_position_size: float = 1_000.0,
    cadence_seconds: int = 300,
    data_dir: str | Path | None = None,
    threshold_pct: float = 1.5,
    owner_user_id: str | None = None,
    openrouter_client: OpenRouterClient | None = None,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    """Cockpit app with the live trading engine attached to ``app.state``.

    Attaches ``bench``, ``market_watch``, ``risk``, ``approvals`` always, and
    ``bench_controller`` when an OpenRouter key/client is available (the
    add-trader wizard needs it; without it the read surfaces still work and the
    create route answers 503).

    The WS-A intelligence layer (``research`` / ``memory`` / ``reflector``) is
    attached too. It binds to one ``owner_user_id`` (explicit, else resolved from
    the env / single-user fallback); when no owner resolves, memory/reflection
    stay dark (history-only) exactly like the manager's None-guards.
    """
    base_dir = Path(data_dir) if data_dir is not None else Path("data")
    base_dir.mkdir(parents=True, exist_ok=True)
    syms = list(symbols) if symbols else list(DEFAULT_COCKPIT_SYMBOLS)

    app = create_cockpit_app(db, transport=transport)
    bench = Bench(syms, initial_balance=initial_balance, max_position_size=max_position_size)
    app.state.bench = bench
    app.state.market_watch = MarketMoveWatcher(threshold_pct=threshold_pct)
    app.state.risk = RiskManager(kill_switch_file=base_dir / ".kill_switch")
    app.state.approvals = ApprovalQueue(db_path=base_dir / "approvals.db", executor=_echo_executor)

    # Real market data for the cockpit charts + fundamentals popup. Bars come
    # from the same Alpaca data key as the live books; fundamentals from whatever
    # provider has a key in the environment (real data only — no synthetic feed).
    from ..data.history import FundamentalsProvider, build_history_service
    from ..data.providers.alpaca import AlpacaBarProvider

    fundamentals_provider: FundamentalsProvider | None = None
    if os.environ.get("FINNHUB_API_KEY"):
        from ..data.providers.finnhub import FinnhubProvider

        fundamentals_provider = FinnhubProvider()
    elif os.environ.get("POLYGON_API_KEY"):
        from ..data.providers.polygon import PolygonProvider

        fundamentals_provider = PolygonProvider()
    history = build_history_service(
        bar_provider=AlpacaBarProvider(),
        fundamentals_provider=fundamentals_provider,
    )
    app.state.history = history

    # WS-A intelligence: research briefs (shared per user) + per-trader memory +
    # the gated reflector. Built once at serve start; lights up the manager chat
    # and the trader decision path.
    _attach_intelligence(app, base_dir, owner_user_id, transport)

    client = openrouter_client
    if client is None:
        try:
            client = OpenRouterClient(zdr=True, transport=transport)
        except OpenRouterError:
            client = None  # no key -> reads still work; add-trader returns 503
    if client is not None:
        # Thread the intelligence layer into the controller so every trader it
        # adds decides on history + research + its own memory, and reflects after
        # each round. owner_user_id stays the *explicit* hint — the controller
        # re-resolves it lazily (fresh-box safe), matching _attach_intelligence.
        app.state.bench_controller = BenchController(
            bench,
            client,
            symbols=syms,
            cadence_seconds=cadence_seconds,
            history=history,
            research=app.state.research,
            memory=app.state.memory,
            reflector=app.state.reflector,
            owner_user_id=owner_user_id,
            db=app.state.db,
        )
    return app


def _attach_intelligence(
    app: FastAPI,
    base_dir: Path,
    owner_user_id: str | None,
    transport: httpx.BaseTransport | None,
) -> str | None:
    """Attach ``research`` / ``memory`` / ``reflector`` to ``app.state``.

    Returns the resolved owner (or None). The research store always attaches (its
    structured reads need no embedder); the per-trader memory + reflector attach
    only once an owner resolves *and* an embedder can be built, since both are
    user-namespaced. The shared vector DB lives under ``data_dir`` unless
    ``TRADING_AGENT_MEMORY_DB`` overrides it, so tests stay self-contained.
    """
    from ..config.users import resolve_owner_user_id
    from ..memory import MemoryStore, Reflector, make_embedder, make_vector_store
    from ..research import ResearchStore

    db: Database = app.state.db
    settings = app.state.settings
    registry = app.state.endpoints

    # Resolve the owner properly (explicit id/username → env → single-user), so the
    # embedder + memory bind to a real user_id, not a raw CLI string.
    owner = resolve_owner_user_id(db, explicit=owner_user_id)
    app.state.owner_user_id = owner
    mem_path = os.environ.get("TRADING_AGENT_MEMORY_DB") or str(base_dir / "memory.db")
    vstore_name = settings.get(owner, "vstore", "sqlite-vec") if owner else "sqlite-vec"
    vstore = make_vector_store(vstore_name, path=mem_path)

    # LocalEmbedder.__init__ is safe with no Ollama — EmbedError is lazy at embed().
    embedder = make_embedder(registry, settings, owner, transport=transport) if owner else None

    app.state.research = ResearchStore(db, vstore, embedder)
    memory = MemoryStore(vstore, embedder) if (owner and embedder is not None) else None
    app.state.memory = memory
    app.state.reflector = (
        Reflector(memory, settings=settings, registry=registry) if memory is not None else None
    )
    return owner


def _print_intelligence_status(app: FastAPI) -> None:
    """One line on whether the WS-A trader-intelligence layer is live."""
    owner = getattr(app.state, "owner_user_id", None)
    if not owner:
        print(
            "intelligence: OFF — no owner resolved; traders run history-only. "
            "Set --owner <user>, TRADING_AGENT_OWNER_ID, or sign up exactly one user."
        )
        return
    try:
        endpoints = app.state.endpoints.list(owner)
        embed_ok = any(e.type == "local" and e.enabled for e in endpoints)
    except Exception:
        embed_ok = False
    embed = (
        "local embed endpoint present"
        if embed_ok
        else "no local embed endpoint — research/memory recall skipped until one is added"
    )
    print(f"intelligence: ON — owner {owner}; research + reflection wired; {embed}.")


def _cockpit_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading-agent cockpit server (multi-user)")
    p.add_argument("--cockpit", action="store_true", help="(routing flag; ignored here)")
    p.add_argument("--host", default="0.0.0.0", help="bind address — 0.0.0.0 for LAN access")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--symbols", default=",".join(DEFAULT_COCKPIT_SYMBOLS))
    p.add_argument("--initial-balance", type=float, default=100_000.0)
    p.add_argument("--cadence", type=int, default=300, help="decision cadence seconds")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--owner",
        default=None,
        help="bind the trader-intelligence layer to this user (id or username); "
        "falls back to TRADING_AGENT_OWNER_ID, then a lone signed-up user",
    )
    p.add_argument("--models", default="", help="comma-separated OpenRouter slugs to seed")
    p.add_argument("--no-feed", action="store_true", help="don't run the synthetic price feed")
    p.add_argument("--bars", type=int, default=500)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--bar-interval", type=float, default=1.0)
    return p.parse_args(argv)


def cockpit_main(argv: list[str] | None = None) -> int:
    """CLI: build the cockpit, drive a synthetic price feed, serve over uvicorn."""
    import uvicorn

    args = _cockpit_args(argv)
    _load_dotenv(Path(".env"))
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    app = build_cockpit(
        symbols=symbols,
        initial_balance=args.initial_balance,
        cadence_seconds=args.cadence,
        data_dir=args.data_dir,
        owner_user_id=args.owner,
    )
    bench = app.state.bench
    watch = app.state.market_watch
    controller = getattr(app.state, "bench_controller", None)
    if controller is None:
        print("note: no OPENROUTER_API_KEY — add-trader is disabled until one is set.")
    _print_intelligence_status(app)
    for slug in (m.strip() for m in args.models.split(",") if m.strip()):
        if controller is not None:
            controller.add_model(slug)

    stop = threading.Event()

    def run_synthetic() -> None:
        seed = args.seed
        while not stop.is_set():
            series = {
                s: synthetic_mean_reverting_bars(s, n=args.bars, seed=seed + i)
                for i, s in enumerate(symbols)
            }
            for idx in range(args.bars):
                if stop.is_set():
                    return
                for s in symbols:
                    bar = series[s][idx]
                    bench.observe_bar(bar)
                    watch.observe(bar.get("symbol"), bar.get("close"))
                stop.wait(args.bar_interval)
            seed += len(symbols)

    if not args.no_feed:
        threading.Thread(target=run_synthetic, name="cockpit-feed", daemon=True).start()

    print("=== trading-agent cockpit (multi-user) ===")
    print(f"open  http://{args.host}:{args.port}/   (Ctrl-C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()
        if controller is not None:
            controller.stop()
        app.state.approvals.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
