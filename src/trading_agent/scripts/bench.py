"""Run the multi-model evaluation bench with a web leaderboard.

One price stream is fanned out to every competitor's isolated paper book; the
web UI lets you add models (live menu from OpenRouter), set the decision cadence,
and start/stop the autonomous loop.

Price sources:
    * **synthetic** (default): mean-reverting series per symbol, no market data key.
    * **--live**: real Alpaca quotes (needs ALPACA_API_KEY / ALPACA_SECRET_KEY).

Needs ``OPENROUTER_API_KEY`` for the model calls (in env or .env).

Run::

    trading-agent-bench --models anthropic/claude-opus-4.7,deepseek/deepseek-v4-flash --autostart
    trading-agent-bench --live --symbols AAPL,MSFT
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

from ..audit import AuditLogger
from ..bench.bench import Bench
from ..bench.controller import BenchController
from ..data_feed import MessageBus
from ..db import DatabaseManager
from ..feeds import synthetic_mean_reverting_bars
from ..llm.openrouter import OpenRouterClient
from ..web.bench_app import create_bench_app


def _load_dotenv(path: Path) -> None:
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
    p = argparse.ArgumentParser(description="Trading-agent multi-model bench")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--symbols", default="SYNTH-A,SYNTH-B", help="comma-separated symbols")
    p.add_argument("--initial-balance", type=float, default=100_000.0)
    p.add_argument("--cadence", type=int, default=300, help="decision cadence seconds")
    p.add_argument("--models", default="", help="comma-separated OpenRouter slugs to seed")
    p.add_argument("--autostart", action="store_true", help="start the decision loop on boot")
    p.add_argument("--no-zdr", action="store_true", help="disable ZDR provider routing")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-path", type=Path, default=Path("data/bench.db"))
    # synthetic feed
    p.add_argument("--bars", type=int, default=500)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--bar-interval", type=float, default=1.0)
    p.add_argument("--no-loop", action="store_true")
    # live feed
    p.add_argument("--live", action="store_true", help="real Alpaca quotes")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = _build_args(argv)
    _load_dotenv(Path(".env"))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    live = args.live
    # If the user kept the synthetic default symbols but asked for live, use equities.
    raw_symbols = "AAPL,MSFT" if (live and args.symbols == "SYNTH-A,SYNTH-B") else args.symbols
    symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]

    db = DatabaseManager(str(args.db_path))
    audit = AuditLogger(db, data_dir=args.data_dir)
    bus = MessageBus()

    client = OpenRouterClient(zdr=not args.no_zdr)
    bench = Bench(symbols, initial_balance=args.initial_balance, audit=audit)
    controller = BenchController(bench, client, symbols=symbols, cadence_seconds=args.cadence)

    for s in symbols:
        bus.subscribe(f"bar.{s}", bench.observe_bar)
        bus.subscribe(f"quote.{s}", bench.observe_quote)

    for slug in (m.strip() for m in args.models.split(",") if m.strip()):
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
                    bus.publish(f"bar.{s}", series[s][idx])
                stop.wait(args.bar_interval)
            if args.no_loop:
                return
            seed += len(symbols)

    def run_live() -> None:
        from ..broker_factory import build_alpaca_broker
        from ..feeds import LiveQuoteFeed

        source = build_alpaca_broker()
        feed = LiveQuoteFeed(
            bus, quote_source=source, symbols=symbols,
            paper_broker=None, poll_interval=args.bar_interval, emit_bars=True,
        )
        audit.info("bench_live_feed_started", module="bench", details={"symbols": symbols})
        while not stop.is_set():
            feed.poll_once()
            stop.wait(max(1.0, args.bar_interval))

    feed_thread = threading.Thread(
        target=run_live if live else run_synthetic, name="bench-feed", daemon=True
    )
    feed_thread.start()

    if args.autostart:
        controller.start()

    app = create_bench_app(controller)
    mode = f"LIVE Alpaca ({', '.join(symbols)})" if live else f"synthetic ({', '.join(symbols)})"
    print(f"=== model bench — {mode} ===")
    print(f"open  http://{args.host}:{args.port}/   (Ctrl-C to stop)")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()
        controller.stop()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
