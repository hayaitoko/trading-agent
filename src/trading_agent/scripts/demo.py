"""End-to-end demo: synthetic mean-reverting price series → strategy → router → paper broker.

Run with::

    python -m trading_agent.scripts.demo
    # or after pip install -e .:
    trading-agent-demo

It uses no live credentials — everything is in-process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..approval_queue import ApprovalQueue
from ..audit import AuditLogger
from ..data_feed import MessageBus
from ..db import DatabaseManager
from ..enums import Mode
from ..feeds import CsvReplayFeed, synthetic_mean_reverting_bars
from ..paper_broker import PaperBroker
from ..risk_manager import RiskLimits, RiskManager
from ..signal_router import SignalRouter
from ..strategies.mean_reversion import MeanReversionStrategy

SYMBOL = "SYNTH-USD"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trading-agent end-to-end demo")
    parser.add_argument("--bars", type=int, default=300, help="number of synthetic bars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-balance", type=float, default=10000.0, help="paper broker starting cash"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="audit/JSONL output dir"
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path("data/demo.db"), help="SQLite db for audit/approvals"
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("strategies/config/mean_reversion.toml"),
        help="strategy TOML",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Wire everything up -------------------------------------------------

    db = DatabaseManager(str(args.db_path))
    audit = AuditLogger(db, data_dir=args.data_dir)

    bus = MessageBus()

    broker = PaperBroker(initial_balance=args.initial_balance)
    broker.connect()

    risk = RiskManager(
        limits=RiskLimits(
            max_daily_loss=10_000.0,
            max_position_size=100.0,
            max_trades_per_hour=1_000,
            max_open_positions=5,
        ),
        kill_switch_file=args.data_dir / ".kill_switch",
    )

    approvals = ApprovalQueue(db_path=args.data_dir / "approvals.db", executor=broker.place_order)
    router = SignalRouter(broker, approval_queue=approvals, global_mode=Mode.AUTONOMOUS)

    strategy = MeanReversionStrategy(config_path=args.strategy_config)

    # Generate synthetic bars and replay them through the bus.
    bars = synthetic_mean_reverting_bars(SYMBOL, n=args.bars, seed=args.seed)
    feed = CsvReplayFeed(bus, bars=bars, default_symbol=SYMBOL)

    trades: list[dict[str, Any]] = []

    def on_bar(bar: dict[str, Any]) -> None:
        # Mark-to-market for the paper broker so SELL orders can settle.
        broker.update_market_prices({bar["symbol"]: bar["close"]})

        signal = strategy.on_data(bar)
        side = signal.get("side", "NEUTRAL")
        if side == "NEUTRAL":
            return

        # Risk gate: kill switch + per-trade size + hourly trade count.
        scope = "demo"
        if risk.check_kill_switch():
            audit.warn("kill_switch_active", module="demo")
            return
        if risk.check_position_size(scope, signal["amount"]):
            audit.warn("position_size_blocked", details={"signal": signal})
            return
        if risk.increment_hourly_trades(scope):
            audit.warn("hourly_trade_limit_blocked", details={"signal": signal})
            return

        try:
            result = router.dispatch(signal)
        except Exception as e:
            audit.error("dispatch_failed", details={"error": str(e), "signal": signal})
            return

        if result is None:
            return

        if side == "LONG":
            risk.open_position(scope, signal["amount"])
        elif side == "SHORT":
            risk.close_position(scope)
        audit.trade(side.lower(), result)
        trades.append({"bar_timestamp": bar.get("timestamp"), **result})

    bus.subscribe(f"bar.{SYMBOL}", on_bar)
    published = feed.replay()

    # --- Report -------------------------------------------------------------

    final_balance = broker.get_balance()["cash"]
    positions = broker.get_positions()
    market_value = broker.get_account_value({SYMBOL: bars[-1]["close"]})
    pnl = market_value - args.initial_balance

    summary = {
        "bars_published": published,
        "initial_balance": args.initial_balance,
        "final_cash": final_balance,
        "open_positions": positions,
        "final_market_value": market_value,
        "pnl": pnl,
        "trades": len(trades),
    }
    audit.info("demo_summary", module="demo", details=summary)

    if not args.quiet:
        print("=== trading-agent demo ===")
        print(f"published bars     : {published}")
        print(f"initial balance    : {args.initial_balance:,.2f}")
        print(f"final cash         : {final_balance:,.2f}")
        print(f"final market value : {market_value:,.2f}")
        print(f"realized + unreal. : {pnl:+,.2f}")
        print(f"trades executed    : {len(trades)}")
        print(f"open positions     : {positions}")
        if trades:
            print("\n--- last 5 trades ---")
            for t in trades[-5:]:
                print(json.dumps(t, default=str))

    approvals.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
