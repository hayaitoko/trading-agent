"""ACT toolkit — WS-Agent A3.

Five tools for executing trades and managing the approval-callback lifecycle:

  ``trade``                   — submit a single trade intent (terminal)
  ``trade_batch``             — submit multiple intents in one turn (terminal)
  ``update_protective_order`` — edit stop/TP/trail on an open position (non-terminal)
  ``confirm_trade``           — execute a pre-approved trade (terminal, callback-turn)
  ``abandon_trade``           — release a pre-approved trade unused (terminal, callback-turn)

All tool modules expose:
  DEFINITION      — OpenAI-compatible tool JSON for injection into the model's tool list
  CATALOG_ENTRY   — flat dict for list_tools() responses
  *Tool class     — callable class with ``.run(...)`` → ToolResult

Supporting dataclasses live in :mod:`~trading_agent.approval_queue`:
  TradeIntent, FillResult, PendingTrade, PendingTradeQueue.

The idempotency key helper is in :mod:`._base` and shared across tools.
"""

from .abandon_trade import CATALOG_ENTRY as ABANDON_CATALOG
from .abandon_trade import DEFINITION as ABANDON_DEF
from .abandon_trade import AbandonTradeTool
from .confirm_trade import CATALOG_ENTRY as CONFIRM_CATALOG
from .confirm_trade import DEFINITION as CONFIRM_DEF
from .confirm_trade import ConfirmTradeTool
from .trade import CATALOG_ENTRY as TRADE_CATALOG
from .trade import DEFINITION as TRADE_DEF
from .trade import TradeTool
from .trade_batch import CATALOG_ENTRY as TRADE_BATCH_CATALOG
from .trade_batch import DEFINITION as TRADE_BATCH_DEF
from .trade_batch import TradeBatchTool
from .update_protective_order import CATALOG_ENTRY as UPDATE_PROTECTIVE_CATALOG
from .update_protective_order import DEFINITION as UPDATE_PROTECTIVE_DEF
from .update_protective_order import UpdateProtectiveOrderTool

__all__ = [
    # Tool classes
    "TradeTool",
    "TradeBatchTool",
    "UpdateProtectiveOrderTool",
    "ConfirmTradeTool",
    "AbandonTradeTool",
    # Definitions (tool catalog JSON)
    "TRADE_DEF",
    "TRADE_BATCH_DEF",
    "UPDATE_PROTECTIVE_DEF",
    "CONFIRM_DEF",
    "ABANDON_DEF",
    # Catalog entries (list_tools data)
    "TRADE_CATALOG",
    "TRADE_BATCH_CATALOG",
    "UPDATE_PROTECTIVE_CATALOG",
    "CONFIRM_CATALOG",
    "ABANDON_CATALOG",
]
