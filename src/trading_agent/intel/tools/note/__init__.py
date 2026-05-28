"""NOTE toolkit — WS-Agent A2.

Five tools for durable state and deferred attention:

  ``reflect``         — write a lesson to the trader's private memory (WS-D)
  ``remind_me``       — time-based self-poke (fires after elapsed time)
  ``watchpoint``      — event-based monitor (fires on price/vol/news condition)
  ``watch_symbol``    — add a symbol to the trader's personal watchlist
  ``unwatch_symbol``  — remove a symbol from the watchlist

All tool modules expose:
  DEFINITION      — OpenAI-compatible tool JSON for injection into the model's tool list
  CATALOG_ENTRY   — flat dict for list_tools() responses
  *Tool class     — callable class with ``.run(...)`` → ToolResult

The scheduler scanner lives in :mod:`~trading_agent.intel.attention_queue` and is
called from :class:`~trading_agent.bench.controller.BenchController._scan_attention`.
"""

from .reflect import CATALOG_ENTRY as REFLECT_CATALOG
from .reflect import DEFINITION as REFLECT_DEF
from .reflect import ReflectTool
from .remind_me import CATALOG_ENTRY as REMIND_CATALOG
from .remind_me import DEFINITION as REMIND_DEF
from .remind_me import RemindMeTool
from .unwatch_symbol import (
    CATALOG_ENTRY as UNWATCH_CATALOG,
)
from .unwatch_symbol import (
    DEFINITION as UNWATCH_DEF,
)
from .unwatch_symbol import (
    UnwatchSymbolTool,
)
from .watch_symbol import (
    CATALOG_ENTRY as WATCH_CATALOG,
)
from .watch_symbol import (
    DEFINITION as WATCH_DEF,
)
from .watch_symbol import (
    WatchSymbolTool,
)
from .watchpoint import (
    CATALOG_ENTRY as WATCHPOINT_CATALOG,
)
from .watchpoint import (
    DEFINITION as WATCHPOINT_DEF,
)
from .watchpoint import (
    WatchpointTool,
    evaluate_condition,
)

__all__ = [
    # Tool classes
    "ReflectTool",
    "RemindMeTool",
    "WatchpointTool",
    "WatchSymbolTool",
    "UnwatchSymbolTool",
    # Definitions (tool catalog JSON)
    "REFLECT_DEF",
    "REMIND_DEF",
    "WATCHPOINT_DEF",
    "WATCH_DEF",
    "UNWATCH_DEF",
    # Catalog entries (list_tools data)
    "REFLECT_CATALOG",
    "REMIND_CATALOG",
    "WATCHPOINT_CATALOG",
    "WATCH_CATALOG",
    "UNWATCH_CATALOG",
    # Watchpoint condition evaluator (used by scheduler)
    "evaluate_condition",
]
