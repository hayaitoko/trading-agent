import json
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_agent.web.state import AppState

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": (
                "List all paper-trading accounts with their id, name, enabled status, "
                "cash, equity (market value of positions), and total."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account",
            "description": (
                "Get a detailed snapshot of one account: cash, positions with current "
                "prices and unrealized P&L, and trades from the last 24 hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The id of the account, e.g. 'paper-aggressive'.",
                    }
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trades",
            "description": "Get the most recent trades for an account, newest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": (
                "List every note file in the agent's memory as a flat array of "
                "relative paths (e.g. 'companies/NVDA.md'). Notes are organized in "
                "companies/, sectors/, macro/, general/."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": (
                "Read a single note by relative path. Returns the raw markdown "
                "including frontmatter so you can see created/updated timestamps "
                "and inline (as of YYYY-MM-DD) markers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path like 'companies/NVDA.md'.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Substring search across every note. Returns matching paths with a "
                "short excerpt around each match. Case-insensitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
]


def _json_default(obj):
    if isinstance(obj, Decimal):
        return f"{obj:.2f}"
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    raise TypeError(f"cannot serialize {type(obj)}")


def _dump(value) -> str:
    return json.dumps(value, default=_json_default)


async def _list_accounts(state: "AppState") -> str:
    rows = []
    for a in state.accounts.values():
        positions = await a.broker.get_positions()
        cash = await a.broker.get_cash()
        equity = sum((p.qty * p.current_price for p in positions), start=Decimal(0))
        rows.append({
            "id": a.id,
            "name": a.name,
            "enabled": a.enabled,
            "cash": cash,
            "equity": equity,
            "total": cash + equity,
            "position_count": len(positions),
        })
    return _dump(rows)


async def _get_account(state: "AppState", account_id: str) -> str:
    if account_id not in state.accounts:
        return _dump({"error": "account not found", "account_id": account_id})
    account = state.accounts[account_id]
    cash = await account.broker.get_cash()
    positions = await account.broker.get_positions()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    trades = await account.broker.get_trades(since=today_start)
    return _dump({
        "id": account.id,
        "name": account.name,
        "enabled": account.enabled,
        "cash": cash,
        "positions": [
            {
                "ticker": p.ticker,
                "qty": p.qty,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "unrealized_pnl": (p.current_price - p.avg_cost) * p.qty,
            }
            for p in positions
        ],
        "trades_today": [
            {
                "ticker": t.ticker,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "executed_at": t.executed_at,
            }
            for t in trades
        ],
    })


async def _get_trades(state: "AppState", account_id: str, limit: int = 20) -> str:
    if account_id not in state.accounts:
        return _dump({"error": "account not found", "account_id": account_id})
    trades = await state.accounts[account_id].broker.get_trades()
    trades_sorted = sorted(trades, key=lambda t: t.executed_at, reverse=True)[:limit]
    return _dump([
        {
            "ticker": t.ticker,
            "side": t.side,
            "qty": t.qty,
            "price": t.price,
            "executed_at": t.executed_at,
        }
        for t in trades_sorted
    ])


def _list_notes(state: "AppState") -> str:
    from trading_agent.notes import list_tree
    if state.notes_dir is None:
        return _dump({"error": "notes_dir not configured"})

    paths: list[str] = []

    def collect(node):
        if node.type == "file":
            paths.append(node.path)
        for child in node.children:
            collect(child)

    collect(list_tree(state.notes_dir))
    return _dump(paths)


def _read_note(state: "AppState", path: str) -> str:
    from trading_agent.notes import NotesError, read_note
    if state.notes_dir is None:
        return _dump({"error": "notes_dir not configured"})
    try:
        return _dump({"path": path, "content": read_note(state.notes_dir, path)})
    except NotesError as e:
        return _dump({"error": str(e), "path": path})


def _search_notes(state: "AppState", query: str, limit: int = 10) -> str:
    from trading_agent.notes import search_notes
    if state.notes_dir is None:
        return _dump({"error": "notes_dir not configured"})
    return _dump(search_notes(state.notes_dir, query, limit=limit))


async def execute(state: "AppState", name: str, args: dict) -> str:
    if name == "list_accounts":
        return await _list_accounts(state)
    if name == "get_account":
        return await _get_account(state, args["account_id"])
    if name == "get_trades":
        return await _get_trades(state, args["account_id"], int(args.get("limit", 20)))
    if name == "list_notes":
        return _list_notes(state)
    if name == "read_note":
        return _read_note(state, args["path"])
    if name == "search_notes":
        return _search_notes(state, args["query"], int(args.get("limit", 10)))
    return _dump({"error": f"unknown tool: {name}"})


SYSTEM_PROMPT = (
    "You are the assistant inside the trading-agent dashboard, a personal homelab tool "
    "that runs paper-trading accounts driven (eventually) by sentiment scraped from "
    "investment forums.\n\n"
    "What you can do via tools:\n"
    "- list_accounts: see all paper-trading accounts and their current state\n"
    "- get_account: detailed snapshot of one account (positions, today's trades)\n"
    "- get_trades: trade history for an account\n"
    "- list_notes / read_note / search_notes: the agent's memory. Notes are\n"
    "  organized under companies/, sectors/, macro/, general/. Every note has\n"
    "  frontmatter with created/updated dates and inline (as of YYYY-MM-DD)\n"
    "  markers for time-bound claims, so you can tell what is archival vs\n"
    "  current.\n\n"
    "What you cannot do yet (and should be honest about):\n"
    "- You have no access to live news, market data, or social-media sentiment.\n"
    "- You have no signal/decision log explaining why a trade was made, because the\n"
    "  scraper and signal layers are not built yet. Today's MockBroker trades are\n"
    "  demo seeds, not algorithm output.\n"
    "- Prices are demo quotes, not real market data.\n\n"
    "Be concise. Use tools when the user asks about specific accounts or positions. "
    "When the user asks about things outside your access (news, why-did-you-buy, "
    "sentiment), say so directly rather than speculating."
)
