"""WS-E — the Manager: an overseer agent + operator chat.

One cheap, configurable model that watches every paper-trading book (live bench
snapshot), the recent research briefs, and trader memories, and talks to the
operator. It **advises, summarizes, and flags — it never trades** (it has no
broker access by construction).

- :mod:`.chat` — conversation persistence over the ``conversations``/``turns``
  tables (WS-0 schema). Backs the cockpit's saved-chats.
- :mod:`.agent` — :class:`ManagerAgent`: context assembly + a single cost-gated
  model call per message, plus :meth:`ManagerAgent.flags` for things worth
  raising to the operator.
"""

from __future__ import annotations

from .agent import (
    DEFAULT_MANAGER_MODEL,
    ManagerAgent,
    ManagerConfigError,
    resolve_manager_ref,
)
from .chat import Conversation, ConversationStore, Turn

__all__ = [
    "DEFAULT_MANAGER_MODEL",
    "Conversation",
    "ConversationStore",
    "ManagerAgent",
    "ManagerConfigError",
    "Turn",
    "resolve_manager_ref",
]
