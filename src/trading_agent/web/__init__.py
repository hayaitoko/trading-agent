"""Web notification center: surfaces approvals, risk warnings, and market moves.

The notification center is a **read-side** over the framework's existing state
(``ApprovalQueue`` SQLite, ``AuditLogger`` ``audit_log`` table) plus a
:class:`~trading_agent.web.market_watch.MarketMoveWatcher` that observes the
bar/quote stream. The only write path is approve/reject, which the web layer
delegates straight to :class:`~trading_agent.approval_queue.ApprovalQueue`.

Build the FastAPI app with :func:`trading_agent.web.app.create_app`; run an
end-to-end smoke server with ``trading-agent-serve`` (see
:mod:`trading_agent.scripts.serve`).
"""

from .market_watch import MarketMove, MarketMoveWatcher
from .notifications import Notification, NotificationCenter

__all__ = [
    "MarketMove",
    "MarketMoveWatcher",
    "Notification",
    "NotificationCenter",
]
