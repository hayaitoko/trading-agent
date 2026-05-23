"""Audit log: writes every recorded event to both SQLite and a per-day JSONL file.

JSONL files are at ``data/audit.YYYY-MM-DD.jsonl`` (artoo activity.py style).
SQLite rows go into the ``audit_log`` table created by :mod:`db`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Union

from .db import DatabaseManager

PathLike = Union[str, Path]


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AuditLogger:
    """Dual-sink audit writer.

    Each :meth:`log` call writes one JSONL line and one row in ``audit_log``.
    A single ``AuditLogger`` instance is safe to share — sqlite connection is
    serialized by ``DatabaseManager`` under the hood, and JSONL appends are
    atomic at the OS level for small writes on POSIX.
    """

    def __init__(
        self,
        db: DatabaseManager,
        data_dir: PathLike = "data",
    ) -> None:
        self.db = db
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- Public API ---------------------------------------------------------

    def log(
        self,
        level: str,
        message: str,
        module: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = _utcnow()
        record = {
            "timestamp": now.isoformat(),
            "level": level.upper(),
            "message": message,
            "module": module,
            "details": details or {},
        }
        self._write_jsonl(now, record)
        self._write_sqlite(now, record)

    def info(self, message: str, **kwargs: Any) -> None:
        self.log("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        self.log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log("ERROR", message, **kwargs)

    def trade(self, action: str, order: dict[str, Any]) -> None:
        """Record a trade event with the full order payload in ``details``."""
        self.log("INFO", f"trade.{action}", module="trading", details={"order": order})

    # --- Internals ----------------------------------------------------------

    def _jsonl_path(self, when: datetime) -> Path:
        return self.data_dir / f"audit.{when.date().isoformat()}.jsonl"

    def _write_jsonl(self, when: datetime, record: dict[str, Any]) -> None:
        path = self._jsonl_path(when)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _write_sqlite(self, when: datetime, record: dict[str, Any]) -> None:
        with self.db.get_connection() as conn:
            self._insert(conn, when, record)

    @staticmethod
    def _insert(conn: sqlite3.Connection, when: datetime, record: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO audit_log(timestamp, level, message, module, details, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record["timestamp"],
                record["level"],
                record["message"],
                record.get("module"),
                json.dumps(record.get("details") or {}, default=str),
                when.isoformat(),
            ),
        )
