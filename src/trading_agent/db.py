import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager


class DatabaseManager:
    """SQLite database connection manager with WAL mode support"""

    def __init__(self, db_path: str = "trading_agent.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA foreign_keys=ON")
        return self._local.connection

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections"""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def _initialize_database(self) -> None:
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Create trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT UNIQUE,
                    status TEXT NOT NULL,
                    strategy TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL,
                    strategy TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)

            # Create audit_log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
                ON trades(symbol, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp
                ON signals(symbol, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
                ON audit_log(timestamp)
            """)


# Global database manager instance
db_manager: DatabaseManager | None = None


def get_db_manager(db_path: str = "trading_agent.db") -> DatabaseManager:
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(db_path)
    return db_manager


def init_db(db_path: str = "trading_agent.db") -> DatabaseManager:
    """Initialize database and return manager instance"""
    return get_db_manager(db_path)
