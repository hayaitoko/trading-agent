import json
import logging
import logging.handlers
import os
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_object = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ('name', 'msg', 'args', 'levelname', 'levelno',
                              'pathname', 'filename', 'module', 'lineno',
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'exc_info', 'exc_text', 'stack_info', 'getMessage'):
                    log_object[key] = value

        return json.dumps(log_object, default=str)


def setup_logging(data_dir: str = "data", log_level: int = logging.INFO) -> None:
    """
    Set up structured JSON logging to stdout and daily rotating JSONL files.

    Args:
        data_dir: Directory where log files will be stored
        log_level: Logging level (default: INFO)
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create JSON formatter
    json_formatter = JSONFormatter()

    # Set up stdout handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)

    # Set up daily rotating file handler
    log_file_path = os.path.join(data_dir, "trading_agent.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding="utf-8"
    )
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    # Configure library loggers to use the same format
    for logger_name in ["urllib3", "requests", "asyncio"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False


# Convenience function for getting a configured logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
