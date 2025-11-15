import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
COMPACT_LOG_FORMAT = "%(levelname)s: %(message)s"


def _get_log_level() -> int:
    """
    Determine base log level.
    Tries to read settings.DEBUG if available, otherwise defaults to INFO.
    """
    try:
        from app.config.settings import settings  # type: ignore
        return logging.DEBUG if getattr(settings, "DEBUG", False) else logging.INFO
    except Exception:
        return logging.INFO


def setup_logging(
    *,
    level: Optional[int] = None,
    json: bool = False,
    file_path: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    compact_console: bool = False,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Features:
    - Idempotent (safe to call multiple times without duplicating handlers)
    - Rotating file handler (size-based)
    - Optional JSON-style minimal logging (no external dependency)
    - Optional compact console output
    - Integrates uvicorn loggers to use unified format

    Args:
        level: Explicit base log level (overrides auto-detected).
        json: If True, emit simplified JSON lines for file handler.
        file_path: Path for rotating file logs.
        max_bytes: Max size before rotation.
        backup_count: Number of rotated backups retained.
        compact_console: Use a minimal console format.

    Returns:
        logging.Logger: Root logger instance.
    """
    root_logger = logging.getLogger()
    target_level = level if level is not None else _get_log_level()
    root_logger.setLevel(target_level)

    # Prevent duplicate handlers if called again
    if getattr(root_logger, "_app_logging_configured", False):
        return root_logger

    # Formatter(s)
    if compact_console:
        console_formatter = logging.Formatter(COMPACT_LOG_FORMAT)
    else:
        console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    if json:
        file_formatter = _JsonLogFormatter()
    else:
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(target_level)
    console_handler.setFormatter(console_formatter)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(target_level)
    file_handler.setFormatter(file_formatter)

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Tune noisy third-party loggers if needed
    for noisy in ["urllib3", "asyncio", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Uvicorn integration (if running under uvicorn)
    for uv_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uv_logger = logging.getLogger(uv_logger_name)
        uv_logger.handlers = []
        uv_logger.propagate = True
        uv_logger.setLevel(target_level)

    root_logger._app_logging_configured = True  # type: ignore[attr-defined]
    return root_logger


class _JsonLogFormatter(logging.Formatter):
    """
    Minimal JSON formatter (no external packages).
    Produces one JSON object per line for easier ingestion.
    """
    def format(self, record: logging.LogRecord) -> str:
        import json
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


__all__ = ["setup_logging"]
