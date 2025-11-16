import logging
import sys
from typing import Optional


DEFAULT_LOG_FORMAT = "[%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s"
COMPACT_LOG_FORMAT = "[%(levelname)s] - %(message)s"


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
    compact_console: bool = False,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Features:
    - Idempotent (safe to call multiple times without duplicating handlers)
    - Colored console output with shortened paths
    - Optional compact console output
    - Integrates uvicorn loggers to use unified format

    Args:
        level: Explicit base log level (overrides auto-detected).
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

    # Formatter
    if compact_console:
        console_formatter = _ColoredFormatter(COMPACT_LOG_FORMAT)
    else:
        console_formatter = _ColoredFormatter(DEFAULT_LOG_FORMAT)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(target_level)
    console_handler.setFormatter(console_formatter)

    # Attach handler
    root_logger.addHandler(console_handler)

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

class _ColoredFormatter(logging.Formatter):
    """
    Formatter with color support for console output.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Shorten the pathname
        original_pathname = record.pathname
        try:
            import os
            cwd = os.getcwd()
            if record.pathname.startswith(cwd):
                record.pathname = os.path.relpath(record.pathname, cwd)
            if record.pathname.startswith('app/'):
                record.pathname = record.pathname[4:]
        except Exception:
            pass
        
        # Add color to levelname
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelname, '')
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        result = super().format(record)
        
        # Restore originals
        record.pathname = original_pathname
        record.levelname = original_levelname
        
        return result

__all__ = ["setup_logging"]
