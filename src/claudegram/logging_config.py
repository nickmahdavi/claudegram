"""Logging setup for claudegram.

Call `setup_logging(log_dir)` once at process startup (e.g., from an entry-point
`main()`). Other modules just do `logger = logging.getLogger(__name__)` and don't
need to care about configuration — it's process-global once dictConfig runs.

This module deliberately does NOT import from `config` — keeps the dependency
direction clean. The caller is responsible for sourcing `log_dir` from wherever
makes sense (env var, CLI flag, hardcoded).
"""

import logging.config
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def setup_logging(log_dir: PathLike = "logs") -> None:
    """Configure process-wide logging. Idempotent — safe to call more than once.

    Creates `log_dir` if missing, then applies handlers:
      - console: INFO+ to stdout
      - file:    DEBUG+ to `<log_dir>/bot.log`, rotated daily, 14 days retention
      - errors:  ERROR+ to `<log_dir>/errors.log`, rotated daily, 30 days retention

    Plus level overrides for a few noisy 3rd-party libraries (httpx, telegram, etc.).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(log_dir / "bot.log"),
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
            },
            "errors": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_dir / "errors.log"),
                "when": "midnight",
                "backupCount": 30,
                "encoding": "utf-8",
            },
        },

        "loggers": {
            "httpx":       {"level": "WARNING", "propagate": True},
            "httpcore":    {"level": "WARNING", "propagate": True},
            "telegram":    {"level": "INFO",    "propagate": True},
            "apscheduler": {"level": "WARNING", "propagate": True},
        },

        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file", "errors"],
        },
    }

    logging.config.dictConfig(config)
