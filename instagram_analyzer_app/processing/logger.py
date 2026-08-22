"""Logging setup.

`get_logger(name)` returns a plain module logger. `job_logger(name, job_id)`
returns a `LoggerAdapter` that decorates every record with `[job_id]` so
concurrent jobs stay distinguishable without any module-level mutation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import settings


_handlers_installed = False


def _install_handlers(level: int) -> None:
    global _handlers_installed
    if _handlers_installed:
        return

    log_dir = Path(os.getenv("LOG_DIR") or settings.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_dir / "instagram_analyzer.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    root = logging.getLogger("instagram_analyzer")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.propagate = False

    _handlers_installed = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    _install_handlers(level)
    if not name.startswith("instagram_analyzer"):
        name = f"instagram_analyzer.{name}"
    return logging.getLogger(name)


class _JobAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        job_id = self.extra.get("job_id") if self.extra else None
        return (f"[{job_id}] {msg}" if job_id else msg), kwargs


def job_logger(name: str, job_id: Optional[str] = None) -> logging.LoggerAdapter:
    return _JobAdapter(get_logger(name), {"job_id": job_id})
