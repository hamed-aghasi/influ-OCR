"""
Logging Configuration Module

Centralized logging setup for all processing modules.
Logs are stored in the logs/ directory with rotation.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_log_dir() -> Path:
    """Get the log directory path."""
    # Check environment variable first
    log_dir = os.getenv('LOG_DIR')
    if log_dir:
        return Path(log_dir)

    # Default to logs/ in the app directory
    app_dir = Path(__file__).parent.parent
    return app_dir / 'logs'


def setup_logger(
    name: str,
    job_id: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.

    Args:
        name: Logger name (usually module name)
        job_id: Optional job ID for job-specific log files
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    log_dir = get_log_dir()
    log_dir.mkdir(exist_ok=True, parents=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler - rotating log file
    if job_id:
        log_file = log_dir / f"{name}_{job_id}.log"
    else:
        log_file = log_dir / f"{name}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (always enabled for visibility in container logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


# Create main app logger
app_logger = setup_logger('instagram_analyzer')
