"""Postgres connection pool. _pool is rebound via `global` here and must not be
re-exported by the package."""

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from ..config import settings
from ..logger import get_logger

logger = get_logger("processing.db_client")   # same name as the flat module: same Logger

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        if not settings.database_configured:
            return None
        from psycopg2.pool import ThreadedConnectionPool

        _pool = ThreadedConnectionPool(
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_max,
            dsn=settings.database_url,
        )
        logger.info("Postgres pool initialized (min=%d max=%d)", settings.db_pool_min, settings.db_pool_max)
        return _pool


@contextmanager
def _conn() -> Iterator[Any]:
    pool = _get_pool()
    if pool is None:
        raise RuntimeError("Database not configured")
    connection = pool.getconn()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        pool.putconn(connection)


def is_database_available() -> bool:
    return settings.database_configured
