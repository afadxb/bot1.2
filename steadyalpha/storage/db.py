"""MySQL connection pool management for the Screener bot."""

from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - mysql connector may be absent in tests
    from mysql.connector import pooling
except ImportError:  # pragma: no cover
    pooling = None

from steadyalpha.settings import get_settings

LOGGER = logging.getLogger(__name__)

_POOL: Optional[pooling.MySQLConnectionPool] = None


def init_pool(min_size: int = 1, max_size: int = 5) -> None:
    """Initialise the global MySQL connection pool."""

    global _POOL
    settings = get_settings()
    if _POOL is not None:
        return

    if pooling is None:
        raise RuntimeError("mysql-connector-python is required for database access")

    LOGGER.info(
        "Initialising MySQL pool to %s:%s/%s", settings.db_host, settings.db_port, settings.db_name
    )
    _POOL = pooling.MySQLConnectionPool(
        pool_name="steadyalpha_pool",
        pool_size=max_size,
        pool_reset_session=True,
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        charset="utf8mb4",
        autocommit=False,
    )


def get_connection():
    """Fetch a connection from the pool, initialising it if needed."""

    global _POOL
    if _POOL is None:
        init_pool()
    if _POOL is None:  # pragma: no cover - defensive
        raise RuntimeError("MySQL connection pool is not initialised")
    return _POOL.get_connection()


__all__ = ["init_pool", "get_connection"]
