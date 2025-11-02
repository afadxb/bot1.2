"""MySQL schema management for the SteadyAlpha Screener."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

try:  # pragma: no cover - mysql connector may be absent in tests
    from mysql.connector import MySQLConnection
except ImportError:  # pragma: no cover
    from typing import Any

    MySQLConnection = Any  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _ensure_table(cursor, statement: str) -> None:
    LOGGER.debug("Ensuring table with statement: %s", statement)
    cursor.execute(statement)


def _ensure_columns(cursor, table: str, columns: Iterable[tuple[str, str]]) -> None:
    cursor.execute("SHOW COLUMNS FROM `{}`".format(table))
    existing = {row[0] for row in cursor.fetchall()}
    for name, ddl in columns:
        if name in existing:
            continue
        LOGGER.info("Adding column %s.%s", table, name)
        cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{name}` {ddl}")


def _ensure_index(cursor, table: str, index_sql: str) -> None:
    cursor.execute("SHOW INDEX FROM `{}`".format(table))
    names = {row[2] for row in cursor.fetchall()}
    idx_name = index_sql.split(" ")[2].strip("`")
    if idx_name in names:
        return
    LOGGER.info("Creating index %s on %s", idx_name, table)
    cursor.execute(index_sql)


def apply_migrations(conn: MySQLConnection) -> None:
    """Create or upgrade schema objects in an idempotent manner."""

    with conn.cursor() as cursor:
        _ensure_table(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS `schema_version` (
                `version` INT PRIMARY KEY,
                `applied_at` DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        )

        _ensure_table(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS `run_summary` (
                `run_id` VARCHAR(26) PRIMARY KEY,
                `started_at` DATETIME NOT NULL,
                `finished_at` DATETIME NOT NULL,
                `notes` VARCHAR(255)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        )

        _ensure_columns(
            cursor,
            "run_summary",
            [("notes", "VARCHAR(255)")],
        )

        _ensure_table(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS `candidates` (
                `run_id` VARCHAR(26) NOT NULL,
                `symbol` VARCHAR(16) NOT NULL,
                `gap_pct` DECIMAL(8,3) NOT NULL,
                `pre_mkt_vol` INT NOT NULL,
                `catalyst_flag` TINYINT NOT NULL,
                `pm_high` DECIMAL(16,6),
                `pm_low` DECIMAL(16,6),
                `prev_high` DECIMAL(16,6),
                `prev_low` DECIMAL(16,6),
                `pm_vwap` DECIMAL(16,6),
                `tags` VARCHAR(255),
                `created_at` DATETIME NOT NULL,
                PRIMARY KEY (`run_id`, `symbol`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        )

        _ensure_columns(
            cursor,
            "candidates",
            [
                ("pm_high", "DECIMAL(16,6)"),
                ("pm_low", "DECIMAL(16,6)"),
                ("prev_high", "DECIMAL(16,6)"),
                ("prev_low", "DECIMAL(16,6)"),
                ("pm_vwap", "DECIMAL(16,6)"),
                ("tags", "VARCHAR(255)"),
            ],
        )

        _ensure_index(
            cursor,
            "candidates",
            "CREATE INDEX `idx_symbol` ON `candidates` (`symbol`)",
        )

        cursor.execute("SELECT version FROM schema_version WHERE version = %s", (SCHEMA_VERSION,))
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (%s, %s)",
                (SCHEMA_VERSION, datetime.utcnow()),
            )

    conn.commit()


__all__ = ["apply_migrations", "SCHEMA_VERSION"]
