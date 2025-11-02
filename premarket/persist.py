"""Persistence utilities for writing outputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

from . import utils


SQLITE_DB_PATH = Path("premarket.db")


def write_json(obj: Any, path: Path) -> None:
    """Write a JSON object to disk."""
    utils.ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""
    utils.ensure_directory(path.parent)
    df.to_csv(path, index=False)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS full_watchlist (
            run_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            symbol TEXT,
            company TEXT,
            sector TEXT,
            industry TEXT,
            exchange TEXT,
            market_cap TEXT,
            pe TEXT,
            price TEXT,
            change_pct TEXT,
            gap_pct TEXT,
            volume TEXT,
            avg_volume_3m TEXT,
            rel_volume TEXT,
            float_shares TEXT,
            short_float_pct TEXT,
            after_hours_change_pct TEXT,
            week52_range TEXT,
            week52_pos TEXT,
            earnings_date TEXT,
            analyst_recom TEXT,
            features_json TEXT,
            score TEXT,
            tier TEXT,
            tags_json TEXT,
            rejection_reasons_json TEXT,
            insider_transactions TEXT,
            institutional_transactions TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS top_n (
            run_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            rank INTEGER,
            symbol TEXT,
            score TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            run_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            rank INTEGER,
            symbol TEXT,
            score TEXT,
            tier TEXT,
            gap_pct TEXT,
            rel_volume TEXT,
            tags_json TEXT,
            why TEXT,
            top_feature1 TEXT,
            top_feature2 TEXT,
            top_feature3 TEXT,
            top_feature4 TEXT,
            top_feature5 TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_summary (
            run_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            summary_date TEXT,
            filters_json TEXT,
            timings_json TEXT,
            notes_json TEXT,
            row_counts_json TEXT,
            tiers_json TEXT,
            env_overrides_json TEXT,
            weights_version TEXT,
            csv_hash TEXT,
            sector_cap_applied INTEGER,
            used_cached_csv INTEGER,
            week52_warning_count INTEGER
        )
        """
    )


def _clear_table(conn: sqlite3.Connection, table: str, run_date: str) -> None:
    conn.execute(f"DELETE FROM {table} WHERE run_date = ?", (run_date,))


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _prepare_full_watchlist_rows(
    run_date: str, generated_at: str, records: Iterable[Dict[str, Any]]
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for record in records:
        rows.append(
            (
                run_date,
                generated_at,
                record.get("symbol"),
                record.get("company"),
                record.get("sector"),
                record.get("industry"),
                record.get("exchange"),
                record.get("market_cap"),
                record.get("pe"),
                record.get("price"),
                record.get("change_pct"),
                record.get("gap_pct"),
                record.get("volume"),
                record.get("avg_volume_3m"),
                record.get("rel_volume"),
                record.get("float_shares"),
                record.get("short_float_pct"),
                record.get("after_hours_change_pct"),
                record.get("week52_range"),
                record.get("week52_pos"),
                record.get("earnings_date"),
                record.get("analyst_recom"),
                _json_or_none(record.get("features")),
                record.get("score"),
                record.get("tier"),
                _json_or_none(record.get("tags")),
                _json_or_none(record.get("rejection_reasons")),
                record.get("insider_transactions"),
                record.get("institutional_transactions"),
            )
        )
    return rows


def _prepare_top_n_rows(
    run_date: str, generated_at: str, records: Iterable[Dict[str, Any]]
) -> list[tuple[Any, ...]]:
    return [
        (
            run_date,
            generated_at,
            record.get("rank"),
            record.get("symbol"),
            record.get("score"),
        )
        for record in records
    ]


def _prepare_watchlist_rows(
    run_date: str, generated_at: str, records: Iterable[Dict[str, Any]]
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for record in records:
        rows.append(
            (
                run_date,
                generated_at,
                record.get("rank"),
                record.get("symbol"),
                record.get("score"),
                record.get("tier"),
                record.get("gap_pct"),
                record.get("rel_volume"),
                _json_or_none(record.get("tags")),
                record.get("Why"),
                record.get("TopFeature1"),
                record.get("TopFeature2"),
                record.get("TopFeature3"),
                record.get("TopFeature4"),
                record.get("TopFeature5"),
            )
        )
    return rows


def _prepare_summary_row(
    run_date: str, generated_at: str, summary: Dict[str, Any]
) -> tuple[Any, ...]:
    return (
        run_date,
        generated_at,
        summary.get("date"),
        _json_or_none(summary.get("filters")),
        _json_or_none(summary.get("timings_sec")),
        _json_or_none(summary.get("notes")),
        _json_or_none(summary.get("row_counts")),
        _json_or_none(summary.get("tiers")),
        _json_or_none(summary.get("env_overrides_used")),
        summary.get("weights_version"),
        summary.get("csv_hash"),
        1 if summary.get("sector_cap_applied") else 0,
        1 if summary.get("used_cached_csv") else 0,
        summary.get("week52_warning_count"),
    )


def write_sqlite_outputs(
    run_date: str,
    generated_at: str,
    full_watchlist: list[Dict[str, Any]],
    top_n_records: list[Dict[str, Any]],
    watchlist_records: list[Dict[str, Any]],
    run_summary: Dict[str, Any],
    db_path: Path | str | None = None,
) -> None:
    """Persist run artifacts into a SQLite database for easy sharing."""

    path = Path(db_path) if db_path is not None else SQLITE_DB_PATH
    utils.ensure_directory(path.parent)

    full_rows = _prepare_full_watchlist_rows(run_date, generated_at, full_watchlist)
    top_rows = _prepare_top_n_rows(run_date, generated_at, top_n_records)
    watch_rows = _prepare_watchlist_rows(run_date, generated_at, watchlist_records)
    summary_row = _prepare_summary_row(run_date, generated_at, run_summary)

    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)

        _clear_table(conn, "full_watchlist", run_date)
        if full_rows:
            conn.executemany(
                """
                INSERT INTO full_watchlist (
                    run_date,
                    generated_at,
                    symbol,
                    company,
                    sector,
                    industry,
                    exchange,
                    market_cap,
                    pe,
                    price,
                    change_pct,
                    gap_pct,
                    volume,
                    avg_volume_3m,
                    rel_volume,
                    float_shares,
                    short_float_pct,
                    after_hours_change_pct,
                    week52_range,
                    week52_pos,
                    earnings_date,
                    analyst_recom,
                    features_json,
                    score,
                    tier,
                    tags_json,
                    rejection_reasons_json,
                    insider_transactions,
                    institutional_transactions
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                full_rows,
            )

        _clear_table(conn, "top_n", run_date)
        if top_rows:
            conn.executemany(
                """
                INSERT INTO top_n (
                    run_date,
                    generated_at,
                    rank,
                    symbol,
                    score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                top_rows,
            )

        _clear_table(conn, "watchlist", run_date)
        if watch_rows:
            conn.executemany(
                """
                INSERT INTO watchlist (
                    run_date,
                    generated_at,
                    rank,
                    symbol,
                    score,
                    tier,
                    gap_pct,
                    rel_volume,
                    tags_json,
                    why,
                    top_feature1,
                    top_feature2,
                    top_feature3,
                    top_feature4,
                    top_feature5
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                watch_rows,
            )

        _clear_table(conn, "run_summary", run_date)
        conn.execute(
            """
            INSERT INTO run_summary (
                run_date,
                generated_at,
                summary_date,
                filters_json,
                timings_json,
                notes_json,
                row_counts_json,
                tiers_json,
                env_overrides_json,
                weights_version,
                csv_hash,
                sector_cap_applied,
                used_cached_csv,
                week52_warning_count
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            summary_row,
        )
