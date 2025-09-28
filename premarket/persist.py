"""Persistence utilities backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import utils


@contextmanager
def _connect(db_path: Path):
    """Yield a SQLite connection ensuring the parent directory exists."""
    utils.ensure_directory(db_path.parent)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _encode_json_columns(df: pd.DataFrame, candidate_columns: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df
    encoded = df.copy()
    for column in candidate_columns:
        if column not in encoded.columns:
            continue
        encoded[column] = encoded[column].apply(_jsonify_value)
    return encoded


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    columns = df.columns
    if not columns:
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (dummy INTEGER)')
        conn.execute(f'DELETE FROM "{name}"')
        return
    schema = ", ".join(f'"{col}" TEXT' for col in columns)
    conn.execute(f'CREATE TABLE "{name}" ({schema})')
    if df.empty:
        return
    placeholders = ", ".join(["?"] * len(columns))
    column_list = ", ".join(f'"{col}"' for col in columns)
    insert_sql = f'INSERT INTO "{name}" ({column_list}) VALUES ({placeholders})'
    rows = []
    for row_idx in range(len(df.index)):
        row_values = []
        for col in columns:
            series = df[col]
            value = series._data[row_idx] if hasattr(series, "_data") else None
            row_values.append(value)
        rows.append(row_values)
    conn.executemany(insert_sql, rows)


def write_outputs(
    db_path: Path,
    *,
    watchlist: pd.DataFrame,
    top_rankings: pd.DataFrame,
    full_watchlist: pd.DataFrame,
    metadata: dict[str, Any],
    rejections: pd.DataFrame | None = None,
    run_summary: dict[str, Any] | None = None,
) -> None:
    """Persist workflow outputs into a SQLite database."""

    watchlist_df = _encode_json_columns(watchlist, ["tags"])
    top_rankings_df = top_rankings.copy()
    full_watchlist_df = _encode_json_columns(full_watchlist, ["features", "tags", "rejection_reasons"])
    rejections_df = _encode_json_columns(rejections, ["rejection_reasons"]) if rejections is not None else None

    run_summary_df = (
        pd.DataFrame([
            {
                "payload": json.dumps(run_summary),
            }
        ])
        if run_summary is not None
        else None
    )
    metadata_df = pd.DataFrame([metadata])
    metadata_df = _encode_json_columns(metadata_df, metadata.keys())

    with _connect(db_path) as conn:
        _write_table(conn, "watchlist", watchlist_df)
        _write_table(conn, "top_rankings", top_rankings_df)
        _write_table(conn, "full_watchlist", full_watchlist_df)
        _write_table(conn, "metadata", metadata_df)
        if rejections_df is not None:
            _write_table(conn, "rejections", rejections_df)
        else:
            _write_table(conn, "rejections", pd.DataFrame())
        if run_summary_df is not None:
            _write_table(conn, "run_summary", run_summary_df)


def write_run_summary(db_path: Path, run_summary: dict[str, Any]) -> None:
    run_summary_df = pd.DataFrame([
        {
            "payload": json.dumps(run_summary),
        }
    ])
    with _connect(db_path) as conn:
        _write_table(conn, "run_summary", run_summary_df)


__all__ = ["write_outputs", "write_run_summary"]
