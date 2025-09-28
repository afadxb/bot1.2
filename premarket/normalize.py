"""Normalization helpers for Finviz data."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from dateutil import parser

from . import utils

COLUMN_ALIASES: Dict[str, str] = {}

_CANONICAL_MAP = {
    "ticker": ["ticker", "symbol"],
    "company": ["company", "name"],
    "sector": ["sector"],
    "industry": ["industry"],
    "exchange": ["exchange"],
    "country": ["country"],
    "market_cap": ["market cap", "market capitalization"],
    "pe": ["p/e", "pe", "pe ratio"],
    "price": ["price", "last"],
    "change_pct": ["change", "% change", "change %"],
    "gap_pct": ["gap", "gap %", "% gap"],
    "volume": ["volume"],
    "avg_volume_3m": ["average volume", "average volume (3m)", "avg volume", "avg volume (3m)"],
    "rel_volume": ["relative volume", "rel volume", "relative vol."],
    "float_shares": ["float", "float shares", "shares float", "float/shares", "float/outstanding"],
    "float_pct": ["float %", "float pct", "float percentage", "float_%"],
    "short_float_pct": ["short float", "short float %"],
    "after_hours_change_pct": ["after-hours change", "after hours change", "after-hours change %"],
    "week52_range": ["52-week range", "52w range"],
    "earnings_date": ["earnings date"],
    "analyst_recom": ["analyst recom.", "analyst recommendation"],
    "insider_transactions": ["insider transactions"],
    "institutional_transactions": ["institutional transactions"],
    "previous_close": ["previous close", "prev close"],
}

for canonical, aliases in _CANONICAL_MAP.items():
    for alias in aliases:
        COLUMN_ALIASES[alias.lower()] = canonical


FLOAT_COLUMNS = {
    "price",
    "change_pct",
    "gap_pct",
    "rel_volume",
    "market_cap",
    "pe",
    "after_hours_change_pct",
    "short_float_pct",
    "float_pct",
}

INT_COLUMNS = {
    "volume",
    "avg_volume_3m",
    "float_shares",
}

PERCENT_COLUMNS = {
    "change_pct",
    "gap_pct",
    "short_float_pct",
    "after_hours_change_pct",
    "float_pct",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical keys."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        canonical = COLUMN_ALIASES.get(key)
        if canonical is None:
            canonical = key.replace(" ", "_")
        rename_map[col] = canonical
    normalized = df.rename(columns=rename_map)
    return normalized


def _parse_datetime(value: object) -> object:
    if value in (None, "", "-"):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return parser.parse(str(value))
    except (parser.ParserError, TypeError, ValueError):
        return None


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce numeric and date columns into typed values."""
    result = df.copy()
    warnings = 0

    for col in set(result.columns) & FLOAT_COLUMNS:
        if col in PERCENT_COLUMNS:
            result[col] = result[col].map(utils.safe_percent)
        else:
            result[col] = result[col].map(utils.safe_float)

    for col in set(result.columns) & INT_COLUMNS:
        result[col] = result[col].map(utils.safe_int)

    if "earnings_date" in result.columns:
        result["earnings_date"] = result["earnings_date"].map(_parse_datetime)

    if "previous_close" in result.columns:
        result["previous_close"] = result["previous_close"].map(utils.safe_float)

    if "price" in result.columns and "gap_pct" not in result.columns:
        result["gap_pct"] = np.nan

    if "gap_pct" in result.columns:
        result["gap_pct"] = result["gap_pct"].map(utils.safe_percent)

    if "gap_pct" in result.columns and result["gap_pct"].isna().all():
        if "price" in result.columns and "previous_close" in result.columns:
            price_series = result["price"].map(utils.safe_float)
            prev_series = result["previous_close"].map(utils.safe_float)
            computed: list[float] = []
            for price_value, prev_value in zip(price_series, prev_series):
                if (
                    price_value is not None
                    and prev_value is not None
                    and prev_value != 0
                ):
                    computed.append((price_value - prev_value) / prev_value * 100)
                else:
                    computed.append(np.nan)
            result["gap_pct"] = pd.Series(computed, index=result.index)

    if "week52_range" in result.columns:
        result["week52_range"] = result["week52_range"].astype(str)
        week_pos, warnings = compute_week52_pos(result)
        result["week52_pos"] = week_pos
    else:
        result["week52_pos"] = np.nan

    return result, warnings


def compute_week52_pos(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Compute the position of price within the 52-week range."""
    prices = df.get("price")
    ranges = df.get("week52_range")
    if prices is None or ranges is None:
        return pd.Series(np.nan, index=df.index), 0

    positions: list[float] = []
    warnings = 0
    for _, row in df.iterrows():
        price = utils.safe_float(row.get("price"))
        range_str = row.get("week52_range")
        parsed = utils.parse_range(range_str)
        if price is None or parsed is None:
            warnings += 1
            positions.append(0.5)
            continue
        low, high = parsed
        if high <= low:
            warnings += 1
            positions.append(0.5)
            continue
        position = (price - low) / (high - low)
        positions.append(float(np.clip(position, 0.0, 1.0)))

    return pd.Series(positions, index=df.index, dtype=float), warnings
