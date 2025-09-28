"""Filtering of candidate securities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Tuple

import pandas as pd

from . import utils


@dataclass
class FilterConfig:
    """Configuration required for filtering."""

    price_min: float
    price_max: float
    avg_vol_min: int
    rel_vol_min: float
    float_min: int
    earnings_exclude_window_days: int
    exclude_exchanges: Iterable[str]
    exclude_countries: Iterable[str]


_DEF_EXCHANGES: Tuple[str, ...] = ("OTC",)


def _should_exclude(value: Any, collection: Iterable[str]) -> bool:
    values = {v.upper() for v in collection}
    if not values:
        return False
    if value is None:
        return False
    return str(value).upper() in values


def apply_hard_filters(df: pd.DataFrame, cfg: FilterConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hard qualification rules returning qualified and rejected frames."""

    qualified_records = []
    rejected_records = []
    now = utils.now_eastern().date()

    exclude_exchanges = tuple(cfg.exclude_exchanges) or _DEF_EXCHANGES
    exclude_countries = tuple(cfg.exclude_countries)

    for _, row in df.iterrows():
        record = row.to_dict()
        reasons: list[str] = []

        price = record.get("price")
        if price is None:
            reasons.append("missing_price")
        else:
            if price < cfg.price_min:
                reasons.append("price_below_min")
            if price > cfg.price_max:
                reasons.append("price_above_max")

        avg_vol = record.get("avg_volume_3m")
        if avg_vol is None or avg_vol < cfg.avg_vol_min:
            reasons.append("avg_vol_below_min")

        rel_vol = record.get("rel_volume")
        if rel_vol is None or rel_vol < cfg.rel_vol_min:
            reasons.append("relvol_below_min")

        float_shares = record.get("float_shares")
        if float_shares is None or float_shares < cfg.float_min:
            reasons.append("float_below_min")

        exchange = record.get("exchange")
        if _should_exclude(exchange, exclude_exchanges):
            reasons.append("exchange_excluded")

        country = record.get("country")
        if _should_exclude(country, exclude_countries):
            reasons.append("country_excluded")

        earnings_date = record.get("earnings_date")
        if isinstance(earnings_date, datetime):
            day_diff = abs((earnings_date.date() - now).days)
            if day_diff <= cfg.earnings_exclude_window_days:
                reasons.append("earnings_within_window")

        record["rejection_reasons"] = reasons

        if reasons:
            rejected_records.append(record)
        else:
            qualified_records.append(record)

    qualified_df = pd.DataFrame(qualified_records)
    rejected_df = pd.DataFrame(rejected_records)
    return qualified_df, rejected_df
