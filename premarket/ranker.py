"""Ranking logic for the watchlist."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import utils


@dataclass
class RankerWeights:
    """Weight configuration for scoring."""

    relvol: float
    gap: float
    avgvol: float
    float_band: float
    short_float: float
    after_hours: float
    change: float
    w52pos: float
    news_fresh: float
    analyst: float
    insider_inst: float


@dataclass
class RankerPenalties:
    """Penalty configuration."""

    earnings_near: float
    pe_outlier: float


@dataclass
class RankerCaps:
    """Caps for penalties."""

    max_single_negative: float


@dataclass
class RankerConfig:
    """Container for ranking configuration."""

    weights: RankerWeights
    penalties: RankerPenalties
    caps: RankerCaps
    earnings_window_days: int


_FEATURE_KEYS = [
    "relvol",
    "gap",
    "avgvol",
    "float_band",
    "short_float",
    "after_hours",
    "change",
    "w52pos",
    "news_fresh",
    "analyst",
    "insider_inst",
]


def compute_score(df: pd.DataFrame, cfg: RankerConfig) -> pd.Series:
    """Compute composite scores using configured weights and penalties."""

    scores = pd.Series(0.0, index=df.index, dtype=float)
    for key in _FEATURE_KEYS:
        weight = getattr(cfg.weights, key)
        column = f"f_{key}"
        if column in df.columns:
            scores = scores + df[column].fillna(0.0) * weight

    penalties = pd.Series(0.0, index=df.index, dtype=float)
    today = utils.now_eastern().date()

    if "earnings_date" in df.columns:
        penalties = penalties + _earnings_penalty(df["earnings_date"], today, cfg)

    if "pe" in df.columns:
        pe_penalty = df["pe"].apply(lambda v: cfg.penalties.pe_outlier if (v is None or (isinstance(v, (int, float)) and v > 200)) else 0.0)
        penalties = penalties + pe_penalty
    else:
        penalties = penalties + cfg.penalties.pe_outlier

    capped_penalties = penalties.map(lambda val: min(val, cfg.caps.max_single_negative))
    scores = scores - capped_penalties
    return scores


def _earnings_penalty(series: pd.Series, today, cfg: RankerConfig) -> pd.Series:
    penalties = []
    for value in series:
        if hasattr(value, "date"):
            day_diff = abs((value.date() - today).days)
            if day_diff <= cfg.earnings_window_days:
                penalties.append(cfg.penalties.earnings_near)
            else:
                penalties.append(0.0)
        else:
            penalties.append(0.0)
    return pd.Series(penalties, index=series.index, dtype=float)


def assign_tiers(scores: pd.Series) -> pd.Series:
    """Assign tiers based on score thresholds."""
    def _tier(value: float) -> str:
        if value >= 0.70:
            return "A"
        if value >= 0.55:
            return "B"
        return "C"

    return scores.map(_tier)


def apply_sector_diversity(
    df: pd.DataFrame, top_n: int, max_fraction: float
) -> tuple[pd.DataFrame, bool]:
    """Apply sector diversity cap returning a limited DataFrame and a trim flag."""

    trimmed = False
    if top_n <= 0:
        return df.head(0), trimmed
    if max_fraction <= 0:
        return df.head(top_n), trimmed
    max_per_sector = max(1, math.floor(top_n * max_fraction))
    counts: dict[str, int] = {}
    selected_indices: list[int] = []

    for idx, row in df.iterrows():
        sector = row.get("sector")
        if pd.isna(sector) or sector is None or sector == "":
            selected_indices.append(idx)
        else:
            key = str(sector)
            current = counts.get(key, 0)
            if current < max_per_sector:
                counts[key] = current + 1
                selected_indices.append(idx)
            else:
                trimmed = True
        if len(selected_indices) >= top_n:
            break

    return df.loc[selected_indices], trimmed
