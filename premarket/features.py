"""Feature engineering for the watchlist."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import utils


def winsorize_and_scale(series: pd.Series) -> pd.Series:
    """Winsorize (1-99 pct) and scale to [0,1]."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    lower = np.nanpercentile(numeric, 1)
    upper = np.nanpercentile(numeric, 99)
    clipped = numeric.clip(lower=lower, upper=upper)
    min_val = float(np.nanmin(clipped))
    max_val = float(np.nanmax(clipped))
    if math.isclose(max_val, min_val):
        return pd.Series(0.0, index=series.index, dtype=float)
    scaled = (clipped - min_val) / (max_val - min_val)
    return scaled.fillna(0.0)


def _float_band_score(value: Any) -> float:
    shares = utils.safe_float(value)
    if shares is None or shares <= 0:
        return 0.0
    log_val = math.log(shares)
    log_mid = math.log(100_000_000)
    sigma = math.log(300_000_000 / 20_000_000) / 2
    score = math.exp(-((log_val - log_mid) ** 2) / (2 * sigma**2))
    if shares < 20_000_000:
        score *= 0.7
    return float(np.clip(score, 0.0, 1.0))


def _gap_score(gap: Any) -> float:
    value = utils.safe_percent(gap)
    if value is None:
        return 0.5
    if value < 0:
        return 0.0
    base = np.clip((value - 2) / 8, 0, 1)
    if value > 20:
        base = min(base, 0.7)
    return float(base)


def _after_hours_score(change: Any, day_change: Any) -> float:
    ah = utils.safe_percent(change)
    if ah is None:
        return 0.5
    day = utils.safe_percent(day_change)
    if day is None:
        day = 0.0
    same_direction = np.sign(ah) == np.sign(day)
    magnitude = min(abs(ah) / 5, 1)
    return float(np.clip(0.5 + (0.3 if same_direction else -0.2) + 0.2 * magnitude, 0.0, 1.0))


def _short_float_score(value: Any) -> float:
    pct = utils.safe_percent(value)
    if pct is None:
        return 0.5

    lower = 5.0
    upper = 20.0
    sweet_mid = (lower + upper) / 2
    half_width = (upper - lower) / 2

    if pct < lower:
        return float(np.clip(pct / lower * 0.5, 0.0, 0.5))
    if pct > upper:
        decay = max(0.0, 1 - (pct - upper) / 20)
        return float(np.clip(decay * 0.6, 0.0, 1.0))

    distance = abs(pct - sweet_mid)
    normalized = 1 - (distance / half_width) ** 2
    return float(np.clip(normalized, 0.0, 1.0))


def _analyst_score(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).lower()
    if "strong" in text and "buy" in text:
        return 1.0
    if "buy" in text:
        return 0.8
    if "outperform" in text or "overweight" in text:
        return 0.7
    if "hold" in text or "neutral" in text:
        return 0.4
    if "sell" in text or "underperform" in text:
        return 0.1
    return 0.0


def _insider_inst_score(insider: Any, institutional: Any) -> float:
    insider_pct = utils.safe_percent(insider) or 0.0
    inst_pct = utils.safe_percent(institutional) or 0.0
    score = 0.0
    if insider_pct > 0:
        score += 0.6
    if inst_pct > 0:
        score += 0.4
    return float(np.clip(score, 0.0, 1.0))


def build_features(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Build feature columns prefixed with f_."""

    result = df.copy()
    index = result.index

    relvol = pd.to_numeric(result.get("rel_volume", pd.Series(np.nan, index=index)), errors="coerce")
    result["f_relvol"] = winsorize_and_scale(relvol)

    avg_vol_series = pd.to_numeric(
        result.get("avg_volume_3m", pd.Series(np.nan, index=index)), errors="coerce"
    )
    log_values = []
    for value in avg_vol_series:
        if value is None or value <= 0 or (isinstance(value, float) and math.isnan(value)):
            log_values.append(np.nan)
        else:
            log_values.append(math.log10(value))
    log_avg = pd.Series(log_values, index=index)
    result["f_avgvol"] = winsorize_and_scale(log_avg)

    floats = result.get("float_shares", pd.Series(np.nan, index=index))
    result["f_float_band"] = floats.map(_float_band_score)

    gap = result.get("gap_pct", pd.Series(np.nan, index=index))
    result["f_gap"] = gap.map(_gap_score)

    change = result.get("change_pct", pd.Series(np.nan, index=index))
    change_scaled = winsorize_and_scale(change)
    result["f_change"] = change_scaled

    after_hours = result.get("after_hours_change_pct", pd.Series(np.nan, index=index))
    result["f_after_hours"] = [
        _after_hours_score(ah, ch)
        for ah, ch in zip(after_hours, change)
    ]

    week_pos = pd.to_numeric(result.get("week52_pos", pd.Series(np.nan, index=index)), errors="coerce")
    result["f_52w_pos"] = week_pos.fillna(0.0)

    short_float = result.get("short_float_pct", pd.Series(np.nan, index=index))
    result["f_short_float"] = short_float.map(_short_float_score)

    analyst = result.get("analyst_recom", pd.Series(None, index=index))
    result["f_analyst"] = analyst.map(_analyst_score)

    insider = result.get("insider_transactions", pd.Series(None, index=index))
    inst = result.get("institutional_transactions", pd.Series(None, index=index))
    result["f_insider_inst"] = [
        _insider_inst_score(i, j) for i, j in zip(insider, inst)
    ]

    news = result.get("news_fresh_score", pd.Series(0.0, index=index))
    result["f_news_fresh"] = pd.to_numeric(news, errors="coerce").fillna(0.0)

    after_hours_pct = pd.to_numeric(after_hours, errors="coerce").fillna(0.0)
    result["after_hours_change_pct"] = after_hours_pct

    price = pd.to_numeric(result.get("price", pd.Series(np.nan, index=index)), errors="coerce")
    avg_volume = avg_vol_series.fillna(0.0)
    result["turnover_dollar"] = (price.fillna(0.0) * avg_volume.fillna(0.0)).astype(float)

    return result
