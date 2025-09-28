"""Heuristics that transform raw news metadata into actionable AI signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

DEFAULT_SOURCE_PRIORS: Mapping[str, float] = {
    "finnhub": 1.0,
    "finviz": 0.85,
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class NewsMetadata:
    """Normalized representation of a single symbol's news metadata."""

    freshness_hours: float | None
    source: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NewsMetadata":
        freshness_raw = payload.get("freshness_hours") if isinstance(payload, Mapping) else None
        source_raw = payload.get("category") if isinstance(payload, Mapping) else None
        freshness = _coerce_float(freshness_raw)
        source = str(source_raw).strip().lower() if source_raw is not None else None
        return cls(freshness, source or None)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_signal(metadata: Mapping[str, object] | NewsMetadata, *, horizon_hours: float) -> float:
    """Return an AI-derived score in ``[0, 1]`` for a single news payload."""

    if not isinstance(metadata, NewsMetadata):
        meta = NewsMetadata.from_payload(metadata)
    else:
        meta = metadata

    horizon = _sanitize_horizon(horizon_hours)
    if meta.freshness_hours is None:
        return 0.0

    recency_ratio = max(0.0, 1.0 - min(max(meta.freshness_hours, 0.0), horizon) / horizon)
    recency_curve = _sigmoid(recency_ratio * 8 - 4)

    source_score = DEFAULT_SOURCE_PRIORS.get(meta.source or "", 0.75)

    burst_bonus = _sigmoid(((horizon - min(meta.freshness_hours, horizon)) / horizon) * 6 - 3)

    score = recency_curve * 0.65 + source_score * 0.25 + burst_bonus * 0.10
    return max(0.0, min(score, 1.0))


def score_batch(payloads: Mapping[str, Mapping[str, object]], *, horizon_hours: float) -> dict[str, float]:
    """Score a batch of news payloads keyed by ticker."""

    horizon = _sanitize_horizon(horizon_hours)
    scores: dict[str, float] = {}
    for symbol, payload in payloads.items():
        score = score_signal(payload, horizon_hours=horizon)
        scores[symbol] = score
    return scores


def summarize_scores(scores: Mapping[str, float], *, strong_threshold: float = 0.6) -> dict[str, object]:
    """Generate lightweight analytics about a batch of news scores."""

    if not scores:
        return {"average": 0.0, "strong_symbols": [], "top_signals": [], "leader": None}

    sanitized: dict[str, float] = {}
    for symbol, value in scores.items():
        try:
            sanitized[symbol] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            sanitized[symbol] = 0.0

    values = list(sanitized.values())
    average = sum(values) / len(values) if values else 0.0
    sorted_signals = sorted(sanitized.items(), key=lambda item: item[1], reverse=True)
    top_signals = [
        {"symbol": symbol, "score": round(score, 3)} for symbol, score in sorted_signals[:3]
    ]
    strong_symbols = [symbol for symbol, score in sorted_signals if score >= strong_threshold]
    leader = top_signals[0] if top_signals else None
    return {
        "average": round(average, 3),
        "strong_symbols": strong_symbols,
        "top_signals": top_signals,
        "leader": leader,
    }


def _sanitize_horizon(horizon_hours: float) -> float:
    try:
        horizon = float(horizon_hours)
    except (TypeError, ValueError):
        horizon = 24.0
    if horizon <= 0:
        horizon = 24.0
    return horizon


__all__ = ["score_signal", "score_batch", "summarize_scores", "NewsMetadata"]
