"""Minimal numpy-like utilities required by the project."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

nan = float("nan")


class ndarray(list):
    """Lightweight ndarray replacement."""

    pass


def array(data: Iterable, dtype: str | None = None):
    return ndarray(data)


def clip(values: Iterable[float] | float, a_min: float | None = None, a_max: float | None = None):
    try:
        iterator = iter(values)  # type: ignore[arg-type]
    except TypeError:
        return clip_scalar(values, a_min, a_max)  # type: ignore[arg-type]

    result = []
    for value in iterator:  # type: ignore[assignment]
        val = value
        if not isinstance(val, (int, float)) or math.isnan(val):
            result.append(val)
            continue
        if a_min is not None and val < a_min:
            val = a_min
        if a_max is not None and val > a_max:
            val = a_max
        result.append(val)
    return result


def log10(value: float) -> float:
    return math.log10(value)


def log(value: float) -> float:
    return math.log(value)


def exp(value: float) -> float:
    return math.exp(value)


def nanpercentile(values: Sequence[float], q: float) -> float:
    clean = sorted(v for v in values if isinstance(v, (int, float)) and not math.isnan(v))
    if not clean:
        return nan
    k = (len(clean) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return clean[int(k)]
    d0 = clean[int(f)] * (c - k)
    d1 = clean[int(c)] * (k - f)
    return d0 + d1


def isnan(value: float) -> bool:
    return math.isnan(value)


def nanmin(values: Sequence[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return min(clean) if clean else nan


def nanmax(values: Sequence[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return max(clean) if clean else nan


def clip_scalar(value: float, lower: float | None, upper: float | None) -> float:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return value
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


def vector_clip(values: Sequence[float], lower: float | None, upper: float | None) -> List[float]:
    return [clip_scalar(v, lower, upper) for v in values]


def clip_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def sqrt(value: float) -> float:
    return math.sqrt(value)

def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

__all__ = [
    "nan",
    "array",
    "clip",
    "log10",
    "log",
    "exp",
    "nanpercentile",
    "isnan",
    "nanmin",
    "nanmax",
    "clip_scalar",
    "vector_clip",
    "clip_value",
    "sqrt",
    "sign",
]
