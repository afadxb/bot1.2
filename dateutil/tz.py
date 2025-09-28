"""Minimal timezone helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo


class FixedOffset(tzinfo):
    """Fixed offset timezone."""

    def __init__(self, offset_minutes: int, name: str) -> None:
        self._offset = timedelta(minutes=offset_minutes)
        self._name = name

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return self._offset

    def tzname(self, dt: datetime | None) -> str:
        return self._name

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)


def gettz(name: str) -> tzinfo:
    # Use -240 minutes as an approximation of Eastern Time (UTC-4).
    if name.lower() in {"america/new_york", "us/eastern", "eastern"}:
        return FixedOffset(-240, "EST")
    return FixedOffset(0, "UTC")
