"""APScheduler helpers for windowed execution."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Callable

try:  # pragma: no cover - APScheduler may be absent in minimal test envs
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:  # pragma: no cover
    class BackgroundScheduler:  # type: ignore[override]
        """Minimal fallback scheduler used when APScheduler is unavailable."""

        def __init__(self, timezone=None) -> None:
            self.timezone = timezone
            self._jobs = []

        def add_job(self, func, trigger=None, **kwargs):
            self._jobs.append((func, trigger, kwargs))

        def start(self) -> None:
            return None

        def shutdown(self, wait: bool = False) -> None:
            return None
from dateutil import tz

LOGGER = logging.getLogger(__name__)


def _parse_hhmm(value: str) -> time:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time value: {value}")
    hour, minute = (int(part) for part in parts)
    return time(hour=hour, minute=minute)


def is_within_window(now: datetime, start: str, end: str) -> bool:
    """Return True if ``now`` falls within the [start, end] window (inclusive)."""

    start_time = _parse_hhmm(start)
    end_time = _parse_hhmm(end)
    window_start = now.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
    window_end = now.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)
    if window_end < window_start:
        window_end += timedelta(days=1)
        if now < window_start:
            now = now + timedelta(days=1)
    return window_start <= now <= window_end


def create_scheduler(tz_name: str) -> BackgroundScheduler:
    """Create a background scheduler aware of the configured timezone."""

    tzinfo = tz.gettz(tz_name)
    if tzinfo is None:
        raise ValueError(f"Invalid timezone: {tz_name}")
    scheduler = BackgroundScheduler(timezone=tzinfo)
    return scheduler


def add_windowed_job(
    scheduler: BackgroundScheduler,
    func: Callable[[], None],
    *,
    job_id: str,
    minutes: int,
    start: str,
    end: str,
    tz_name: str,
) -> None:
    """Schedule ``func`` every ``minutes`` while respecting the active window."""

    tzinfo = tz.gettz(tz_name)
    if tzinfo is None:
        raise ValueError(f"Invalid timezone: {tz_name}")

    def _wrapped() -> None:
        now = datetime.now(tzinfo)
        if not is_within_window(now, start, end):
            LOGGER.debug("Skipping %s outside window", job_id)
            return
        LOGGER.info("Executing job %s", job_id)
        func()

    scheduler.add_job(
        _wrapped,
        trigger="interval",
        minutes=minutes,
        id=job_id,
        max_instances=1,
        coalesce=True,
        next_run_time=None,
    )


__all__ = ["create_scheduler", "add_windowed_job", "is_within_window"]
