"""Utilities for scheduling recurring watchlist runs."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time as dtime, timedelta
from typing import Callable, Optional, Sequence

from dateutil import tz

from . import orchestrate, utils


def parse_schedule(spec: str) -> list[dtime]:
    """Parse a comma or semicolon separated schedule string.

    Each entry must be in ``HH:MM`` or ``HH:MM:SS`` 24-hour format. Duplicate
    entries are collapsed and the resulting list is sorted chronologically.
    """

    if spec is None:
        raise ValueError("schedule specification cannot be None")

    parts = [part.strip() for part in spec.replace(";", ",").split(",")]
    times: set[dtime] = set()

    for part in parts:
        if not part:
            continue
        try:
            hour, minute, *rest = [int(piece) for piece in part.split(":")]
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid schedule time: {part}") from exc

        if len(rest) > 1:
            raise ValueError(f"Invalid schedule time: {part}")

        second = rest[0] if rest else 0

        try:
            parsed = dtime(hour=hour, minute=minute, second=second)
        except ValueError as exc:
            raise ValueError(f"Invalid schedule time: {part}") from exc

        times.add(parsed)

    if not times:
        raise ValueError("No valid schedule times provided")

    return sorted(times)


def _resolve_timezone(tz_name: Optional[str]) -> tz.tzinfo:
    resolved = tz.gettz(tz_name) if tz_name else None
    if resolved is None:
        resolved = tz.gettz(utils.DEFAULT_TZ_NAME)
    return resolved


def next_run_datetime(times: Sequence[dtime], now: datetime, tzinfo: Optional[tz.tzinfo] = None) -> datetime:
    """Return the next occurrence for a set of daily times."""

    if not times:
        raise ValueError("times must not be empty")

    tzinfo = tzinfo or now.tzinfo or _resolve_timezone(None)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tzinfo)

    today = now.date()
    for scheduled in times:
        candidate = datetime.combine(today, scheduled, tzinfo)
        if candidate >= now:
            return candidate

    tomorrow = today + timedelta(days=1)
    return datetime.combine(tomorrow, times[0], tzinfo)


def run_schedule(
    params: orchestrate.RunParams,
    times: Sequence[dtime],
    *,
    timezone: Optional[str] = None,
    runs: Optional[int] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> int:
    """Run the orchestrator on a recurring schedule.

    Parameters
    ----------
    params:
        Base ``RunParams`` used for each invocation. The ``run_date`` will be
        updated to match the scheduled occurrence.
    times:
        Iterable of ``datetime.time`` entries indicating the times-of-day to
        trigger a run.
    timezone:
        Optional timezone override. Defaults to ``params.timezone`` and
        ultimately falls back to the project default.
    runs:
        Optional cap on the number of executions before exiting. ``None`` means
        run indefinitely.
    now_fn:
        Callable returning the current time. Primarily for testing.
    sleep_fn:
        Callable accepting the number of seconds to sleep before the next run.
    """

    if runs is not None and runs <= 0:
        return 0

    schedule_times = list(times)
    if not schedule_times:
        raise ValueError("times must not be empty")

    tzinfo = _resolve_timezone(timezone or params.timezone)
    now_provider = now_fn or (lambda: datetime.now(tzinfo))

    import time as _time

    sleeper = sleep_fn or _time.sleep

    executed = 0
    last_code = 0

    while True:
        current = now_provider()
        next_run = next_run_datetime(schedule_times, current, tzinfo)
        wait_seconds = (next_run - current).total_seconds()
        if wait_seconds > 0:
            sleeper(wait_seconds)

        updated_params = replace(params, run_date=next_run.date())
        last_code = orchestrate.run(updated_params)
        executed += 1

        if runs is not None and executed >= runs:
            return last_code

    return last_code

