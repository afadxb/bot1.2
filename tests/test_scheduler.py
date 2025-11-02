from __future__ import annotations

from datetime import date, datetime, time as dtime

import pytest
from dateutil import tz

from premarket import orchestrate, scheduler


def test_parse_schedule_returns_sorted_unique_times():
    parsed = scheduler.parse_schedule("07:30,05:15;07:30,09:00")
    assert [t.strftime("%H:%M:%S") for t in parsed] == [
        "05:15:00",
        "07:30:00",
        "09:00:00",
    ]


@pytest.mark.parametrize("spec", ["", "abc", "25:00", "10:61", "07:30:30:01"])
def test_parse_schedule_invalid_values_raise(spec):
    with pytest.raises(ValueError):
        scheduler.parse_schedule(spec)


def test_next_run_datetime_rolls_to_next_day():
    tzinfo = tz.gettz("UTC")
    times = [dtime(hour=7, minute=30), dtime(hour=8, minute=0)]
    now = datetime(2024, 1, 1, 9, 0, tzinfo=tzinfo)

    upcoming = scheduler.next_run_datetime(times, now, tzinfo)

    assert upcoming.date() == date(2024, 1, 2)
    assert upcoming.time() == times[0]


def test_run_schedule_executes_requested_runs(monkeypatch, tmp_path):
    tz_name = "UTC"
    tzinfo = tz.gettz(tz_name)
    times = [dtime(hour=7, minute=30), dtime(hour=8, minute=0)]

    now_values = iter(
        [
            datetime(2024, 1, 1, 7, 0, tzinfo=tzinfo),
            datetime(2024, 1, 1, 7, 35, tzinfo=tzinfo),
        ]
    )

    sleep_calls: list[float] = []
    run_dates: list[date] = []

    def fake_now():
        return next(now_values)

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def fake_run(params):
        run_dates.append(params.run_date)
        return 0

    monkeypatch.setattr(orchestrate, "run", fake_run)

    params = orchestrate.RunParams(
        config_path=tmp_path / "cfg.yaml",
        output_base_dir=tmp_path / "out",
        top_n=5,
        use_cache=True,
        news_override=None,
        log_file=None,
        run_date=date(2024, 1, 1),
        timezone=tz_name,
        env_overrides=[],
    )

    exit_code = scheduler.run_schedule(
        params,
        times,
        timezone=tz_name,
        runs=2,
        now_fn=fake_now,
        sleep_fn=fake_sleep,
    )

    assert exit_code == 0
    assert run_dates == [date(2024, 1, 1), date(2024, 1, 1)]
    assert sleep_calls == [30 * 60, 25 * 60]


def test_run_schedule_zero_runs_returns_immediately(tmp_path):
    params = orchestrate.RunParams(
        config_path=tmp_path / "cfg.yaml",
        output_base_dir=tmp_path / "out",
        top_n=5,
        use_cache=True,
        news_override=None,
        log_file=None,
        run_date=date(2024, 1, 1),
        timezone="UTC",
        env_overrides=[],
    )

    code = scheduler.run_schedule(
        params,
        [dtime(hour=7)],
        runs=0,
    )

    assert code == 0
