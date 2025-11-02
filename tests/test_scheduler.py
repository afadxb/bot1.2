from datetime import datetime

import pytest
from dateutil import tz

from steadyalpha import scheduler


def test_is_within_window_simple():
    tzinfo = tz.gettz("America/New_York")
    now = datetime(2024, 1, 1, 8, 45, tzinfo=tzinfo)
    assert scheduler.is_within_window(now, "08:30", "09:15")


def test_is_within_window_wraps_midnight():
    tzinfo = tz.gettz("America/New_York")
    now = datetime(2024, 1, 1, 0, 30, tzinfo=tzinfo)
    assert scheduler.is_within_window(now, "23:30", "00:45")
    later = datetime(2024, 1, 1, 1, 0, tzinfo=tzinfo)
    assert not scheduler.is_within_window(later, "23:30", "00:45")


@pytest.mark.parametrize("value", ["24:00", "09", "10:70", "abc"])
def test_is_within_window_invalid(value):
    tzinfo = tz.gettz("UTC")
    now = datetime(2024, 1, 1, 10, 0, tzinfo=tzinfo)
    with pytest.raises(ValueError):
        scheduler.is_within_window(now, value, "11:00")
