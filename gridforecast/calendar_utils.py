"""Calendar helpers: settlement periods, clock changes and bank holidays.

A GB settlement day runs from 00:00 to 24:00 UK local time and is split into
half-hour settlement periods. Normally that gives 48 periods, but on the day
the clocks go forward (last Sunday of March) the day has 23 hours and 46
periods, and on the day they go back (last Sunday of October) it has 25 hours
and 50 periods. Working on a UTC half-hour grid and converting to local time
only for features avoids every off-by-one-hour problem this would otherwise cause.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd

from .config import LOCAL_TZ


def local_day_periods(day: dt.date) -> pd.DatetimeIndex:
    """UTC start times of every settlement period in a UK local day (46, 48 or 50 of them)."""
    start = pd.Timestamp(day).tz_localize(LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="30min", inclusive="left")


@lru_cache(maxsize=None)
def _holidays_for(years: tuple) -> frozenset:
    import holidays
    h = holidays.country_holidays("GB", subdiv="ENG", years=list(years))
    return frozenset(h.keys())


def uk_holidays(years) -> frozenset:
    """England and Wales bank holidays (the bulk of GB demand) for the given years."""
    return _holidays_for(tuple(sorted(set(years))))


def is_christmas_period(dates: pd.Series) -> pd.Series:
    """24 December to 2 January, when demand behaves like a long holiday."""
    m, d = dates.dt.month, dates.dt.day
    return ((m == 12) & (d >= 24)) | ((m == 1) & (d <= 2))
