"""Day-count conventions and schedule generation.

Two conventions are used throughout the project:

* ``ACT/360``  -- money-market leg (deposits, futures, 3M Euribor floating leg).
* ``30E/360``  -- fixed leg of the interest rate swaps (European 30/360).

A third, ``ACT/365``, is used as the *modelling* time measure: every year
fraction fed to the Hull-White model (simulation of the state variable and the
affine coefficients) is expressed in ACT/365.  Mixing ACT/360 and ACT/365
between simulation and pricing biases the variance of the state variable by
365/360, so the choice is centralised here and never made ad hoc.
"""

from __future__ import annotations

import calendar
import datetime as dt
from typing import List, Sequence, Union

import pandas as pd

DateLike = Union[dt.date, dt.datetime, pd.Timestamp]

ACT_360 = 360
ACT_365 = 365

#: Time measure used by the Hull-White model (see module docstring).
MODEL_BASIS = ACT_365


def year_fraction_act(t1: DateLike, t2: DateLike, basis: int = ACT_365) -> float:
    """Year fraction between ``t1`` and ``t2`` under an ACT/``basis`` convention."""
    return (pd.Timestamp(t2) - pd.Timestamp(t1)).days / basis


def year_fraction_30e_360(t1: DateLike, t2: DateLike) -> float:
    """Year fraction between ``t1`` and ``t2`` under the 30E/360 convention.

    Both day numbers are capped at 30, which is what distinguishes the
    *European* variant from the US one.
    """
    t1, t2 = pd.Timestamp(t1), pd.Timestamp(t2)
    y1, m1, d1 = t1.year, t1.month, min(30, t1.day)
    y2, m2, d2 = t2.year, t2.month, min(30, t2.day)
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360


def business_date_offset(
    base_date: DateLike,
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> pd.Timestamp:
    """Shift ``base_date`` and roll the result forward to the next business day.

    This is a *modified-following-like* rule restricted to weekends: no holiday
    calendar is applied, consistently with the market data provided with the
    assignment.  Day overflow (e.g. 31 January + 1 month) falls back to the last
    valid day of the target month.
    """
    base_date = pd.Timestamp(base_date)

    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    try:
        shifted = base_date.replace(year=year, month=month, day=base_date.day)
    except ValueError:
        last_day = calendar.monthrange(year, month)[1]
        shifted = base_date.replace(year=year, month=month, day=last_day)

    shifted += dt.timedelta(days=day_offset)

    if shifted.weekday() == 5:  # Saturday
        shifted += dt.timedelta(days=2)
    elif shifted.weekday() == 6:  # Sunday
        shifted += dt.timedelta(days=1)

    return shifted


def date_series(t0: DateLike, t1: DateLike, frequency: int) -> List[pd.Timestamp]:
    """Return the payment schedule from ``t0`` to ``t1``, ``t0`` included.

    ``frequency`` is expressed as the number of dates per year (1 = annual,
    4 = quarterly).  Every date is generated as an offset from ``t0`` -- never
    by chaining offsets -- so that the schedule does not drift and so that
    schedules of different frequencies built from the same ``t0`` share their
    dates exactly.  That property is what allows the simulation grid to carry
    every payment date of the portfolio.
    """
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    if 12 % frequency != 0:
        raise ValueError("frequency must divide 12 (1, 2, 3, 4, 6 or 12 a year)")

    dates = [t0]
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // frequency))
    if dates[-1] > t1:
        dates.pop()
    if dates[-1] != t1:
        dates.append(t1)

    return dates


def schedule_year_fractions(schedule: Sequence[DateLike]) -> List[float]:
    """30E/360 accrual factors between consecutive dates of ``schedule``."""
    ordered = sorted(pd.Timestamp(d) for d in schedule)
    return [
        year_fraction_30e_360(ordered[i - 1], ordered[i]) for i in range(1, len(ordered))
    ]
