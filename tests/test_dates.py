import pandas as pd
import pytest

from ccr.dates import (
    business_date_offset,
    date_series,
    schedule_year_fractions,
    year_fraction_30e_360,
    year_fraction_act,
)


def test_30e_360_full_year_is_one():
    assert year_fraction_30e_360("2008-02-19", "2009-02-19") == pytest.approx(1.0)


def test_30e_360_caps_day_at_thirty():
    # 31 January to 28 February: both day numbers capped at 30 -> 28/360 is not
    # the answer, 30/360 is.
    assert year_fraction_30e_360("2008-01-31", "2008-02-29") == pytest.approx(29 / 360)


def test_act_basis_differ_by_the_expected_ratio():
    act_360 = year_fraction_act("2008-02-19", "2009-02-19", 360)
    act_365 = year_fraction_act("2008-02-19", "2009-02-19", 365)
    assert act_360 / act_365 == pytest.approx(365 / 360)


def test_business_date_offset_rolls_off_the_weekend():
    # 19 February 2011 is a Saturday.
    assert business_date_offset("2008-02-19", year_offset=3) == pd.Timestamp("2011-02-21")


def test_annual_schedule_is_a_subset_of_the_quarterly_one():
    """The property the whole simulation relies on.

    Payment dates are read off the simulation grid, so every annual payment
    date must coincide exactly with a quarterly grid date.  Generating dates as
    offsets from a common origin is what guarantees it; chaining offsets would
    not.
    """
    start, end = pd.Timestamp("2008-02-19"), business_date_offset("2008-02-19", year_offset=10)
    annual = set(date_series(start, end, 1))
    quarterly = set(date_series(start, end, 4))
    assert annual <= quarterly


def test_schedule_year_fractions_sum_to_the_tenor():
    start, end = pd.Timestamp("2008-02-19"), business_date_offset("2008-02-19", year_offset=5)
    quarterly = date_series(start, end, 4)
    assert sum(schedule_year_fractions(quarterly)) == pytest.approx(5.0, abs=0.02)


def test_date_series_rejects_frequency_not_dividing_twelve():
    with pytest.raises(ValueError):
        date_series("2008-02-19", "2018-02-19", 5)
