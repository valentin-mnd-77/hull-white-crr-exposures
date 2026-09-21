"""Shared fixtures.

Tests that need the market data set are skipped automatically when the CSV
files are absent, since the data is not redistributed with the repository (see
``data/README.md``).  Everything that can be tested without it -- conventions,
model algebra, collateral mechanics, CVA -- runs on a synthetic flat curve and
is therefore always executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccr.curve import DiscountCurve, bootstrap  # noqa: E402
from ccr.dates import ACT_365  # noqa: E402
from ccr.market_data import load_market_data  # noqa: E402

DATA_DIR = ROOT / "data"
REQUIRED_FILES = ["depos.csv", "futures.csv", "expiry.csv", "swaps.csv", "dt.csv"]


@pytest.fixture(scope="session")
def market():
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        pytest.skip(f"market data not available: {', '.join(missing)}")
    return load_market_data(DATA_DIR)


@pytest.fixture(scope="session")
def curve(market):
    return bootstrap(market.settlement_date, market.depos, market.futures, market.swaps)


@pytest.fixture(scope="session")
def flat_curve() -> DiscountCurve:
    """Flat 4% continuously compounded curve, yearly pillars over 12 years."""
    reference = pd.Timestamp("2008-02-19")
    dates = pd.DatetimeIndex(
        [reference] + [reference + pd.DateOffset(years=y) for y in range(1, 13)]
    )
    taus = np.array([(d - reference).days / ACT_365 for d in dates])
    return DiscountCurve(dates, np.exp(-0.04 * taus))
