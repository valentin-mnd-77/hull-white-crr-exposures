import numpy as np
import pandas as pd
import pytest

from ccr.curve import bootstrap
from ccr.swap import par_swap_rate


def test_flat_curve_interpolation_is_exact(flat_curve):
    # A flat zero curve must be reproduced exactly between pillars.
    mid = flat_curve.reference_date + pd.Timedelta(days=200)
    tau = (mid - flat_curve.reference_date).days / 365
    assert flat_curve.discount(mid) == pytest.approx(np.exp(-0.04 * tau))
    assert flat_curve.zero_rate(mid) == pytest.approx(0.04)


def test_zero_rate_is_flat_before_the_first_pillar(flat_curve):
    early = flat_curve.reference_date + pd.Timedelta(days=3)
    assert flat_curve.zero_rate(early) == pytest.approx(flat_curve.zero_rate(flat_curve.pillar_dates[1]))


def test_discount_factors_are_decreasing(curve):
    assert np.all(np.diff(curve.discount_factors) < 0)
    assert curve.discount(curve.reference_date) == pytest.approx(1.0)


def test_market_swaps_are_repriced_at_par(market, curve):
    """Core invariant of the bootstrap, from the 2y pillar onwards."""
    mid = market.swaps[["BID", "ASK"]].mean(axis=1)
    for maturity, quote in mid.items():
        if maturity <= market.swaps.index[0]:
            continue  # the 1y point is not a pillar, see test below
        schedule = [d for d in market.swaps.index if d <= maturity]
        model = par_swap_rate(schedule, curve, market.settlement_date)
        assert model == pytest.approx(quote, abs=1e-10)


def test_one_year_point_carries_the_futures_swap_basis(market, curve):
    """The 1y rate comes from the futures strip, not from the swap quote.

    The gap is a property of the construction; pinning it down documents the
    choice and catches an accidental change of pillar selection.
    """
    quote = market.swaps[["BID", "ASK"]].mean(axis=1).iloc[0]
    model = par_swap_rate([market.swaps.index[0]], curve, market.settlement_date)
    basis_bp = (model - quote) * 1e4
    assert basis_bp == pytest.approx(-11.6, abs=0.5)


def test_rate_shock_moves_zero_rates_by_the_shock(market):
    """A +1 bp shock on market rates must raise zero rates by about 1 bp.

    Applying the shock to futures *prices* instead of futures rates -- the
    original bug -- moves the short end by -100 bp and fails this test.
    """
    base = bootstrap(market.settlement_date, market.depos, market.futures, market.swaps)
    bumped = bootstrap(
        market.settlement_date, market.depos, market.futures, market.swaps, shock=1e-4
    )
    for date in base.pillar_dates[1:]:
        delta_bp = (bumped.zero_rate(date) - base.zero_rate(date)) * 1e4
        assert 0.5 < delta_bp < 1.5
