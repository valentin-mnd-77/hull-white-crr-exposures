from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ccr.curve_source import load_curve
from ccr.cva import CreditParameters, cva_sensitivities
from ccr.dates import ACT_365, business_date_offset, year_fraction_act
from ccr.engine import SimulationConfig
from ccr.exposure import CSAParameters
from ccr.hull_white import HullWhiteParameters
from ccr.swap import InterestRateSwap, SwapType
from ccr.synthetic_curve import SETTLEMENT_DATE, synthetic_curve

# ------------------------------------------------------------- offline tests


def test_fallback_curve_is_a_valid_discount_curve():
    curve = synthetic_curve()
    assert curve.reference_date == SETTLEMENT_DATE
    assert curve.discount(SETTLEMENT_DATE) == pytest.approx(1.0)
    assert np.all(np.diff(curve.discount_factors) < 0)
    assert curve.pillar_dates[-1] >= SETTLEMENT_DATE + pd.DateOffset(years=50)


def test_fallback_shift_moves_every_zero_rate_by_the_shift():
    base, bumped = synthetic_curve(), synthetic_curve(shift=1e-4)
    for years in (0.5, 2, 5, 10, 30):
        date = SETTLEMENT_DATE + pd.DateOffset(days=int(365 * years))
        delta_bp = (bumped.zero_rate(date) - base.zero_rate(date)) * 1e4
        assert delta_bp == pytest.approx(1.0, abs=1e-6)


def test_auto_source_falls_back_when_data_is_missing(tmp_path):
    source = load_curve(tmp_path)  # empty directory
    assert source.is_synthetic
    assert source.settlement_date == SETTLEMENT_DATE
    assert source.build(0.0).discount_factors == pytest.approx(source.curve.discount_factors)


def test_explicit_market_source_fails_loudly_without_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_curve(tmp_path, source="market")


def test_unknown_source_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_curve(tmp_path, source="yahoo")


def test_cva_sensitivities_have_the_expected_signs_offline():
    """End-to-end on the fallback curve: the three sensitivities of a payer
    swap are all positive (directional in rates, linear in credit, long vol)."""
    source = load_curve(source="synthetic")
    settlement = source.settlement_date
    swap = InterestRateSwap.at_par(settlement, 10, 1, 100e6, SwapType.PAYER, source.curve)
    swap_2 = InterestRateSwap.at_par(settlement, 5, 4, 150e6, SwapType.RECEIVER, source.curve)

    sensitivities = cva_sensitivities(
        curve_builder=source.build,
        curve=source.curve,
        swap=swap,
        swap_netting=swap_2,
        break_date=business_date_offset(settlement, year_offset=3, month_offset=6),
        csa=CSAParameters.counterparty_fully_collateralised(),
        hw_params=HullWhiteParameters(mean_reversion=0.0205, sigma=0.0094),
        credit=CreditParameters(),
        config=SimulationConfig(n_scenarios=4_000, seed=3),
    )
    assert np.all(sensitivities["delta_cva"].iloc[1:] > 0)


# ------------------------------------------------- accuracy against the data


def test_fallback_tracks_the_bootstrapped_curve(curve):
    """Pins the accuracy stated in the README.

    Within 2.5 bp from two years onwards; the money-market structure of
    February 2008 below two years is smoothed out, by up to ~20 bp.
    """
    fallback = synthetic_curve()
    for date in curve.pillar_dates[1:]:
        tau = year_fraction_act(curve.reference_date, date, ACT_365)
        error_bp = abs(fallback.zero_rate(date) - curve.zero_rate(date)) * 1e4
        assert error_bp < (2.5 if tau >= 2 else 25.0), f"{date:%Y-%m-%d}: {error_bp:.2f} bp"


def test_auto_source_prefers_market_data(market):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    source = load_curve(data_dir)
    assert not source.is_synthetic
    assert source.settlement_date == market.settlement_date
