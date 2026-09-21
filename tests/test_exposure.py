import numpy as np
import pandas as pd
import pytest

from ccr.exposure import (
    CSAParameters,
    expected_positive_exposure,
    peak_pfe,
    potential_future_exposure,
    update_variation_margin,
)


def test_uncollateralised_call_matches_the_mark_to_market():
    csa = CSAParameters()
    mtm = np.array([10.0, -10.0, 0.0])
    vm = update_variation_margin(mtm, np.zeros(3), csa)
    assert vm == pytest.approx(np.array([10.0, -10.0, 0.0]))


def test_threshold_leaves_the_first_slice_unsecured():
    csa = CSAParameters(threshold_counterparty=5.0, threshold_bank=5.0)
    vm = update_variation_margin(np.array([4.0, 12.0, -12.0]), np.zeros(3), csa)
    assert vm == pytest.approx(np.array([0.0, 7.0, -7.0]))


def test_minimum_transfer_amount_blocks_small_calls():
    csa = CSAParameters(mta_counterparty=3.0, mta_bank=3.0)
    vm = np.array([10.0, 10.0])
    # +2 is below the MTA and is not transferred, +5 is.
    updated = update_variation_margin(np.array([12.0, 15.0]), vm, csa)
    assert updated == pytest.approx(np.array([10.0, 15.0]))


def test_cap_limits_the_collateral_held():
    csa = CSAParameters(cap_counterparty=8.0)
    vm = update_variation_margin(np.array([20.0]), np.zeros(1), csa)
    assert vm == pytest.approx(np.array([8.0]))


def test_case_study_csa_is_one_sided_and_complete():
    """The arrangement required by the case study.

    The bank must never post -- a negative mark-to-market leaves the balance at
    zero -- and the counterparty must cover its exposure in full, with no cap
    biting.  The parameters originally used (a 2 M threshold for the bank, a
    15 M cap on the counterparty) fail both halves of this test.
    """
    csa = CSAParameters.counterparty_fully_collateralised()
    mtm = np.array([-50e6, -1e6, 0.0, 1e6, 50e6])
    vm = update_variation_margin(mtm, np.zeros(5), csa)
    assert vm == pytest.approx(np.array([0.0, 0.0, 0.0, 1e6, 50e6]))
    # Residual exposure at the posting date is nil.
    assert np.maximum(mtm - vm, 0.0) == pytest.approx(np.zeros(5))


def test_collateral_is_returned_when_the_exposure_reverses():
    csa = CSAParameters.counterparty_fully_collateralised()
    vm = update_variation_margin(np.array([30e6]), np.zeros(1), csa)
    vm = update_variation_margin(np.array([-5e6]), vm, csa)
    assert vm == pytest.approx(np.zeros(1))


def test_exposure_metrics_ignore_the_negative_part():
    exposure = np.array([-10.0, 0.0, 10.0, 30.0])
    assert expected_positive_exposure(exposure) == pytest.approx(10.0)
    assert potential_future_exposure(exposure, 0.5) == pytest.approx(5.0)


def test_peak_pfe_reports_value_and_date():
    index = pd.date_range("2008-02-19", periods=4, freq="91D")
    profiles = pd.DataFrame({"base": [1.0, 7.0, 3.0, 0.0]}, index=index)
    peak = peak_pfe(profiles)
    assert peak.loc["base", "peak_pfe"] == 7.0
    assert peak.loc["base", "date"] == index[1]
