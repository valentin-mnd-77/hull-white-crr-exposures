import numpy as np
import pandas as pd
import pytest

from ccr.cva import CreditParameters, compute_cva
from ccr.dates import business_date_offset
from ccr.engine import SimulationConfig, run_simulation, simulation_grid
from ccr.exposure import CSAParameters
from ccr.hull_white import HullWhiteParameters
from ccr.swap import InterestRateSwap, SwapType

PARAMS = HullWhiteParameters(mean_reversion=0.0205, sigma=0.0094)


# --------------------------------------------------------------------- CVA


def test_cva_is_zero_without_exposure(flat_curve):
    grid = simulation_grid(flat_curve.reference_date, SimulationConfig())
    epe = pd.Series(0.0, index=grid)
    assert compute_cva(epe, flat_curve, CreditParameters()) == 0.0


def test_cva_is_linear_in_lgd(flat_curve):
    grid = simulation_grid(flat_curve.reference_date, SimulationConfig())
    epe = pd.Series(1e6, index=grid)
    half = compute_cva(epe, flat_curve, CreditParameters(lgd=0.3))
    full = compute_cva(epe, flat_curve, CreditParameters(lgd=0.6))
    assert full == pytest.approx(2 * half)


def test_cva_increases_with_the_hazard_rate(flat_curve):
    grid = simulation_grid(flat_curve.reference_date, SimulationConfig())
    epe = pd.Series(1e6, index=grid)
    low = compute_cva(epe, flat_curve, CreditParameters(hazard_rate=0.01))
    high = compute_cva(epe, flat_curve, CreditParameters(hazard_rate=0.05))
    assert high > low


def test_cva_of_a_constant_exposure_matches_its_closed_form(flat_curve):
    """Constant EPE over [0, T]: CVA = LGD * E * integral of B dPD.

    With a flat zero rate r and a flat hazard rate h the discrete sum telescopes
    to LGD * E * h/(r+h) * (1 - e^{-(r+h)T}) as the grid is refined.
    """
    config = SimulationConfig(horizon_years=10, steps_per_year=12)
    grid = simulation_grid(flat_curve.reference_date, config)
    exposure, r, h, lgd = 1e6, 0.04, 0.02, 0.6
    epe = pd.Series(exposure, index=grid)

    numeric = compute_cva(epe, flat_curve, CreditParameters(hazard_rate=h, lgd=lgd))
    analytic = lgd * exposure * h / (r + h) * (1 - np.exp(-(r + h) * 10))
    assert numeric == pytest.approx(analytic, rel=0.01)


# ------------------------------------------------------------------ engine


@pytest.fixture(scope="module")
def small_run(flat_curve):
    """A short, low-count run: enough to check structural properties."""
    settlement = flat_curve.reference_date
    swap = InterestRateSwap.at_par(settlement, 10, 1, 100e6, SwapType.PAYER, flat_curve)
    swap_2 = InterestRateSwap.at_par(settlement, 5, 4, 150e6, SwapType.RECEIVER, flat_curve)
    break_date = business_date_offset(settlement, year_offset=3, month_offset=6)
    config = SimulationConfig(n_scenarios=4_000, seed=1)
    return swap, run_simulation(
        flat_curve,
        settlement,
        swap,
        swap_2,
        break_date,
        CSAParameters.counterparty_fully_collateralised(),
        PARAMS,
        config,
    ), break_date


def test_netting_never_increases_exposure(small_run):
    """Sub-additivity of the positive part: max(a+b,0) <= max(a,0)+max(b,0)."""
    _, results, _ = small_run
    assert np.all(results.epe["netting"] <= results.epe["no_netting"] + 1e-9)
    assert np.all(results.pfe["netting"] <= results.pfe["no_netting"] + 1e-9)


def test_no_netting_is_the_sum_of_the_standalone_profiles(small_run):
    _, results, _ = small_run
    assert results.epe["no_netting"].to_numpy() == pytest.approx(
        (results.epe["base"] + results.epe["swap_2"]).to_numpy()
    )


def test_break_clause_kills_the_exposure_after_the_break_date(small_run):
    _, results, break_date = small_run
    after = results.epe.index > break_date
    assert np.all(results.epe.loc[after, "break"] == 0.0)
    assert results.epe.loc[~after, "break"].to_numpy() == pytest.approx(
        results.epe.loc[~after, "base"].to_numpy()
    )


def test_collateral_reduces_exposure_without_ever_increasing_it(small_run):
    _, results, _ = small_run
    assert np.all(results.epe["csa"] <= results.epe["base"] + 1e-9)
    assert results.epe["csa"].sum() < results.epe["base"].sum()


def test_exposure_vanishes_at_maturity(small_run):
    _, results, _ = small_run
    assert results.epe.iloc[-1]["base"] == pytest.approx(0.0)
    assert results.pfe.iloc[-1]["base"] == pytest.approx(0.0)


def test_profiles_are_non_negative(small_run):
    _, results, _ = small_run
    assert np.all(results.epe.to_numpy() >= 0.0)
    assert np.all(results.pfe.to_numpy() >= 0.0)
