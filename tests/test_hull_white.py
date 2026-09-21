import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from ccr.dates import MODEL_BASIS, year_fraction_act
from ccr.hull_white import (
    HullWhiteParameters,
    affine_coefficients,
    simulate_state_paths,
    simulated_discount_factors,
    state_variance,
)

PARAMS = HullWhiteParameters(mean_reversion=0.0205, sigma=0.0094)


def _numerical_exponent(tau_s: float, tau_T: float, params: HullWhiteParameters) -> float:
    """``-1/2 * int_0^s [sigma(u,T)^2 - sigma(u,s)^2] du`` by quadrature.

    This is the formula as written in the lecture notes; the implementation
    uses its closed form.  Agreement between the two is what pins down the
    sign, which was inverted in the original code.
    """
    a, sigma = params.mean_reversion, params.sigma

    def integrand(u: float) -> float:
        vol_T = sigma / a * (1 - np.exp(-a * (tau_T - u)))
        vol_s = sigma / a * (1 - np.exp(-a * (tau_s - u)))
        return vol_T**2 - vol_s**2

    integral, _ = quad(integrand, 0.0, tau_s)
    return -0.5 * integral


def test_affine_coefficients_match_the_numerical_integral(flat_curve):
    grid = pd.DatetimeIndex(
        [flat_curve.reference_date + pd.DateOffset(years=y) for y in range(1, 11)]
    )
    valuation_date = grid[4]
    A, C = affine_coefficients(valuation_date, grid, PARAMS, flat_curve)

    tau_s = year_fraction_act(flat_curve.reference_date, valuation_date, MODEL_BASIS)
    df_s = flat_curve.discount(valuation_date)

    for j, date in enumerate(grid):
        if date <= valuation_date:
            assert A[j] == 1.0 and C[j] == 0.0
            continue
        tau_T = year_fraction_act(flat_curve.reference_date, date, MODEL_BASIS)
        expected = (
            flat_curve.discount(date)
            / df_s
            * np.exp(_numerical_exponent(tau_s, tau_T, PARAMS))
        )
        assert A[j] == pytest.approx(expected, rel=1e-10)


def test_affine_exponent_is_negative(flat_curve):
    """Sign check, stated on its own because it is the bug that was found.

    ``A(s, tau)`` must sit *below* the forward discount factor
    ``B(0, tau) / B(0, s)``.  The original code used ``exp(+I/2)`` and put it
    above, overstating the 10y discount factor seen from 5y by ~150 bp.
    """
    grid = pd.DatetimeIndex(
        [flat_curve.reference_date + pd.DateOffset(years=y) for y in range(1, 11)]
    )
    valuation_date = grid[4]
    A, _ = affine_coefficients(valuation_date, grid, PARAMS, flat_curve)
    forward = np.array(
        [flat_curve.discount(d) / flat_curve.discount(valuation_date) for d in grid]
    )
    later = grid > valuation_date
    assert np.all(A[later] < forward[later])


def test_c_coefficient_matches_its_closed_form(flat_curve):
    grid = pd.DatetimeIndex(
        [flat_curve.reference_date + pd.DateOffset(years=y) for y in range(1, 11)]
    )
    valuation_date = grid[2]
    _, C = affine_coefficients(valuation_date, grid, PARAMS, flat_curve)
    for j, date in enumerate(grid[3:], start=3):
        tau = year_fraction_act(valuation_date, date, MODEL_BASIS)
        expected = (1 - np.exp(-PARAMS.mean_reversion * tau)) / PARAMS.mean_reversion
        assert C[j] == pytest.approx(expected, rel=1e-12)


def test_discount_factor_is_one_at_the_valuation_date(flat_curve):
    grid = pd.DatetimeIndex(
        [flat_curve.reference_date + pd.DateOffset(years=y) for y in range(1, 11)]
    )
    x = np.array([-0.01, 0.0, 0.01])
    dfs = simulated_discount_factors(x, grid[3], grid, PARAMS, flat_curve)
    assert dfs[:, 3] == pytest.approx(np.ones(3))


def test_simulated_state_has_the_theoretical_moments():
    """``E[x_t] = 0`` and ``Var[x_t] = sigma^2/(2a)(1 - e^{-2at})``.

    The mean is compared against four Monte Carlo standard errors rather than
    an arbitrary threshold, so the test stays meaningful if the scenario count
    or the parameters change.
    """
    n_scenarios = 200_000
    grid = pd.DatetimeIndex(pd.date_range("2008-05-20", periods=40, freq="91D"))
    paths = simulate_state_paths(grid, "2008-02-19", PARAMS, n_scenarios, seed=42)

    for i in (9, 24, 39):
        tau = year_fraction_act("2008-02-19", grid[i], MODEL_BASIS)
        variance = state_variance(tau, PARAMS)
        standard_error = np.sqrt(variance / n_scenarios)
        assert paths[:, i].mean() == pytest.approx(0.0, abs=4 * standard_error)
        assert paths[:, i].var() == pytest.approx(variance, rel=0.02)


def test_simulation_is_reproducible():
    grid = pd.DatetimeIndex(pd.date_range("2008-02-19", periods=10, freq="91D"))
    first = simulate_state_paths(grid, "2008-02-19", PARAMS, 1_000, seed=7)
    second = simulate_state_paths(grid, "2008-02-19", PARAMS, 1_000, seed=7)
    assert np.array_equal(first, second)


def test_parameters_must_be_positive():
    with pytest.raises(ValueError):
        HullWhiteParameters(mean_reversion=0.0, sigma=0.01)
    with pytest.raises(ValueError):
        HullWhiteParameters(mean_reversion=0.02, sigma=-0.01)
