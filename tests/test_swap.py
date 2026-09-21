import numpy as np
import pandas as pd
import pytest

from ccr.engine import SimulationConfig, simulation_grid
from ccr.swap import InterestRateSwap, SwapType


@pytest.mark.parametrize(
    "maturity, frequency, swap_type",
    [
        (10, 1, SwapType.PAYER),
        (5, 4, SwapType.RECEIVER),
        (5, 2, SwapType.PAYER),
    ],
)
def test_par_swap_has_zero_value_at_inception(flat_curve, maturity, frequency, swap_type):
    """The invariant that catches a missing accrual factor.

    Without ``delta_i`` in the fixed cash flows a quarterly swap is off by
    roughly a factor four -- on the real data set, -81 M EUR at inception on a
    150 M EUR notional.
    """
    swap = InterestRateSwap.at_par(
        settlement_date=flat_curve.reference_date,
        maturity_years=maturity,
        fixed_frequency=frequency,
        notional=100_000_000,
        swap_type=swap_type,
        curve=flat_curve,
    )
    assert swap.mtm(flat_curve) == pytest.approx(0.0, abs=1e-6)


def test_accruals_match_the_payment_frequency(flat_curve):
    swap = InterestRateSwap.at_par(
        flat_curve.reference_date, 5, 4, 1.0, SwapType.PAYER, flat_curve
    )
    assert swap.accruals == pytest.approx(np.full(len(swap.accruals), 0.25), abs=0.01)
    assert len(swap.payment_dates) == 20


def test_par_rate_of_a_flat_curve_is_the_zero_rate(flat_curve):
    swap = InterestRateSwap.at_par(
        flat_curve.reference_date, 10, 1, 1.0, SwapType.PAYER, flat_curve
    )
    # Continuously compounded 4% -> annually compounded par rate.
    assert swap.fixed_rate == pytest.approx(np.exp(0.04) - 1, rel=1e-3)


def test_payer_and_receiver_are_mirror_images(flat_curve):
    payer = InterestRateSwap.at_par(
        flat_curve.reference_date, 10, 1, 1e8, SwapType.PAYER, flat_curve
    )
    receiver = InterestRateSwap(
        settlement_date=payer.settlement_date,
        maturity_years=payer.maturity_years,
        fixed_frequency=payer.fixed_frequency,
        notional=payer.notional,
        swap_type=SwapType.RECEIVER,
        fixed_rate=payer.fixed_rate + 0.005,  # off market on purpose
    )
    off_market_payer = InterestRateSwap(
        settlement_date=payer.settlement_date,
        maturity_years=payer.maturity_years,
        fixed_frequency=payer.fixed_frequency,
        notional=payer.notional,
        swap_type=SwapType.PAYER,
        fixed_rate=payer.fixed_rate + 0.005,
    )
    assert receiver.mtm(flat_curve) == pytest.approx(-off_market_payer.mtm(flat_curve))


def test_mtm_paths_reduce_to_the_spot_value_on_a_deterministic_curve(flat_curve):
    """With no randomness the simulated MtM must match the analytic one."""
    config = SimulationConfig(n_scenarios=1, horizon_years=10, steps_per_year=4)
    grid = simulation_grid(flat_curve.reference_date, config)
    swap = InterestRateSwap.at_par(
        flat_curve.reference_date, 10, 1, 1e8, SwapType.PAYER, flat_curve
    )

    valuation_date = grid[3]  # one year into the trade
    forward_dfs = np.array(
        [
            flat_curve.discount(d) / flat_curve.discount(valuation_date) if d > valuation_date else 1.0
            for d in grid
        ]
    ).reshape(1, -1)

    simulated = swap.mtm_paths(forward_dfs, grid, valuation_date)[0]

    # Analytic forward value: same formula, computed directly off the curve.
    outstanding = [d for d in swap.payment_dates if d > valuation_date]
    dfs = flat_curve.discounts(outstanding) / flat_curve.discount(valuation_date)
    accruals = swap.fixed_cash_flows.loc[outstanding].to_numpy()
    expected = swap.notional * ((1.0 - dfs[-1]) - float(accruals @ dfs))

    assert simulated == pytest.approx(expected, rel=1e-12)


def test_mtm_is_zero_after_expiry(flat_curve):
    config = SimulationConfig(n_scenarios=5, horizon_years=10, steps_per_year=4)
    grid = simulation_grid(flat_curve.reference_date, config)
    swap = InterestRateSwap.at_par(
        flat_curve.reference_date, 5, 4, 1e8, SwapType.RECEIVER, flat_curve
    )
    dfs = np.ones((5, len(grid)))
    assert np.all(swap.mtm_paths(dfs, grid, grid[-1]) == 0.0)


def test_payment_dates_must_lie_on_the_grid(flat_curve):
    config = SimulationConfig(n_scenarios=2, horizon_years=10, steps_per_year=1)
    grid = simulation_grid(flat_curve.reference_date, config)
    swap = InterestRateSwap.at_par(
        flat_curve.reference_date, 5, 4, 1e8, SwapType.PAYER, flat_curve
    )
    with pytest.raises(ValueError, match="missing from the simulation grid"):
        swap.mtm_paths(np.ones((2, len(grid))), grid, grid[0])
