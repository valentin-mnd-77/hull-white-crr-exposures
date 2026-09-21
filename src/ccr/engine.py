"""Monte Carlo engine producing the exposure profiles of the case study.

One pass over the simulation grid produces, for every mitigation scenario, the
EPE and PFE profiles.  Scenarios share the same paths of the state variable, so
differences between them are differences in the contractual arrangement and not
Monte Carlo noise -- the same property is used deliberately for the CVA
sensitivities (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .curve import DiscountCurve
from .dates import ACT_365, DateLike, business_date_offset, date_series, year_fraction_act
from .exposure import (
    CSAParameters,
    expected_positive_exposure,
    potential_future_exposure,
    update_variation_margin,
)
from .hull_white import HullWhiteParameters, simulate_state_paths, simulated_discount_factors
from .swap import InterestRateSwap

#: Exposure scenarios reported by :func:`run_simulation`.
SCENARIOS = ["base", "break", "swap_2", "no_netting", "netting", "csa"]


@dataclass(frozen=True)
class SimulationConfig:
    """Grid and Monte Carlo settings."""

    n_scenarios: int = 250_000
    horizon_years: int = 10
    steps_per_year: int = 4
    confidence_level: float = 0.95
    seed: int = 20080215

    @property
    def n_steps(self) -> int:
        return self.horizon_years * self.steps_per_year


@dataclass
class SimulationResults:
    """EPE and PFE profiles, indexed by simulation date, one column per scenario."""

    epe: pd.DataFrame
    pfe: pd.DataFrame
    grid: pd.DatetimeIndex
    expected_collateral: pd.Series = field(default_factory=pd.Series)


def simulation_grid(settlement_date: DateLike, config: SimulationConfig) -> pd.DatetimeIndex:
    """Equally spaced future simulation dates, excluding ``t0``.

    Every date is generated as an offset from the settlement date, so the grid
    contains the payment dates of both the annual and the quarterly leg.
    """
    horizon = business_date_offset(settlement_date, year_offset=config.horizon_years)
    dates = date_series(settlement_date, horizon, config.steps_per_year)[1:]
    return pd.DatetimeIndex(dates)


def run_simulation(
    curve: DiscountCurve,
    settlement_date: DateLike,
    swap: InterestRateSwap,
    swap_netting: InterestRateSwap,
    break_date: DateLike,
    csa: CSAParameters,
    hw_params: HullWhiteParameters,
    config: SimulationConfig,
    margin_frequency: int = 1,
) -> SimulationResults:
    """Run the full exposure simulation.

    Parameters
    ----------
    swap
        The 10y payer IRS whose counterparty risk is being measured.
    swap_netting
        The second trade entering the netting set under scenario 6b.
    break_date
        Date of the mandatory break clause (scenario 6a).
    csa
        Collateral agreement of scenario 6c.
    margin_frequency
        Margin calls per year.
    """
    settlement_date = pd.Timestamp(settlement_date)
    break_date = pd.Timestamp(break_date)
    grid = simulation_grid(settlement_date, config)

    state_paths = simulate_state_paths(
        grid, settlement_date, hw_params, config.n_scenarios, seed=config.seed
    )

    # Explicit margin call schedule rather than a day-count heuristic.
    margin_dates = set(
        pd.DatetimeIndex(date_series(settlement_date, grid[-1], margin_frequency)[1:])
    )

    # Collateral is remunerated at the shortest zero rate available on the
    # curve, as prescribed by the case study.  Using the simulated short rate
    # instead would be scenario-consistent but is not what is asked.
    short_rate = curve.zero_rate(curve.pillar_dates[1])

    epe = pd.DataFrame(0.0, index=grid, columns=SCENARIOS)
    pfe = pd.DataFrame(0.0, index=grid, columns=SCENARIOS)
    expected_collateral = pd.Series(0.0, index=grid)

    variation_margin = np.zeros(config.n_scenarios)
    previous_date = settlement_date

    for i, date in enumerate(grid):
        dfs = simulated_discount_factors(
            state_paths[:, i], date, grid, hw_params, curve
        )
        mtm = swap.mtm_paths(dfs, grid, date)
        mtm_netting = swap_netting.mtm_paths(dfs, grid, date)

        # --- collateral: accrue interest, then call, then measure exposure ---
        variation_margin *= np.exp(
            short_rate * year_fraction_act(previous_date, date, ACT_365)
        )
        if date in margin_dates:
            variation_margin = update_variation_margin(mtm, variation_margin, csa)

        exposures = {
            "base": np.maximum(mtm, 0.0),
            "break": np.maximum(mtm, 0.0) if date <= break_date else np.zeros_like(mtm),
            "swap_2": np.maximum(mtm_netting, 0.0),
            "no_netting": np.maximum(mtm, 0.0) + np.maximum(mtm_netting, 0.0),
            "netting": np.maximum(mtm + mtm_netting, 0.0),
            "csa": np.maximum(mtm - variation_margin, 0.0),
        }

        for name, exposure in exposures.items():
            epe.loc[date, name] = expected_positive_exposure(exposure)
            pfe.loc[date, name] = potential_future_exposure(
                exposure, config.confidence_level
            )
        expected_collateral.loc[date] = float(np.mean(variation_margin))

        previous_date = date

    return SimulationResults(
        epe=epe, pfe=pfe, grid=grid, expected_collateral=expected_collateral
    )
