"""Vanilla interest rate swap: schedule, par rate and mark-to-market.

Single-curve framework (2008 data, pre-multicurve): the same curve is used for
discounting and for projecting the floating leg.  Under that assumption the
floating leg of a swap resetting at ``s`` is worth ``N * (1 - B(s, T_N))`` and
the fixed leg ``N * sum_i K * delta_i * B(s, t_i)``.

The accrual factor ``delta_i`` matters.  Dropping it -- a natural slip, because
the annual leg of the 10y swap has ``delta ~ 1`` and hides the bug -- multiplies
a quarterly fixed leg by roughly four.  ``tests/test_swap.py`` checks the only
invariant that catches it: a swap struck at its par rate is worth zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from .curve import DiscountCurve
from .dates import (
    DateLike,
    business_date_offset,
    date_series,
    schedule_year_fractions,
)


class SwapType(Enum):
    """Sign of the swap seen from the fixed-rate payer / receiver."""

    PAYER = 1
    RECEIVER = -1


def basis_point_value(
    payment_dates: Sequence[DateLike],
    curve: DiscountCurve,
    settlement_date: DateLike | None = None,
) -> float:
    """Annuity of the fixed leg, seen from ``settlement_date``.

    For a forward starting swap the discount factors are normalised by the
    discount factor at settlement, so that the result is the annuity *as of*
    the settlement date.
    """
    settlement_date = (
        curve.reference_date if settlement_date is None else pd.Timestamp(settlement_date)
    )
    accruals = schedule_year_fractions([settlement_date, *payment_dates])
    df_settlement = curve.discount(settlement_date)
    if df_settlement == 0.0:
        raise ZeroDivisionError("null discount factor at settlement date")
    dfs = curve.discounts(payment_dates) / df_settlement
    return float(np.sum(np.asarray(accruals) * dfs))


def par_swap_rate(
    payment_dates: Sequence[DateLike],
    curve: DiscountCurve,
    settlement_date: DateLike | None = None,
) -> float:
    """Fixed rate that makes the swap worth zero at ``settlement_date``."""
    settlement_date = (
        curve.reference_date if settlement_date is None else pd.Timestamp(settlement_date)
    )
    df_start = curve.discount(settlement_date)
    df_end = curve.discount(payment_dates[-1])
    floating_leg = 1.0 - df_end / df_start
    return floating_leg / basis_point_value(payment_dates, curve, settlement_date)


@dataclass(frozen=True)
class InterestRateSwap:
    """A spot starting vanilla IRS with a 30E/360 fixed leg.

    The floating leg is 3M Euribor (ACT/360, quarterly).  Since the swap is
    valued only on reset dates of the floating leg, its convention never enters
    the valuation explicitly -- see :meth:`mtm_paths`.
    """

    settlement_date: pd.Timestamp
    maturity_years: int
    fixed_frequency: int
    notional: float
    swap_type: SwapType
    fixed_rate: float

    @classmethod
    def at_par(
        cls,
        settlement_date: DateLike,
        maturity_years: int,
        fixed_frequency: int,
        notional: float,
        swap_type: SwapType,
        curve: DiscountCurve,
    ) -> "InterestRateSwap":
        """Build the swap struck at the par rate implied by ``curve``."""
        settlement_date = pd.Timestamp(settlement_date)
        expiry = business_date_offset(settlement_date, year_offset=maturity_years)
        payment_dates = date_series(settlement_date, expiry, fixed_frequency)[1:]
        rate = par_swap_rate(payment_dates, curve, settlement_date)
        return cls(
            settlement_date=settlement_date,
            maturity_years=maturity_years,
            fixed_frequency=fixed_frequency,
            notional=notional,
            swap_type=swap_type,
            fixed_rate=rate,
        )

    # ------------------------------------------------------------------ dates
    @property
    def expiry(self) -> pd.Timestamp:
        return business_date_offset(self.settlement_date, year_offset=self.maturity_years)

    @property
    def schedule(self) -> list[pd.Timestamp]:
        """Fixed leg schedule, settlement date included as first element."""
        return date_series(self.settlement_date, self.expiry, self.fixed_frequency)

    @property
    def payment_dates(self) -> list[pd.Timestamp]:
        return self.schedule[1:]

    @property
    def accruals(self) -> np.ndarray:
        """30E/360 accrual factors of the fixed leg."""
        return np.asarray(schedule_year_fractions(self.schedule))

    @property
    def fixed_cash_flows(self) -> pd.Series:
        """Fixed leg cash flows per unit of notional: ``K * delta_i``."""
        return pd.Series(self.fixed_rate * self.accruals, index=self.payment_dates)

    # ----------------------------------------------------------------- values
    def mtm(self, curve: DiscountCurve) -> float:
        """Mark-to-market at the curve reference date."""
        dfs = curve.discounts(self.payment_dates)
        floating_leg = 1.0 - dfs[-1]
        fixed_leg = float(np.sum(self.fixed_cash_flows.to_numpy() * dfs))
        return self.swap_type.value * self.notional * (floating_leg - fixed_leg)

    def mtm_paths(
        self,
        simulated_dfs: np.ndarray,
        grid: pd.DatetimeIndex,
        valuation_date: DateLike,
    ) -> np.ndarray:
        """Mark-to-market at ``valuation_date`` across Monte Carlo scenarios.

        Parameters
        ----------
        simulated_dfs
            Matrix ``B(s, t_j)`` of shape ``(n_scenarios, len(grid))``, where
            ``s`` is ``valuation_date`` and ``t_j`` runs over ``grid``.
        grid
            Simulation grid; it must contain every payment date of the swap
            that is still outstanding.
        valuation_date
            The date ``s``.  It must be a reset date of the floating leg, which
            holds by construction because the grid is quarterly and the
            floating leg is 3M Euribor.  This is what allows the floating leg to
            be valued as ``1 - B(s, T_N)`` with no accrued interest term.

        Returns
        -------
        np.ndarray
            Mark-to-market per scenario, shape ``(n_scenarios,)``.
        """
        valuation_date = pd.Timestamp(valuation_date)
        n_scenarios = simulated_dfs.shape[0]

        if valuation_date >= self.expiry:
            return np.zeros(n_scenarios)

        grid = pd.DatetimeIndex(grid)
        missing = [d for d in self.payment_dates if d not in grid]
        if missing:
            raise ValueError(
                f"payment dates missing from the simulation grid: {missing[:3]}"
            )
        if valuation_date not in grid:
            raise ValueError("valuation date must belong to the simulation grid")

        # Outstanding payments: strictly after s (a payment falling on s has
        # just been settled and is no longer part of the contract value).
        outstanding = np.array([d > valuation_date for d in self.payment_dates])
        positions = np.array([grid.get_loc(d) for d in self.payment_dates])

        cash_flows = self.fixed_cash_flows.to_numpy()[outstanding]
        dfs = simulated_dfs[:, positions[outstanding]]

        floating_leg = 1.0 - simulated_dfs[:, grid.get_loc(self.payment_dates[-1])]
        fixed_leg = dfs @ cash_flows

        return self.swap_type.value * self.notional * (floating_leg - fixed_leg)
