"""Single-curve discount factor bootstrap and interpolation.

The curve is bootstrapped from deposits, futures and par swap rates, following
the standard pillar split:

* deposits up to the settlement date of the first future (included),
* the futures strip up to the 2y swap date,
* par swap rates from the 2y point onwards.

The 1y swap rate is therefore *not* a pillar of the curve: the 1y point comes
from the futures strip and is only used to initialise the basis point value of
the swap leg.  As a consequence the bootstrapped curve reprices every market
swap from 2y onwards exactly, but leaves a futures/swap basis at the 1y point
(about -12 bp on the February 2008 data set).  This is a property of the
construction, not a bug -- ``tests/test_curve.py`` pins it down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .dates import ACT_360, ACT_365, DateLike, year_fraction_30e_360, year_fraction_act


@dataclass(frozen=True)
class DiscountCurve:
    """A discount factor curve with log-linear (zero-rate linear) interpolation.

    Interpolation is performed on continuously compounded zero rates expressed
    in ACT/365 time rather than directly on discount factors: interpolating
    discount factors linearly produces a non-smooth, arbitrageable forward
    curve.  Outside the pillar range the zero rate is held flat, which is the
    market convention for the very short end and avoids the spurious
    ``z(0) = 0`` node that a naive implementation introduces at the reference
    date.
    """

    pillar_dates: pd.DatetimeIndex
    discount_factors: np.ndarray
    _taus: np.ndarray = field(init=False, repr=False)
    _zeros: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.pillar_dates)
        dfs = np.asarray(self.discount_factors, dtype=float)
        if len(dates) != len(dfs):
            raise ValueError("pillar_dates and discount_factors must have equal length")
        order = np.argsort(dates.values)
        dates, dfs = dates[order], dfs[order]
        object.__setattr__(self, "pillar_dates", dates)
        object.__setattr__(self, "discount_factors", dfs)

        # Zero rates on the pillars, excluding the reference date itself where
        # the zero rate is undefined (tau = 0).
        taus = np.array(
            [year_fraction_act(dates[0], d, ACT_365) for d in dates], dtype=float
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            zeros = -np.log(dfs[1:]) / taus[1:]
        object.__setattr__(self, "_taus", taus[1:])
        object.__setattr__(self, "_zeros", zeros)

    @property
    def reference_date(self) -> pd.Timestamp:
        return self.pillar_dates[0]

    def zero_rate(self, date: DateLike) -> float:
        """Continuously compounded ACT/365 zero rate at ``date`` (flat outside)."""
        tau = year_fraction_act(self.reference_date, date, ACT_365)
        if tau <= 0:
            return float(self._zeros[0])
        return float(np.interp(tau, self._taus, self._zeros))

    def discount(self, date: DateLike) -> float:
        """Discount factor B(t0, ``date``)."""
        tau = year_fraction_act(self.reference_date, date, ACT_365)
        if tau <= 0:
            return 1.0
        return float(np.exp(-self.zero_rate(date) * tau))

    def discounts(self, dates: Iterable[DateLike]) -> np.ndarray:
        """Vectorised :meth:`discount`."""
        return np.array([self.discount(d) for d in dates], dtype=float)

    def to_series(self) -> pd.Series:
        return pd.Series(self.discount_factors, index=self.pillar_dates, name="discount")

    def zero_rate_series(self) -> pd.Series:
        return pd.Series(self._zeros, index=self.pillar_dates[1:], name="zero_rate")


def _futures_used(futures: pd.DataFrame, two_year_swap_date: pd.Timestamp) -> pd.DataFrame:
    """Futures whose expiry does not extend beyond the 2y swap pillar."""
    mask = futures["Expiry"] <= two_year_swap_date
    return futures.loc[mask].copy()


def bootstrap(
    reference_date: DateLike,
    depos: pd.DataFrame,
    futures: pd.DataFrame,
    swaps: pd.DataFrame,
    shock: float = 0.0,
) -> DiscountCurve:
    """Bootstrap the discount curve from mid market quotes.

    Parameters
    ----------
    reference_date
        Curve reference (settlement) date, i.e. the date where B = 1.
    depos
        Deposit quotes indexed by maturity date, columns ``BID``/``ASK`` as
        decimal rates (4.25% -> 0.0425).
    futures
        Futures quotes indexed by contract date, columns ``BID``/``ASK`` as
        *prices* plus ``Settle``/``Expiry`` dates.
    swaps
        Par swap quotes indexed by maturity date, columns ``BID``/``ASK`` as
        decimal rates.
    shock
        Parallel shift applied to every market **rate**, in decimal
        (1 bp -> 1e-4).  Futures are quoted as prices, so the shift is applied
        to the implied rate ``(100 - price)/100`` and *not* to the price
        itself -- applying it to the price would turn a +1 bp rate shift into a
        -100 bp shift on the futures strip.

    Returns
    -------
    DiscountCurve
    """
    reference_date = pd.Timestamp(reference_date)
    pillar_dates: list[pd.Timestamp] = [reference_date]
    dfs: list[float] = [1.0]

    # ---------------------------------------------------------------- deposits
    depo_dates = depos.index[depos.index <= futures.index[0]].to_list()
    depo_rates = depos.loc[depo_dates, ["BID", "ASK"]].mean(axis=1).to_numpy() + shock
    depo_taus = np.array(
        [year_fraction_act(reference_date, d, ACT_360) for d in depo_dates]
    )
    depo_dfs = 1.0 / (1.0 + depo_rates * depo_taus)

    pillar_dates += list(depo_dates)
    dfs += list(depo_dfs)

    # ----------------------------------------------------------------- futures
    two_year_swap_date = swaps.index[1]
    strip = _futures_used(futures, two_year_swap_date)
    prices = strip[["BID", "ASK"]].mean(axis=1).to_numpy()
    forward_rates = (100.0 - prices) / 100.0 + shock

    for rate, (_, contract) in zip(forward_rates, strip.iterrows()):
        t_start, t_end = contract["Settle"], contract["Expiry"]
        tau = year_fraction_act(t_start, t_end, ACT_360)
        # B(t0, t_start) is read off the curve built so far (deposits, then
        # previously bootstrapped futures): the strip is not contiguous, so the
        # start date generally falls between two pillars.
        df_start = DiscountCurve(pd.DatetimeIndex(pillar_dates), np.array(dfs)).discount(
            t_start
        )
        df_end = df_start / (1.0 + rate * tau)
        pillar_dates.append(pd.Timestamp(t_end))
        dfs.append(df_end)

    # ------------------------------------------------------------------- swaps
    swap_rates = swaps[["BID", "ASK"]].mean(axis=1).to_numpy() + shock
    curve_so_far = DiscountCurve(pd.DatetimeIndex(pillar_dates), np.array(dfs))

    # The 1y point is taken from the futures strip; it only seeds the BPV.
    one_year_date = swaps.index[0]
    bpv = curve_so_far.discount(one_year_date) * year_fraction_30e_360(
        reference_date, one_year_date
    )
    previous_date = one_year_date

    for rate, swap_date in zip(swap_rates[1:], swaps.index[1:]):
        accrual = year_fraction_30e_360(previous_date, swap_date)
        df = (1.0 - rate * bpv) / (1.0 + rate * accrual)
        pillar_dates.append(pd.Timestamp(swap_date))
        dfs.append(df)
        bpv += df * accrual
        previous_date = swap_date

    return DiscountCurve(pd.DatetimeIndex(pillar_dates), np.array(dfs))
