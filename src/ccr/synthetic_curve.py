"""Offline fallback curve, so that the project runs without the market data set.

The raw market quotes are not redistributed with this repository.  To keep the
analysis reproducible for anyone who clones it, this module ships a Svensson
parametrisation of the zero curve, fitted once by least squares on the zero
rates of the curve bootstrapped from the original data set::

    z(t) = b0 + b1 * f1(t/l1) + b2 * f2(t/l1) + b3 * f2(t/l2)
    f1(x) = (1 - e^{-x}) / x,   f2(x) = f1(x) - e^{-x}

Only six derived parameters are distributed, not the quotes themselves.

**Accuracy.**  Beyond two years the fit stays within 2.5 bp of the bootstrapped
curve.  Below two years it cannot follow the February 2008 money-market
structure -- a hump around six months followed by a dip priced by the futures
strip -- and misses it by up to about 20 bp.  The consequences on the exposure
metrics are quantified in the README; ``tests/test_synthetic_curve.py`` pins
the accuracy so that any change of parameters is caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curve import DiscountCurve
from .dates import ACT_365, DateLike, business_date_offset, year_fraction_act

#: Reference dates of the original data set.
TODAY = pd.Timestamp("2008-02-15")
SETTLEMENT_DATE = pd.Timestamp("2008-02-19")


@dataclass(frozen=True)
class SvenssonParameters:
    """Continuously compounded, ACT/365 zero-rate parametrisation."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lambda1: float
    lambda2: float

    def zero_rate(self, tau: np.ndarray | float) -> np.ndarray:
        tau = np.maximum(np.asarray(tau, dtype=float), 1e-8)  # z(0) is the limit b0 + b1
        x1, x2 = tau / self.lambda1, tau / self.lambda2
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        return self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3


#: Fitted on the zero rates of the EUR curve bootstrapped on 19 February 2008
#: (deposits, 3M Euribor futures and par swaps, 58 pillars from 1d to 50y).
SVENSSON_FEB_2008 = SvenssonParameters(
    beta0=0.03557842,
    beta1=0.00651726,
    beta2=0.03976928,
    beta3=-0.02635580,
    lambda1=12.05232690,
    lambda2=2.34634049,
)


def synthetic_curve(
    shift: float = 0.0,
    reference_date: DateLike = SETTLEMENT_DATE,
    params: SvenssonParameters = SVENSSON_FEB_2008,
) -> DiscountCurve:
    """Discount curve generated from the Svensson parameters.

    Pillars are monthly up to 15 years and annual up to 50 years, dense enough
    for the zero-rate interpolation of :class:`DiscountCurve` to be exact to
    well below a basis point on a smooth curve.

    Parameters
    ----------
    shift
        Parallel shift of the zero rates, in decimal (1 bp -> 1e-4).  This is
        the offline counterpart of the market-rate shock applied in
        :func:`ccr.curve.bootstrap`, used for the rate sensitivity of the CVA.
    """
    reference_date = pd.Timestamp(reference_date)
    dates = [reference_date]
    dates += [business_date_offset(reference_date, month_offset=m) for m in range(1, 181)]
    dates += [business_date_offset(reference_date, year_offset=y) for y in range(16, 51)]

    taus = np.array([year_fraction_act(reference_date, d, ACT_365) for d in dates])
    zeros = params.zero_rate(taus) + shift
    return DiscountCurve(pd.DatetimeIndex(dates), np.exp(-zeros * taus))
