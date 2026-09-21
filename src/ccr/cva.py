"""Unilateral CVA and its sensitivities.

Under the usual simplifying assumptions -- constant loss given default, flat
hazard rate, independence between exposure and default, and a bank that cannot
itself default -- the CVA of a netting set is the discounted expected loss::

    CVA(0) = LGD * sum_i EPE(t_i) * B(0, t_i) * ( S(t_{i-1}) - S(t_i) )
    S(t)   = exp(-lambda * t)

Independence is the strongest of these assumptions: it rules out wrong-way
risk, i.e. the tendency of the exposure to be large precisely when the
counterparty is close to default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curve import DiscountCurve, bootstrap
from .dates import ACT_365, DateLike, year_fraction_act
from .engine import SimulationConfig, run_simulation
from .exposure import CSAParameters
from .hull_white import HullWhiteParameters
from .market_data import MarketData
from .swap import InterestRateSwap


@dataclass(frozen=True)
class CreditParameters:
    """Credit assumptions of the counterparty."""

    hazard_rate: float = 0.02  # 200 bp flat
    lgd: float = 0.6


def compute_cva(
    epe_profile: pd.Series,
    curve: DiscountCurve,
    credit: CreditParameters,
    reference_date: DateLike | None = None,
) -> float:
    """Discrete approximation of the unilateral CVA.

    The exposure is sampled on the simulation grid, and the marginal default
    probability over ``(t_{i-1}, t_i]`` is charged at the exposure observed at
    ``t_i``.  This is the conservative convention: defaults are assumed to
    occur at the end of each interval.
    """
    reference_date = (
        curve.reference_date if reference_date is None else pd.Timestamp(reference_date)
    )
    taus = np.array(
        [year_fraction_act(reference_date, d, ACT_365) for d in epe_profile.index]
    )
    survival = np.exp(-credit.hazard_rate * taus)
    previous_survival = np.concatenate(([1.0], survival[:-1]))
    marginal_default = previous_survival - survival

    discounts = curve.discounts(epe_profile.index)

    return float(credit.lgd * np.sum(epe_profile.to_numpy() * discounts * marginal_default))


def cva_sensitivities(
    market: MarketData,
    curve: DiscountCurve,
    swap: InterestRateSwap,
    swap_netting: InterestRateSwap,
    break_date: DateLike,
    csa: CSAParameters,
    hw_params: HullWhiteParameters,
    credit: CreditParameters,
    config: SimulationConfig,
    scenario: str = "base",
    rate_bump: float = 1e-4,
    sigma_bump: float = 1e-4,
    hazard_bump: float = 1e-4,
) -> pd.DataFrame:
    """CVA sensitivities to the three risk factors that actually drive it.

    Each sensitivity is obtained by **full revaluation**: the exposure profile
    is re-simulated under the bumped input rather than held fixed.  For the
    rate sensitivity this matters more than anything else -- a parallel shift
    moves the mark-to-market of the swap, hence the whole EPE profile, an
    effect an order of magnitude larger than the change in discounting alone.

    All runs reuse the same seed, so the reported differences are free of
    Monte Carlo noise to first order (common random numbers).

    Returns
    -------
    pd.DataFrame
        One row per risk factor with the bumped CVA and the difference.
    """
    base_results = run_simulation(
        curve, swap.settlement_date, swap, swap_netting, break_date, csa, hw_params, config
    )
    base_cva = compute_cva(base_results.epe[scenario], curve, credit, swap.settlement_date)

    rows = [("base", np.nan, base_cva, 0.0)]

    # --- interest rates: re-bootstrap, re-simulate, keep the traded fixed rate
    bumped_curve = bootstrap(
        curve.reference_date, market.depos, market.futures, market.swaps, shock=rate_bump
    )
    ir_results = run_simulation(
        bumped_curve,
        swap.settlement_date,
        swap,
        swap_netting,
        break_date,
        csa,
        hw_params,
        config,
    )
    ir_cva = compute_cva(
        ir_results.epe[scenario], bumped_curve, credit, swap.settlement_date
    )
    rows.append(("IR01 (+1 bp parallel)", rate_bump, ir_cva, ir_cva - base_cva))

    # --- credit: hazard rate of the counterparty, exposure unchanged
    bumped_credit = CreditParameters(
        hazard_rate=credit.hazard_rate + hazard_bump, lgd=credit.lgd
    )
    cs_cva = compute_cva(
        base_results.epe[scenario], curve, bumped_credit, swap.settlement_date
    )
    rows.append(("CS01 (+1 bp hazard rate)", hazard_bump, cs_cva, cs_cva - base_cva))

    # --- volatility: re-simulate with a wider distribution of future rates
    bumped_params = HullWhiteParameters(
        mean_reversion=hw_params.mean_reversion, sigma=hw_params.sigma + sigma_bump
    )
    vega_results = run_simulation(
        curve,
        swap.settlement_date,
        swap,
        swap_netting,
        break_date,
        csa,
        bumped_params,
        config,
    )
    vega_cva = compute_cva(
        vega_results.epe[scenario], curve, credit, swap.settlement_date
    )
    rows.append(("Vega (+1 bp of sigma)", sigma_bump, vega_cva, vega_cva - base_cva))

    return pd.DataFrame(rows, columns=["risk_factor", "bump", "cva", "delta_cva"]).set_index(
        "risk_factor"
    )
