"""Counterparty credit risk metrics and CSA collateral mechanics.

All quantities are measured from the perspective of party B (the bank).  A
positive mark-to-market is an asset for B and therefore a credit exposure to
counterparty C; a positive variation margin is collateral *held* by B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CSAParameters:
    """Credit Support Annex parameters, per party.

    Independent amounts, haircuts, rounding rules and settlement lags are out
    of scope: collateral is assumed to be posted in cash, immediately after the
    margin call, and remunerated at the risk-free rate.
    """

    threshold_bank: float = 0.0
    threshold_counterparty: float = 0.0
    mta_bank: float = 0.0
    mta_counterparty: float = 0.0
    cap_bank: float = np.inf
    cap_counterparty: float = np.inf

    @classmethod
    def counterparty_fully_collateralised(cls) -> "CSAParameters":
        """The arrangement required by the case study.

        The bank never posts collateral -- an infinite threshold means its
        negative mark-to-market never triggers a call, and a zero cap makes
        that unconditional -- while the counterparty covers its exposure in
        full: zero threshold, zero minimum transfer amount, no cap.
        """
        return cls(
            threshold_bank=np.inf,
            threshold_counterparty=0.0,
            mta_bank=0.0,
            mta_counterparty=0.0,
            cap_bank=0.0,
            cap_counterparty=np.inf,
        )


def update_variation_margin(
    mtm: np.ndarray, variation_margin: np.ndarray, csa: CSAParameters
) -> np.ndarray:
    """Variation margin after a margin call, scenario by scenario.

    The target margin is::

        VM* = max(MtM - THR_C, 0)   if MtM >= 0
              min(MtM + THR_B, 0)   if MtM <  0

    A call is only effective when the required move exceeds the relevant
    minimum transfer amount, and the resulting balance is clipped by the caps.
    """
    target = np.where(
        mtm >= 0,
        np.maximum(mtm - csa.threshold_counterparty, 0.0),
        np.minimum(mtm + csa.threshold_bank, 0.0),
    )
    # threshold_bank = +inf gives min(+inf, 0) = 0: no call on the bank.
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    required = target - variation_margin
    call = np.where(
        required > 0,
        np.where(required >= csa.mta_counterparty, required, 0.0),
        np.where(required <= -csa.mta_bank, required, 0.0),
    )
    return np.clip(variation_margin + call, -csa.cap_bank, csa.cap_counterparty)


def expected_positive_exposure(exposure: np.ndarray) -> float:
    """``EPE(t) = E[ max(Exposure_t, 0) ]``."""
    return float(np.mean(np.maximum(exposure, 0.0)))


def expected_negative_exposure(exposure: np.ndarray) -> float:
    """``ENE(t) = E[ min(Exposure_t, 0) ]``, the mirror metric feeding DVA."""
    return float(np.mean(np.minimum(exposure, 0.0)))


def potential_future_exposure(exposure: np.ndarray, alpha: float = 0.95) -> float:
    """``PFE_alpha(t) = Quantile_alpha( max(Exposure_t, 0) )``."""
    return float(np.quantile(np.maximum(exposure, 0.0), alpha))


def peak_pfe(pfe_profiles: pd.DataFrame) -> pd.DataFrame:
    """Peak-PFE per scenario: ``sup_t PFE_alpha(t)`` and the date where it occurs."""
    return pd.DataFrame(
        {
            "peak_pfe": pfe_profiles.max(axis=0),
            "date": pfe_profiles.idxmax(axis=0),
        }
    )
