"""Hull-White (extended Vasicek) one-factor model.

The short rate is written ``r_t = phi(t) + x_t`` where ``phi`` is the
deterministic shift that reprices the initial curve exactly and ``x`` is a
driftless Ornstein-Uhlenbeck process::

    dx_t = -a x_t dt + sigma dW_t,   x_0 = 0

Discount factors have an affine representation that avoids re-integrating
``phi`` in every scenario::

    B(s, tau) = A(s, tau) * exp(-x_s * C(s, tau))
    A(s, tau) = B(0, tau) / B(0, s) * exp(-1/2 * I(s, tau))
    C(s, tau) = (1 - exp(-a (tau - s))) / a
    I(s, tau) = int_0^s [ sigma(u, tau)^2 - sigma(u, s)^2 ] du
    sigma(u, x) = sigma * (1 - exp(-a (x - u))) / a

Note the **minus** sign in front of the integral: with a plus sign the discount
factors are overstated by ~150 bp at the 10y point seen from 5y, and the whole
exposure profile is biased.  ``I`` is available in closed form through the
variance of the integrated state variable,

    V(tau) = Var[ int_0^tau x_u du ]
           = sigma^2/a^2 * ( tau + 2/a e^{-a tau} - 1/(2a) e^{-2a tau} - 3/(2a) )
    -1/2 * I(s, tau) = 1/2 * ( V(s) - V(tau) + V(tau - s) )

which is the classical Hull-White expression.  The two forms are checked
against each other (closed form vs numerical quadrature) in
``tests/test_hull_white.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curve import DiscountCurve
from .dates import MODEL_BASIS, DateLike, year_fraction_act


@dataclass(frozen=True)
class HullWhiteParameters:
    """Calibrated parameters of the model."""

    mean_reversion: float
    sigma: float

    def __post_init__(self) -> None:
        if self.mean_reversion <= 0:
            raise ValueError("mean reversion speed must be strictly positive")
        if self.sigma <= 0:
            raise ValueError("volatility must be strictly positive")


def integrated_state_variance(tau: float, params: HullWhiteParameters) -> float:
    """``V(tau) = Var[ int_0^tau x_u du ]``."""
    a, sigma = params.mean_reversion, params.sigma
    return (sigma**2 / a**2) * (
        tau + 2.0 / a * np.exp(-a * tau) - 1.0 / (2 * a) * np.exp(-2 * a * tau) - 3.0 / (2 * a)
    )


def state_variance(tau: float, params: HullWhiteParameters) -> float:
    """``Var[x_tau] = sigma^2 / (2a) * (1 - exp(-2 a tau))``."""
    a, sigma = params.mean_reversion, params.sigma
    return sigma**2 / (2 * a) * (1.0 - np.exp(-2 * a * tau))


def simulate_state_paths(
    grid: pd.DatetimeIndex,
    start_date: DateLike,
    params: HullWhiteParameters,
    n_scenarios: int,
    seed: int | None = None,
    x0: float = 0.0,
) -> np.ndarray:
    """Simulate ``x`` on ``grid`` with the exact (non-discretised) transition.

    Given ``x_t``, ``x_{t+dt}`` is Gaussian with mean ``x_t e^{-a dt}`` and
    variance ``sigma^2/(2a)(1 - e^{-2 a dt})``, so the scheme carries no
    discretisation bias regardless of the step size.  Time is measured in
    ACT/365, the same basis as the affine coefficients.

    Returns
    -------
    np.ndarray
        Shape ``(n_scenarios, len(grid))``.
    """
    rng = np.random.default_rng(seed)
    grid = pd.DatetimeIndex(grid)

    paths = np.empty((n_scenarios, len(grid)), dtype=float)
    x = np.full(n_scenarios, float(x0))
    previous_date = pd.Timestamp(start_date)

    for i, date in enumerate(grid):
        dt = year_fraction_act(previous_date, date, MODEL_BASIS)
        decay = np.exp(-params.mean_reversion * dt)
        std = np.sqrt(state_variance(dt, params))
        # Normal draws are generated step by step rather than as one large
        # matrix: it halves peak memory at 250k scenarios and leaves the
        # sequence reproducible through the seeded generator.
        x = x * decay + std * rng.standard_normal(n_scenarios)
        paths[:, i] = x
        previous_date = date

    return paths


def affine_coefficients(
    valuation_date: DateLike,
    pricing_grid: pd.DatetimeIndex,
    params: HullWhiteParameters,
    curve: DiscountCurve,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A(s, tau)`` and ``C(s, tau)`` for every ``tau`` in the grid.

    Dates at or before ``valuation_date`` get ``A = 1`` and ``C = 0`` so that
    ``B(s, tau) = 1``; they are masked out by the pricing functions anyway.
    """
    valuation_date = pd.Timestamp(valuation_date)
    pricing_grid = pd.DatetimeIndex(pricing_grid)
    reference_date = curve.reference_date

    tau_s = year_fraction_act(reference_date, valuation_date, MODEL_BASIS)
    df_s = curve.discount(valuation_date)
    v_s = integrated_state_variance(tau_s, params)

    A = np.ones(len(pricing_grid))
    C = np.zeros(len(pricing_grid))

    for j, date in enumerate(pricing_grid):
        if date <= valuation_date:
            continue
        tau_T = year_fraction_act(reference_date, date, MODEL_BASIS)
        exponent = 0.5 * (
            v_s
            - integrated_state_variance(tau_T, params)
            + integrated_state_variance(tau_T - tau_s, params)
        )
        A[j] = curve.discount(date) / df_s * np.exp(exponent)
        C[j] = (1.0 - np.exp(-params.mean_reversion * (tau_T - tau_s))) / params.mean_reversion

    return A, C


def simulated_discount_factors(
    x_at_valuation: np.ndarray,
    valuation_date: DateLike,
    pricing_grid: pd.DatetimeIndex,
    params: HullWhiteParameters,
    curve: DiscountCurve,
) -> np.ndarray:
    """Discount curve ``B(s, tau)`` in every scenario, shape ``(n_scenarios, n_tau)``."""
    A, C = affine_coefficients(valuation_date, pricing_grid, params, curve)
    return A * np.exp(-np.outer(x_at_valuation, C))
