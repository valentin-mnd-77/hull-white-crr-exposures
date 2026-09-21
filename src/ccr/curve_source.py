"""Single entry point for the discount curve, with an automatic offline fallback.

When the five CSV files of the market data set are present, the curve is
bootstrapped from them.  When they are not -- the normal situation after
cloning, since the data is not redistributed -- the Svensson fallback of
:mod:`ccr.synthetic_curve` is used instead, so the whole analysis still runs.

The two sources expose the same interface, including a ``build(shock)``
function that rebuilds the curve under a parallel rate shock: that is what the
CVA rate sensitivity needs, and it spares the rest of the code from knowing
where the curve came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .curve import DiscountCurve, bootstrap
from .market_data import MarketData, load_market_data
from .synthetic_curve import SETTLEMENT_DATE, TODAY, synthetic_curve

MARKET_FILES = ("dt.csv", "depos.csv", "futures.csv", "expiry.csv", "swaps.csv")


@dataclass(frozen=True)
class CurveSource:
    """A discount curve together with its provenance."""

    name: str
    today: pd.Timestamp
    settlement_date: pd.Timestamp
    curve: DiscountCurve
    build: Callable[[float], DiscountCurve]
    market: MarketData | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.market is None


def market_data_available(data_dir: str | Path = "data") -> bool:
    data_dir = Path(data_dir)
    return all((data_dir / name).exists() for name in MARKET_FILES)


def load_curve(data_dir: str | Path = "data", source: str = "auto") -> CurveSource:
    """Return the discount curve from the best available source.

    Parameters
    ----------
    data_dir
        Directory holding the market data CSV files.
    source
        ``"auto"`` (market data if present, fallback otherwise), ``"market"``
        or ``"synthetic"``.
    """
    if source not in {"auto", "market", "synthetic"}:
        raise ValueError("source must be 'auto', 'market' or 'synthetic'")

    use_market = source == "market" or (source == "auto" and market_data_available(data_dir))

    if use_market:
        if not market_data_available(data_dir):
            raise FileNotFoundError(
                f"market data not found in {data_dir}; see data/README.md"
            )
        market = load_market_data(data_dir)

        def build(shock: float = 0.0) -> DiscountCurve:
            return bootstrap(
                market.settlement_date, market.depos, market.futures, market.swaps, shock=shock
            )

        return CurveSource(
            name="market data (bootstrap)",
            today=market.today,
            settlement_date=market.settlement_date,
            curve=build(0.0),
            build=build,
            market=market,
        )

    def build_synthetic(shock: float = 0.0) -> DiscountCurve:
        return synthetic_curve(shift=shock, reference_date=SETTLEMENT_DATE)

    return CurveSource(
        name="synthetic (Svensson fallback)",
        today=TODAY,
        settlement_date=SETTLEMENT_DATE,
        curve=build_synthetic(0.0),
        build=build_synthetic,
        market=None,
    )
