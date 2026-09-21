"""Loading of the February 2008 EUR market data set.

The raw files are not redistributed with this repository (see ``data/README.md``).
Quotes are normalised here once and for all: deposit and swap quotes are
converted from percentages to decimals, futures are kept as prices and joined
with their settlement/expiry dates.  Every other module then assumes decimal
rates, which removes the ``/100`` scattered across the original code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATE_FORMAT = "%d-%b-%y"


@dataclass(frozen=True)
class MarketData:
    """Mid-market quotes and the two reference dates of the data set."""

    today: pd.Timestamp
    settlement_date: pd.Timestamp
    depos: pd.DataFrame  # index: maturity date, columns: BID/ASK (decimal)
    futures: pd.DataFrame  # index: contract date, columns: BID/ASK (price), Settle, Expiry
    swaps: pd.DataFrame  # index: maturity date, columns: BID/ASK (decimal)


def _read_dates(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    raw = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    dates = pd.to_datetime(raw["TARGET"], format=DATE_FORMAT)
    return dates.loc["Today"], dates.loc["Settlement"]


def load_market_data(data_dir: str | Path = "data") -> MarketData:
    """Read the five CSV files of the data set into a :class:`MarketData`."""
    data_dir = Path(data_dir)

    today, settlement_date = _read_dates(data_dir / "dt.csv")

    depos = pd.read_csv(
        data_dir / "depos.csv", usecols=["Depos", "BID", "ASK"], index_col="Depos"
    )
    depos.index = pd.to_datetime(depos.index, format=DATE_FORMAT)
    depos[["BID", "ASK"]] /= 100.0

    futures_prices = pd.read_csv(
        data_dir / "futures.csv", usecols=["Futures", "BID", "ASK"], index_col="Futures"
    )
    futures_prices.index = pd.to_datetime(futures_prices.index, format=DATE_FORMAT)

    expiries = pd.read_csv(
        data_dir / "expiry.csv",
        usecols=["Futures", "Settle", "Expiry"],
        index_col="Futures",
    )
    expiries.index = pd.to_datetime(expiries.index, format=DATE_FORMAT)
    for column in ("Settle", "Expiry"):
        expiries[column] = pd.to_datetime(expiries[column], format=DATE_FORMAT)

    futures = futures_prices.join(expiries).sort_index()

    swaps = pd.read_csv(
        data_dir / "swaps.csv", usecols=["Swaps", "BID", "ASK"], index_col="Swaps"
    )
    swaps.index = pd.to_datetime(swaps.index, format=DATE_FORMAT)
    swaps[["BID", "ASK"]] /= 100.0

    return MarketData(
        today=today,
        settlement_date=settlement_date,
        depos=depos.sort_index(),
        futures=futures,
        swaps=swaps.sort_index(),
    )
