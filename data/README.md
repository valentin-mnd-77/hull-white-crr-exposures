# Market data

The EUR market data set used by the notebook (Euribor deposits, futures and swap
quotes as of 15 February 2008) is course material and is **not redistributed
here**. Tests that depend on it skip automatically when the files are absent.

Drop the following five CSV files in this directory to reproduce the analysis.
Dates are formatted `%d-%b-%y` (e.g. `19-Feb-08`); bid and ask are quoted in
percent for rates and in price for futures.

| File | Index | Columns |
|---|---|---|
| `dt.csv` | `Market` (`Today`, `Settlement`) | `TARGET` — the two reference dates |
| `depos.csv` | `Depos` — maturity date | `BID`, `ASK` — deposit rates, in % |
| `futures.csv` | `Futures` — contract date | `BID`, `ASK` — futures prices |
| `expiry.csv` | `Futures` — contract date | `Settle`, `Expiry` — the underlying period |
| `swaps.csv` | `Swaps` — maturity date | `BID`, `ASK` — par swap rates, in % |

The loader (`ccr.market_data.load_market_data`) converts deposit and swap quotes
to decimals, keeps futures as prices, and joins the futures with their
settlement and expiry dates.
