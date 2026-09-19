# Data sources — four FRED daily exchange-rate series

These four CSV files are redistributed with the package so that the real-data study of §8 runs
without a download or an account. They are **not** covered by the package's MIT license: they are
public-domain data of the U.S. Federal Reserve, redistributed under the FRED terms of use.

| File | FRED series | What it is | First obs. | Last obs. | Rows |
|---|---|---|---|---|---|
| `DEXJPUS.csv` | `DEXJPUS` | Japanese yen per U.S. dollar, daily | 1971-01-04 | 2026-06-18 | 14 469 |
| `DEXUSUK.csv` | `DEXUSUK` | U.S. dollars per pound sterling, daily | 1971-01-04 | 2026-06-18 | 14 469 |
| `DEXCAUS.csv` | `DEXCAUS` | Canadian dollars per U.S. dollar, daily | 1971-01-04 | 2026-06-12 | 14 465 |
| `DEXUSEU.csv` | `DEXUSEU` | U.S. dollars per euro, daily | 1999-01-04 | 2026-06-18 | 7 164 |

- **Source.** Board of Governors of the Federal Reserve System (US), retrieved from FRED, Federal
  Reserve Bank of St. Louis, `https://fred.stlouisfed.org/series/<ID>`; CSV endpoint
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`.
- **Retrieved.** June 2026, as published; the files are unmodified.
- **Missing values.** Market holidays appear as an empty second field. `run_realdata.py` drops
  those rows and any non-positive price, then takes log-differences; pooled over the four series
  this leaves **48 603** daily log-returns, which is the number reported in the paper.
- **Two further series** (`SP500`, `DCOILWTICO`) were present in the source folder and were
  excluded by the pre-registration before any analysis; they are not shipped here.
