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

# Data source — EEGdenoiseNet muscular-artefact epochs (Study 2)

| File | Shape | Rate | SHA-256 |
|---|---|---|---|
| `EMG_all_epochs.npy` | (5598, 512) float64 | 256 Hz, 2 s epochs | `8bb16dff87582823488dd2189dba9d8fa019a8cbc9af9a8a19acb421b3fcfb16` |

- **Source.** Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., Liu, Q. (2021). EEGdenoiseNet: a
  benchmark dataset for deep learning solutions of EEG denoising. *Journal of Neural Engineering*
  18(5), 056057. Data repository: G-Node GIN `NCClab/EEGdenoiseNet`
  (<https://gin.g-node.org/NCClab/EEGdenoiseNet>), file `EMG_all_epochs.npy`, unmodified.
- **Licence.** The data repository is released under **CC0 1.0 Universal** (its `LICENSE` file);
  the authors' separate code repository on GitHub is MIT and is not redistributed here.
- **Use in this package.** `run_realdata_study2.py` reads this file only; the study's
  pre-registration, data-selection rule and screen table are in `../REAL_DATA_SPEC.md` (Study 2).
  The EOG and clean-EEG files of the same dataset failed the selection rule and are not shipped.
