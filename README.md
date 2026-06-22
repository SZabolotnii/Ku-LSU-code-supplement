# Verification package — "One Series, Many Tasks"

Self-contained, open reproduction of every computational and formal claim in the paper
*One Truncated Likelihood Expansion: Estimation, Testing, and Classification as a Single
Captured-Fraction Functional* (S. V. Zabolotnii, 2026).

> **Open repository:** <https://github.com/SZabolotnii/Ku-LSU-code-supplement> — this directory is
> the canonical source for that public code supplement.

All data are **synthetic** Monte-Carlo samples generated in code (global seed `2026`); there is
no external dataset. Two independent artifacts:

| Part | What it certifies | How |
|---|---|---|
| `python/` | The fifteen numerical checks — A1–A2, T1–T4, A3, D1, H1, L1 (unification) plus U1, P1, P2, S1, C1 (uncertainty, power/ROC, RMSE, data-driven selection, completeness) behind §5–§6 | `numpy` Monte-Carlo, seed 2026 |
| `lean/`   | The Gaussian-anchor detection identities (Thm. *Detection bridge* + the GSA corollary) | Lean 4 / Mathlib v4.26.0, 0 `sorry` |

## Quick start

```bash
# 1. Numerical checks (several minutes; ~multi-GB peak RAM from 4M-sample arrays;
#    the U1/P1/P2/S1/C1 operating-characteristic checks add Monte-Carlo power/RMSE studies)
python3 -m pip install -r requirements.txt
python3 python/run_three_branch_unification.py        # prints "UNIFICATION VERIFICATION : PASS (15 checks)"

# 2. Figures (recomputed from the gate; nothing hardcoded)
python3 python/make_figures.py                         # writes figs/*.{png,pdf}

# 3. Machine-checked Lean core (standalone; needs the Lean toolchain via elan)
cd lean && lake exe cache get && lake build            # "Build completed successfully", 0 sorry
```

A `Makefile` wraps these: `make gate`, `make figures`, `make lean`, `make all`.

## What to expect

`python/expected_output.txt` is the recorded stdout of one pinned run (seed 2026; numpy 2.4.4,
Python 3.13/3.14). The script ends with:

```
UNIFICATION VERIFICATION : PASS  (15 checks)
  A1 ✓  A2 ✓  T1 ✓  T2 ✓  T3 ✓  A3 ✓  T4 ✓  D1 ✓  H1 ✓  L1 ✓
  U1 ✓  P1 ✓  P2 ✓  S1 ✓  C1 ✓
```

Monte-Carlo quantities are compared against **tolerances**, not bit-for-bit: trailing digits may
differ across BLAS builds, but every PASS criterion is robust. The headline numbers:

- **Termination (Gaussian):** estimation `g=1`, detection `c=1`, decomposition `R²` jumps `0→1` at `x²`.
- **Joint rise (Laplace):** `g 0.50→0.98`, `c 0.71→0.94`, `R²(log p) 0.00→1.00` (terminates at `|x|`).
- **Covariance-shift anchor:** order-1 `c≈10⁻⁶` → order-2 `c=1`; Ghosh head = LLR to `2.7e-15`;
  sequential `J(s)/‖z‖² = c` to `1.2e-13`.
- **Local identity:** `c(δ) → g` as `δ→0` (`|c−g|: 0.088 → 0.005`).
- **Heavy tails:** Student-t(3) fractional `g 0.88→0.96` and Cauchy `g 0.60→0.75`, where the
  monomial Gram entry `E[x⁴]≈5152` diverges (the polynomial/Hermite basis is not `L²`).

## The fifteen checks

*Unification (10).* `A1,A2` Gaussian termination (score linear; QDA = true LLR). `T1,T2` estimation
efficiency rises and equals the empirical ARE. `T3` detection captured-ratio rises. `A3` GSA
covariance-shift anchor (designed blindness `S=1→S=2`; Ghosh = Bayes; `F·K=Y` collinear). `T4`
sequential information ratio `J(s)/‖z‖²` equals `c`. `D1` decomposition `R²(log p)` terminates /
rises. `H1` heavy-tail fractional advantage. `L1` local cross-branch identity `c(δ)→g`.

*Operating characteristics & completeness (5, added in revision).* `U1` uncertainty quantification —
headline readings as mean ± standard error with 95% CIs over independent seeds. `P1` power: the
order-`m` projected-score test (level calibrated to 0.05) has power that rises with order and tracks
the captured-fraction theory `Φ(√(gI)h−z_α)`. `P2` estimation: one-step PMM estimator RMSE falls and
empirical efficiency rises with order, tracking the analytic `g`. `S1` data-driven selection —
held-out captured fraction picks the order `m`; a Hill tail-index estimate sets the admissible
exponent ceiling `α̂/2`. `C1` completeness — `κ_m(generic z)→1` as the fractional exponents densify in
`(0,α/2)`, on both a light tail (Laplace) and a heavy tail (Student-`t₃`), confirming the
completeness theorem.

## Layout

```
verification/
  README.md  LICENSE  requirements.txt  Makefile  run_all.sh  .gitignore
  python/  run_three_branch_unification.py  make_figures.py  expected_output.txt
  lean/    lakefile.lean  lean-toolchain  lake-manifest.json  LikelihoodSeries.lean
  figs/    (generated)
```

The Lean project is standalone (pinned `leanprover/lean4:v4.26.0` + Mathlib via the committed
`lake-manifest.json`); it does **not** depend on any sibling repository. `lake exe cache get`
fetches prebuilt Mathlib oleans so the single core file builds in seconds.

## License & citation
Code: MIT (`LICENSE`). If you use this package, please cite the paper (see `CITATION.cff`).
