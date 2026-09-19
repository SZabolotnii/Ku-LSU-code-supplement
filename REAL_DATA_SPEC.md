# Ku-LSU real-data study — pre-registration

**Written 2026-09-16, BEFORE any analysis code was run.** Answers Reviewer 1's request
(`STPA-D-26-00487`, major revision) for "a decent real data set". Criteria and abort conditions are
fixed here; the run reports whatever comes out (see R-5).

*Editorial note, 2026-09-19.* Four passages of the original working document named other,
unpublished manuscripts of ours and internal file locations. They are reworded here to describe the
same facts without those identifiers. No hypothesis, criterion, threshold, arm, split or outcome is
altered; the amendments and the correction below stand exactly as they were written on 2026-09-16.

## Why this design and not a benchmark campaign

The paper's operational claim (Prop. `prop:power`) is that the captured fraction of the score *is*
the asymptotic relative efficiency and the local power. That is a **local** statement about a
correctly specified family. On real data the family is misspecified and the observations are
dependent, so the claim can fail — and in two earlier unpublished pilots of our own, on other
data, it did:

- on partial-discharge antenna recordings the deflection `J` peaked at exponent `a = 0.4` while
  held-out AUC rose monotonically to `|x|³`: `J` and AUC ranked the dictionary in **opposite**
  directions.
- in a second pilot `J(s)` rose monotonically exactly as the classical theorem says, and predicted
  nothing about the decision rule's operating curve.

The diagnosis in both cases was the same: the mechanism works when the heavy tail is a **nuisance
shared by both classes**, and inverts when the heavy tail **is** the class contrast. This study is
therefore built so that it is informative whichever way it comes out: it contains one sub-case of
each kind, and the adverse sub-case carries a pre-registered prediction *against* the functional.

## Data

- FRED series `DEXJPUS` (Japanese yen per US dollar, daily, from 1971-01-04, 14 469 rows). Public
  domain, no licence or anonymity constraint.
- **Excluded by prior commitment:** two further heavy-tailed series available to us — a crude-oil
  price series and a radar-clutter set — are already used in other work of ours, and are excluded
  here so that this study reports no data reported elsewhere.
- Series transformed to daily log-returns `r_t = log(P_t / P_{t-1})`; non-trading gaps and missing
  quotes dropped. No other cleaning.
- **Split by date, fixed here:** first 60 % of observations = TRAIN (exponent selection, all
  coefficients, all critical values), last 40 % = TEST (every reported metric). No observation is
  used for both. No re-splitting, no repetition with a different split.

## R-0 — admissibility gate (abort condition)

Hill tail-index estimate `alpha_hat` on `|r|` (upper 10 %) computed on TRAIN.

- If `alpha_hat >= 4`: the fourth moment is unproblematic, the paper's regime does not apply, and
  **the study is declared uninformative and reported as such** — not quietly replaced by another
  series.
- If `alpha_hat < 4`: proceed. The admissible exponent ceiling is `alpha_hat / 2`.

## Branch E — estimation (real noise, known truth)

Disjoint blocks of `n = 250` consecutive TEST returns. Each block is centred by its own median, so
the true location is 0 by construction; a known shift `delta` is then added. The **noise is real**
(dependent, heavy-tailed, non-stationary); only the estimand is synthetic. `delta` is fixed at
`0.25 * MAD(train)`.

Arms: sample mean | median | Huber M (k = 1.345) | polynomial PMM {x, x³} | fractional PMM (odd
half, exponents from R-0 on TRAIN) | — no oracle MLE exists here, which is the point.

- **E-1** (primary): fractional PMM RMSE `<=` polynomial PMM RMSE, paired bootstrap 95 % CI on the
  difference excluding 0. PASS / FAIL.
- **E-2**: fractional PMM RMSE within 5 % of the best arm. PASS / FAIL.

## Branch T — testing

Same blocks. H0 = no shift; H1 = shift `delta`. Critical values calibrated to size 0.05 **on TRAIN
blocks**, applied unchanged to TEST. Arms: t-test | sign | Wilcoxon | projected score (Hermite) |
projected score (fractional).

- **T-1**: paired difference (fractional  best classical arm) with paired SE.
  `z >= 2` = win, `|z| < 2` = tie, `z <= -2` = loss. All three are reportable outcomes.

## Branch C — classification, two sub-cases

### C-loc — heavy tail is a SHARED nuisance (favourable regime)

Class 0 = TEST returns; class 1 = the same returns shifted by `delta`. The tail is identical in both
classes; only location differs.

- **C-1**: held-out AUC of the fractional projection `>=` that of the polynomial projection, paired
  bootstrap 95 % CI excluding 0. PASS / FAIL.

### C-scale — heavy tail IS the contrast (adverse regime)

Labels from realized volatility computed on a **disjoint** trailing window (21 trading days ending
the day *before* the observation, so no observation contributes to its own label): class 1 if that
window's realized volatility is in the top tercile of TRAIN, class 0 if in the bottom tercile,
middle tercile discarded.

Six dictionaries are ranked two ways: by TRAIN captured fraction / information functional, and by
TEST AUC.

- **C-2 (adverse prediction):** the Spearman rank correlation between the two rankings is
  **non-positive**. This is a prediction *against* the captured fraction as a selector in this
  regime. If it comes out strongly positive, the prior finding is refuted for this data class and
  the paper says so.

## R-5 — reporting rule

Every arm and every criterion above is printed with its outcome, PASS or FAIL, in
`expected_output_realdata.txt`. No arm is dropped after seeing its number; no criterion is moved.
A failed criterion is reported as failed in the manuscript.

## Amendment 1 — 2026-09-16, before any outcome was observed

The first execution of `run_realdata.py` aborted with a `ZeroDivisionError` while printing the
estimation table, **before any RMSE value was displayed**. Inspecting the cause exposed two defects
in the design above. Both are fixed here; no criterion is relaxed and no threshold moved.

1. **The median arm was exact by construction.** The design centred each block by its own median and
   then added `delta`, so `median(block) = delta` identically and that arm had RMSE exactly zero.
   The estimand is replaced by a **paired within-block difference**: for each block `b`, every
   estimator is applied to `b` and to `b + delta`, and its error is
   `(est(b + delta) - est(b)) - delta`. This is exact for any location-equivariant estimator — all
   five arms are — requires no artificial centring, and privileges none of them.

2. **Too few blocks.** `DEXJPUS` alone yields 22 TEST blocks of 250, which cannot resolve a power
   difference. Two changes: block length `n = 125`, and the study pools **four FX series** ---
   `DEXJPUS`, `DEXUSUK`, `DEXCAUS`, `DEXUSEU` — each split 60/40 by its own date range
   before pooling. `SP500` stays out: it is a different asset class and only 2611 rows, and mixing it
   would confound the tail regime. The two series excluded above remain excluded.
   The admissibility gate R-0 is now evaluated on the pooled TRAIN returns.

Everything else — the split rule, the criteria E-1, E-2, T-1, C-1, the adverse prediction C-2, and
the reporting rule R-5 — stands unchanged.

## Amendment 2 — 2026-09-16, before any outcome was observed

Amendment 1's paired within-block estimand is **degenerate** and is withdrawn. For a
location-equivariant estimator, `est(b + delta) - est(b) = delta` identically, whatever the sample;
the construction cancels the sampling noise it was supposed to measure, and every arm returned RMSE
exactly zero. Again this was visible from the structure of the output (all five arms identical at
`0.000000`), not from any arm's relative standing.

**Replacement estimand for branch E.** Each series' TEST segment is centred once by *that segment's
grand median*, a single constant estimated from roughly 5 000 observations. The true location of
every block is then 0 up to an error of order `1/sqrt(5000)`, which is negligible beside a block's
own error of order `1/sqrt(125)`. Each estimator is applied to each block and its error is the
estimate itself. No arm is privileged: the centring constant comes from 40 times more data than any
block, so a block's own median is not exact.

**Secondary analysis added to branch T.** The pre-registered protocol transports critical values from
TRAIN to TEST. Should the realized TEST sizes depart materially from 0.05, a power comparison between
arms is confounded by calibration rather than by efficiency. The pre-registered TRAIN-calibrated
result stands as the primary outcome and is reported first with its realized sizes; a secondary,
clearly labelled size-corrected comparison (critical values from TEST null blocks) is reported
beside it. The primary verdict is **not** replaced by the secondary one.

## Amendment 3 — 2026-09-16, written AFTER the primary outcomes were observed

Unlike Amendments 1 and 2, this one is written with the results in hand and is therefore **not** a
pre-registration of the primary study. It is a labelled diagnostic, and its outcome must be read as
exploratory whichever way it falls.

**What the primary study found.** E-1 FAIL, E-2 PASS, T-1 TIE (primary and secondary), C-1 FAIL,
C-2 CONFIRMED. All five estimators lie within 4.8 % of one another; the fractional test exactly ties
Wilcoxon once both are calibrated on the same data; the fractional projection is *significantly
worse* than the polynomial one on the favourable location contrast; and the captured fraction ranks
dictionaries essentially at random against held-out AUC. The efficiency gain the synthetic study
shows on i.i.d. Student-$t_3$ (0.95 against 0.49) does **not** transfer to daily FX returns.

**Diagnosis to be tested.** Daily FX returns are volatility-clustered: the heavy marginal tail is
produced largely by a time-varying conditional scale rather than by a heavy conditional law. A
dictionary whose exponents are tuned to the *pooled marginal* tail index is then mismatched to every
individual block, and no fixed dictionary can be right for all of them. If that is the cause, the
mechanism should reappear once the clustering is removed.

**Test.** Repeat branches E and C-loc on **block-devolatilized** returns: each block is divided by
its own robust scale (MAD within the block) before any basis is applied, which removes the
between-block scale variation while leaving the within-block shape untouched. Prediction, fixed
before the run: on devolatilized returns the fractional arm recovers an edge over the polynomial arm
on at least one of E-1 and C-1. If it does not, the volatility-clustering explanation is wrong and
the failure is more fundamental.

## Correction 4 — 2026-09-16, after the first analysis of C-2 was reported

The first analysis of criterion C-2 was reported as CONFIRMED, with Spearman `-0.200`. **That report
was wrong and is withdrawn.** Two defects, both in our implementation, not in the data:

1. **The captured fraction was computed in sample.** It was the training `R^2` of the fitted design,
   not a held-out quantity. That is not the selector this paper proposes — check S1 of the controlled
   study uses a held-out captured fraction — and an in-sample `R^2` rises with the number of columns,
   so the richest dictionary tops the ranking by construction. Recomputed out of sample
   (coefficients on the first 70 % of TRAIN, `kappa` on the remaining 30 %), the correlation is
   **`+0.029`**: the pre-registered prediction is **refuted**, not confirmed.

2. **The criterion was never testable on these data.** The six dictionaries span an AUC range of
   `0.0010` (0.6912 to 0.6922), and only two of the five non-best arms are separable from the best at
   all, by margins between `0.0002` and `0.0017`. A ranking rule cannot be shown to succeed or fail
   among six options that are, for practical purposes, identical.

C-2 is therefore reported in the manuscript as **NOT TESTABLE** rather than confirmed or refuted.
Both numbers — in-sample `-0.200` and held-out `+0.029` — are printed in the program output so the
difference is visible. The reason the dictionaries tie is that a volatility contrast is carried
almost entirely by `|x|`, and every admissible system contains an even element.

This correction was prompted by a reader asking why the study was being described as a negative
result when the integer-power dictionaries had plainly worked. That question was right.

## Provenance

- Script: `python/run_realdata.py` of the reproduction package, global seed 2026.
- Data: the four CSV files shipped in `data/` of that package; provenance in `data/SOURCE.md`.
