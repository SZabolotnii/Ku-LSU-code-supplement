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

## Correction 5 - 2026-09-21, internal revision audit

This amendment supersedes the interpretations and numbers of Corrections 2-4 above.
Those entries are retained as historical records, not current results. These changes
were made after examining the initial results and are not pre-registered decisions.

The original per-series 60/40 split is frozen before all processing. The training
median centers both parts; no test median is estimated. Trailing 21-observation
volatility is computed independently within each series, excluding the current
return. Regime thresholds use training observations only. Removing middle-regime
observations does not recompute the split. Regression tests cover split membership,
window boundaries and training invariance to changes in test values.

The four series have different calendar cutoffs. Pooled training periods overlap
test periods of other currencies, so this remains a descriptive per-series holdout,
not globally forward-looking forecasting. Dependence between training and test
samples is not ruled out. No prospective forecasting claim is made.

E-1/E-2 cannot identify parameter risk: neither a grand median nor an injected shift
makes the unknown location of each market block known. The output now reports RMS
deviations about the training reference without iid bootstrap significance.
T reports descriptive baseline/shifted rejection rates without iid paired z-tests
or WIN/TIE verdicts. Test-null recalibration is secondary and not independently
calibrated power. These limitations also apply to the exploratory scale diagnostic.

C-loc and C-scale fit centered class labels, whose conditional expectation is
eta(x)-pi, not Lambda(x). Their validation R-squared is not kappa(Lambda).
Consequently C-2 is not tested in its original likelihood-fraction interpretation.
The inner coefficient validation uses a first-70%/last-30% split within each TRAIN
segment, conditional on the full-TRAIN exponent and regime choices; it is not fully
nested selection validation.

AUC uncertainty uses 2000 paired bootstrap draws of nonoverlapping 63-calendar-day
clusters jointly across currencies. Synthetic class copies retain their shared date
and bootstrap multiplicity. These intervals are conditional diagnostics; stationarity,
block-length adequacy, training uncertainty and cross-split dependence are not certified.
Comparisons to the TEST-selected best C-scale arm are exploratory and not adjusted for
selection or multiplicity.

The corrected C-scale counts are TRAIN 19384 / TEST 9416, from original totals
29160 / 19443. C-loc fractional/polynomial AUC is 0.5397/0.5495 with paired gap
-0.0098 and block interval [-0.0123,-0.0075]. C-scale AUC spans 0.6908-0.6922;
the label-validation R-squared rank correlation is -0.200. The exploratory normalized
C-loc gap is +0.0014 with interval [-0.0001,+0.0028]. No general normalization rule
or real-data parameter-risk advantage is established. Exact recorded output is in
`python/expected_output_realdata.txt`.

---

# Study 2 — real noise in the second-moment-failure regime

**Written 2026-09-22, after the corrected FX study above was complete and BEFORE any Study 2
analysis code existed.** Motivation, data-selection rule, screen table, protocol and criteria are
fixed here; the run reports whatever comes out. This study does not replace the FX study; both are
reported.

## Why a second study

The FX study evaluated the framework where the training Hill estimate was 2.804: the fourth
moment is doubtful, but the sample Gram of the cubic dictionary is finite and, at 29 160 training
observations, stable enough (condition number 9.9e3) for the polynomial projection to act as an
ordinary estimator. The paper's completeness and efficiency results promise nothing in favour of a
fractional dictionary there; they say the fractional dictionary *remains admissible* where the
polynomial one loses its population meaning. Study 2 therefore targets the regime the theory
names: **a real noise law whose second moment fails within the block on which estimation is
performed.** The FX study also could not identify estimator risk, because a global training median
does not make the location of a market block known. Study 2 fixes that by construction (below).

## Data-selection rule (outcome-blind)

A candidate series is admissible iff, measured on its TRAIN part only:

1. **Tail.** Hill estimate on |x| (upper 10 %) *after within-block standardisation* `< 2` (second
   moment fails), and the per-block median Hill `< 4`. The pooled-before-standardisation Hill is
   not used: it confounds tail weight with scale mixing across blocks (the FX pooled estimate of
   2.80 becomes 2.46 within blocks of 125; the EMG pooled estimate of 1.22 becomes 1.43).
2. **Symmetry / known location.** Pair differences `d_t = x_{2t} - x_{2t-1}` inside a block have
   location 0 under within-pair exchangeability; the TRAIN sign balance `P(d > 0)` must lie in
   `[0.48, 0.52]`.
3. **Block independence.** Blocks are recording segments; the correlation between the last sample
   of one block and the first of the next must be below 0.2, and the within-block lag-1
   autocorrelation of `d` below 0.5.
4. **Not reported elsewhere by us** in a manuscript. A dataset used in another project of ours
   only as an unpublished applicability screen is admissible; one carrying a reported result of
   ours is not.

The screen was run over every series in our data lake; no AUC, RMSE or dictionary fit was
computed during screening. Full screen (2026-09-22):

| series | Hill pooled | Hill after within-block std. | per-block median Hill | P(x>0) | acf(1) | acf(1) of \|x\| | verdict |
|---|---|---|---|---|---|---|---|
| FX DEXJPUS log-returns (study above) | 2.82 | 2.46 (blocks of 125) | n/a | 0.494 | 0.02 | 0.19 | outside rule 1 |
| EEGdenoiseNet EMG epochs, raw | 1.22 | 2.22 (512) | 3.68 | 0.500 | 0.11 | 0.61 | rule 1 borderline |
| EEGdenoiseNet EMG epochs, pair differences | 1.06 | **1.43** (256) | 2.68 (IQR 1.8-3.2) | 0.500 | -0.13 | 0.60 | **admissible** |
| EEGdenoiseNet EOG epochs | 2.34 | - | - | 0.500 | 0.998 | 0.997 | fails rule 3 |
| EEGdenoiseNet EEG epochs | 3.60 | - | - | 0.500 | 0.93 | 0.86 | fails rules 1, 3 |
| MIT-BIH NSTDB `em` ch0, first difference | 2.43 | 2.28 (512) | 2.47 | 0.433 | 0.70 | 0.71 | fails rules 2, 3 |
| MIT-BIH NSTDB `ma` ch0, first difference | 2.14 | 2.33 (512) | 2.99 | 0.398 | 0.63 | 0.65 | fails rules 2, 3 |
| MIT-BIH NSTDB `bw`, both channels | 2.5-3.5 | - | - | 0.49 | 1.00 | 1.00 | fails rule 3 |
| Partial-discharge antenna, NonPD (noise) windows | 4.21 | - | - | 0.500 | 0.37 | 0.26 | fails rule 1 (light-tailed noise) |
| Partial-discharge antenna, PD windows | 1.34 | - | - | 0.500 | 0.73 | 0.82 | tail is the signal, wrong regime |
| XJTU-SY bearing 1_1, first / last minute | 4.66 / 3.09 | - | - | 0.500 | 0.40 / 0.82 | 0.15 / 0.70 | fails rule 1 |
| IPIX sea clutter, WTI crude oil | - | - | - | - | - | - | excluded by the prior commitment above |

Notes on the EMG rows. Epoch scale varies 600-fold and is autocorrelated across consecutive
epochs (0.94 at lag 1, 0.46 at lag 50): the epochs come from continuous recordings, and the
pooled Hill of 1.2 is largely a scale mixture. After per-epoch standardisation the tail is still
heavy (1.43 for pair differences; per-epoch median 2.7), so the sixth moment the cubic dictionary
needs fails in essentially every epoch and the variance in roughly a third. Boundary correlation
between consecutive epochs is 0.10. **Only one series passes: EMG pair differences.** It is the
study's data, chosen before any dictionary was fitted. Ownership: the EMG epochs were used in
another project of ours only as an unpublished applicability screen (rule 4).

## Data

- EEGdenoiseNet, file `EMG_all_epochs.npy`: 5 598 muscular-artefact epochs x 512 samples,
  256 Hz, licence CC0 1.0 (G-Node GIN `NCClab/EEGdenoiseNet`; Zhang et al., *J. Neural Eng.*
  18(5) 056057, 2021). The file is redistributed in `data/` with its SHA-256 in `data/SOURCE.md`.
- Block = epoch. Within each epoch `d_t = x_{2t} - x_{2t-1}`, `t = 1..256`.
- **Split, fixed here:** epochs in file order; first 60 % (3 358 epochs) = TRAIN, last 40 %
  (2 240 epochs) = TEST. Nothing in TEST informs exponents, coefficients, fitted laws or critical
  values. No re-splitting.
- **Scale.** Each block is divided by its own MAD (normal-consistent), computed on the block's
  `d` *before* any shift is injected. MAD about the block median is shift-invariant, so this uses
  no knowledge of the shift. Every arm sees the same standardised block. A single global scale is
  inapplicable when block scales differ by three orders of magnitude.

## R-0 — admissibility gate (abort condition)

Hill estimate `alpha_hat` on |d| (upper 10 %) of the pooled standardised TRAIN blocks. If
`alpha_hat >= 2`, the study is declared uninformative and reported as such. Exponent ceiling
`alpha_hat / 2`; fractional exponents `linspace(0.35 * ceiling, 0.92 * ceiling, 3)`, as in the FX
study.

## Estimand

A known shift `delta` (in block-MAD units) is added to every standardised `d` of a TEST block.
Primary `delta = 0.25`; secondary `delta = 1.0`, both reported. The true location of every block is
`delta` by construction: pairing, not centering, makes it known.

## Arms

sample mean | median | Huber M (k = 1.345) | Student-t MLE for location, with degrees of freedom
and scale fitted once on standardised TRAIN (the parametric robust competitor) | polynomial PMM
{x, x^3} (finite-sample stress arm; its population Gram does not exist) | Hermite-projected score,
order 4 (same status) | fractional PMM, odd subsystem, exponents from R-0.

Score coefficients for the three projection arms solve `F k = b` on standardised TRAIN with the
same kernel score proxy as the FX study (`fit_score_coefs`, `fit_hermite_coefs`). The PMM and
projection estimators are the root of the projected estimating equation on `[-3, 3]` MAD units
(grid of 61 + bisection, as in the FX study).

## Branch E — estimation

Block RMSE about the known `delta` over the 2 240 TEST blocks. Paired block bootstrap, blocks as
the resampling unit, 2 000 draws, on the difference of RMSE between arms.

- **E-1 (primary):** fractional RMSE < polynomial RMSE, 95 % paired CI excluding 0. PASS / FAIL.
- **E-2:** fractional RMSE <= 1.10 x the best of {median, Huber, t-MLE}. PASS / FAIL.
- **E-3 (diagnostic, not a criterion):** the ratio mean-RMSE / fractional-RMSE, and the split-half
  instability of the TRAIN Gram entry `mean(d^6)` (first half vs second half of TRAIN epochs).

## Branch T — testing

H0: `delta = 0`; H1: `delta = 0.25`. Critical values at size 0.05 from TRAIN blocks, applied
unchanged to TEST. Arms: t | sign | Wilcoxon | Hermite-projected score | fractional-projected
score.

- **T-1:** paired difference in TEST rejection rate under H1, fractional minus the best classical
  arm (t, sign, Wilcoxon), with paired SE over blocks; `z >= 2` win, `|z| < 2` tie, `z <= -2` loss.
  All three reportable. TEST sizes are reported beside the powers.

## Branch C — classification (shared-nuisance case)

Class 0 = standardised TEST `d`; class 1 = the same plus `delta = 0.25`. Centered-label least
squares on each dictionary, coefficients from TRAIN, with the FX study's caveat that the target
is `eta - pi`, not `Lambda`.

- **C-1:** held-out AUC of the fractional design >= polynomial design, 95 % paired
  epoch-bootstrap CI (an epoch's class-0 and class-1 rows resampled together) excluding 0.
  PASS / FAIL.

## Prediction versus realisation (the theorem's own check; reported as numbers)

On standardised TRAIN, a Student-t law is fitted (df, scale). Under that fitted law, the captured
Fisher fraction `kappa` of the fractional dictionary and of the sign dictionary
(`sgn(x)`, the a -> 0 limit) are computed by quadrature. Predicted efficiency ratio
fractional / median = `kappa_frac / kappa_sign`; realised ratio = TEST RMSE^2(median) /
RMSE^2(fractional) at `delta = 0.25`. Both numbers and their gap are reported. This is not a
PASS / FAIL criterion because the fitted law is a proxy for an unknown score and the blocks are
dependent.

## Reporting rule

R-5 above applies: every arm, every criterion, both deltas, PASS / FAIL as they come out, printed
to `python/expected_output_realdata_study2.txt`. If E-1 fails, the manuscript states that the
fractional dictionary did not beat the polynomial one even where the polynomial Gram has no
population meaning.

## Amendment S2-1 and Correction 6 — 2026-09-22, written BEFORE the Study 2 run on real data

The Study 2 pipeline was exercised first on **synthetic** epochs (t_{1.5} scale mixture, no real
data) as a known-answer control. The fractional PMM arm came out *worse* than the median
(RMSE 0.0759 vs 0.0704) although its captured fraction under the fitted law predicts the
opposite. The cause is the kernel score proxy shared with the FX study: scipy's default
bandwidth (Scott's factor times the sample standard deviation) is set by the largest
observations of a heavy-tailed sample and oversmooths the density, so the projection
coefficients were unrelated to the score (with the exact t score the same arm gives 0.0681).
**Fix:** Silverman's rule of thumb with the robust spread `min(sd, IQR/1.349)` in place of the
sample sd, the textbook default, implemented once in `kernel_score()` and used by every
projection arm (fractional, polynomial, Hermite) in both studies. On the synthetic control the
arm moves to 0.0702. No other element of the protocol changes.

Because the FX study used the same proxy, it is re-run with the corrected bandwidth
(**Correction 6** to that study) and its recorded output replaced; whatever moves is reported.
*Recorded after the re-run:* the fractional arm moved against itself everywhere the proxy
enters. Raw block RMS fractional 0.000540 -> 0.000702 (polynomial 0.000518 -> 0.000556; mean,
median, Huber unchanged); projected-score rejection rates fractional 0.0130/0.4870 ->
0.0260/0.4416 and Hermite 0.0325/0.5584 -> 0.0455/0.5065; test-null recalibrated fractional
0.6494 -> 0.6234; in the exploratory block-normalised diagnostic the fractional RMS
0.096993 -> 0.102838, so it is no longer the smallest of the five (Huber 0.100445 is). The
label-regression AUCs, which do not use the proxy, are unchanged. The manuscript's Table 7
and the surrounding text carry the re-run numbers.
The quadrature in the kappa prediction was also changed from a truncated to an infinite
integration range after the sign-dictionary check under Cauchy (8/pi^2) missed by 0.3 %.

## Post-run notes — 2026-09-22, after the single Study 2 run

- **Outcome, as recorded in `python/expected_output_realdata_study2.txt`:** R-0 passes (pooled
  standardised TRAIN Hill 1.362, exponents {0.238, 0.432, 0.626}); E-1 PASS (fractional RMSE
  0.0692 vs polynomial 0.1531, Hermite 0.1649, mean 0.1678; every paired interval excludes 0);
  E-2 FAIL (Huber 0.0581, fitted-t MLE 0.0645, median 0.0702; the fractional error is 19 % above
  the best robust arm against the 10 % bound); T-1 LOSS to Wilcoxon (power 0.985 vs 0.996,
  z = -4.5; the fractional test beats t, sign and Hermite); C-1 PASS (AUC +0.0007, interval
  [+0.0006, +0.0007]); kappa prediction 1.089 vs realised 1.029. Both deltas give identical
  errors, every arm being location-equivariant.
- **Per-block Hill in the screen table vs the run.** The screen row "2.68 (IQR 1.8-3.2)" was
  computed with a screening copy of the Hill estimator whose order-statistic floor is
  `k = max(50, 0.1 n)`, i.e. the upper 20 % of a 256-sample block; the package's `hill_alpha`
  uses `k = max(10, 0.1 n)`, the upper 10 %, and under it the per-epoch median is 3.46 over all
  epochs and 3.56 on TRAIN. Pooled standardised values, where the floor is irrelevant, agree
  (1.43 vs 1.428). The screen row is left as recorded; rule 1's second clause (< 4) holds under
  either definition, and the manuscript quotes the package's.
- **Test epochs are heavier-tailed than training epochs:** pooled standardised Hill 1.539 vs
  1.362, per-epoch median 3.29 vs 3.56, kurtosis 286 vs 172. Nothing fitted on training data
  sees this; it is one reason the kappa prediction and the fitted-t arm are proxies. The
  diagnostic line was added to the script after the run and the script re-run; every other
  printed number is unchanged.
- **Why E-2 failed, read against the fitted law (added 2026-09-22 after the run; no data
  touched).** `python/reference_efficiencies.py` (recorded output
  `expected_output_reference_efficiencies.txt`) gives the closed-form efficiencies of the
  median, Huber(1.345, MAD scale) and Wilcoxon arms against the t_nu MLE. Under the fitted
  nu = 1.855 they are 0.835 / 0.834 / 0.846, below the fractional kappa 0.910, so the fitted law
  predicted the fractional arm ahead of Huber by about 9 %. At the per-epoch tail (nu 3.3-3.56,
  bracketing the per-epoch median Hill estimates) Huber and Wilcoxon are at 0.958-0.971. The
  realised order (Huber 16 % better) is therefore the signature of a shape mixture: the kernel
  score is fitted to the pooled law, which no single epoch follows, while the robust arms are
  near-efficient on the within-epoch law. The pair-difference design is symmetric by
  construction, so this study could at best tie the robust class; the manuscript's §8.2 says
  so and the Limitations record that neither study exercises the asymmetric laws the theorem
  admits.

## What this can and cannot show

It can show that, on a real noise law in the infinite-variance regime, the moment-based branch
collapses while the fractional dictionary keeps all three readings usable at an efficiency near
a fitted parametric robust estimator. It cannot show a universal advantage of fractional
dictionaries and does not alter the FX conclusions. Within-block dependence (lag-1 -0.13; burst
clustering of |d| 0.60) is real and left in the data; the `kappa` prediction assumes independence,
which is why the prediction-realisation gap is reported rather than tested.
