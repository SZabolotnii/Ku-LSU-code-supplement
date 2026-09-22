"""
run_realdata.py -- the real-data study for the Statistical Papers revision (Reviewer 1).

Design, criteria and abort conditions were fixed in review/REAL_DATA_SPEC_2026-09-16.md
before initial analysis. Corrections made after that analysis are documented in
REAL_DATA_SPEC.md; they are not pre-registered decisions.

Data: FRED DEXJPUS (yen per US dollar, daily, from 1971), from the dataset lake.
Split 60/40 by date: TRAIN fixes exponents, coefficients and critical values; TEST
carries evaluation metrics; training diagnostics are reported separately.

The study contains one favourable sub-case (heavy tail as a SHARED nuisance) and one
adverse sub-case (heavy tail IS the class contrast) with a pre-registered prediction
AGAINST the captured fraction, because the portfolio has twice measured it failing in
exactly that regime.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import optimize, stats

from run_comparators import (
    _design_frac,
    _design_poly,
    _hermite_cols,
    _psi_root,
    hill_alpha,
    huber_location,
    _auc,
)
from run_three_branch_unification import SEED, frac_basis

NBLOCK = 125        # Amendment 1: was 250, too few blocks to resolve power
NBOOT = 2000
ALPHA = 0.05
TRAIN_FRAC = 0.60
VOL_WINDOW = 21


SERIES = ["DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXUSEU"]   # Amendment 1: four FX series.
                                                        # SP500 excluded (other asset class,
                                                        # 2611 rows); DCOILWTICO excluded by
                                                        # prior portfolio commitment.


def resolve_data():
    """The four CSVs, looked for in the package's own data/ first.

    The files are public-domain FRED series and ship with this package, so a reader needs
    no download and no account.  RESEARCH_DATA_ROOT redirects to a local dataset lake
    (<root>/store/fred-heavytail/data), which is where they were originally read from.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(os.path.dirname(here), "data")]
    root = os.environ.get("RESEARCH_DATA_ROOT")
    if root:
        candidates.append(os.path.join(root, "store", "fred-heavytail", "data"))
    paths = []
    for nm in SERIES:
        for d in candidates:
            p = os.path.join(d, nm + ".csv")
            if os.path.exists(p):
                paths.append((nm, p))
                break
        else:
            sys.exit(f"dataset not found: {nm}.csv in " + " or ".join(candidates))
    return paths


def load_returns(path):
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")
    px = np.asarray(raw[raw.dtype.names[1]], dtype=float)
    ok = np.isfinite(px) & (px > 0)
    dates = np.asarray(raw[raw.dtype.names[0]], dtype='datetime64[D]')[ok]
    px = px[ok]
    r = np.diff(np.log(px))
    return dates[1:][np.isfinite(r)], r[np.isfinite(r)]


def prepare_series(name, dates, returns):
    """Freeze the original split before any centering, rolling window or filtering."""
    train = np.arange(len(returns)) < int(TRAIN_FRAC * len(returns))
    center = float(np.median(returns[train]))
    r = np.asarray(returns) - center
    rv = np.full(len(r), np.nan)
    for i in range(VOL_WINDOW, len(r)):
        rv[i] = np.std(r[i - VOL_WINDOW:i])
    fit = np.arange(len(r)) < int(0.7 * np.sum(train))
    return dict(name=name, dates=dates, r=r, train=train, fit=fit, rv=rv, center=center)


def scale_samples(series):
    """Apply TRAIN terciles without changing original series-wise split membership."""
    rv_train = np.concatenate([s['rv'][s['train'] & np.isfinite(s['rv'])] for s in series])
    thresholds = np.quantile(rv_train, [1 / 3, 2 / 3])
    parts = {k: [] for k in ('xtr', 'ytr', 'xte', 'yte', 'dates_tr', 'dates_te', 'fit')}
    for s in series:
        lab = np.where(s['rv'] >= thresholds[1], 1, np.where(s['rv'] <= thresholds[0], 0, -1))
        for suffix, mask in [('tr', s['train']), ('te', ~s['train'])]:
            keep = mask & (lab >= 0)
            parts['x' + suffix].append(s['r'][keep])
            parts['y' + suffix].append(lab[keep].astype(float))
            parts['dates_' + suffix].append(s['dates'][keep])
            if suffix == 'tr':
                parts['fit'].append(s['fit'][keep])
    return {**{k: np.concatenate(v) for k, v in parts.items()}, 'thresholds': thresholds}


def calendar_boot_indices(dates, rng, block_days=63):
    """Resample nonoverlapping 63-calendar-day clusters, jointly across currencies.

    Synthetic copies retain the same date and hence the same multiplicity. These are
    conditional, exploratory intervals; stationarity and block-length adequacy are not
    established by this exercise.
    """
    groups = (dates.astype('datetime64[D]').astype(np.int64) // block_days)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    cuts = np.r_[0, np.cumsum(np.bincount(inv))]
    draw = rng.integers(0, len(cuts) - 1, len(cuts) - 1)
    return np.concatenate([order[cuts[j]:cuts[j + 1]] for j in draw])


def blocks_of(x, n=NBLOCK):
    k = len(x) // n
    return x[: k * n].reshape(k, n)


def fmt_exponents(a):
    """Exponent set as plain numbers.  numpy scalars repr as np.float64(0.491) under
    numpy >= 2, which is noise in output the supplement prints verbatim."""
    return "[" + ", ".join(f"{float(x):.3f}" for x in a) + "]"


def branch_estimation(tr, Bte, alphas, delta, rng):
    """Descriptive RMS about the TRAIN median, not risk about a known block location."""
    sc = robust_scale(tr)
    Phi_f, _ = frac_basis(alphas)
    Phi_p, _ = frac_basis([1.0, 3.0])
    k_f, cond_f = fit_score_coefs(tr, alphas, sc)
    k_p, cond_p = fit_score_coefs(tr, [1.0, 3.0], sc)
    P_raw = Phi_f(tr)
    cond_raw = float(np.linalg.cond((P_raw.T @ P_raw) / len(tr)))
    arms = {
        "sample mean": lambda b: float(b.mean()),
        "median": lambda b: float(np.median(b)),
        "Huber(1.345)": lambda b: huber_location(b),
        "polynomial PMM {x,x^3}":
            lambda b: sc * _psi_root(b / sc, Phi_p, k_p, lo=-4*delta/sc, hi=4*delta/sc),
        "fractional PMM":
            lambda b: sc * _psi_root(b / sc, Phi_f, k_f, lo=-4*delta/sc, hi=4*delta/sc),
    }
    estimates = {name: np.array([f(b) for b in Bte]) for name, f in arms.items()}
    rms = {name: float(np.sqrt(np.mean(v**2))) for name, v in estimates.items()}
    return dict(rms=rms, best=min(rms.values()), nblocks=len(Bte),
                cond_frac=cond_f, cond_poly=cond_p, cond_frac_raw=cond_raw, scale=sc)


def robust_scale(x):
    """TRAIN-estimated robust scale used for every arm's numerical conditioning.

    Scaling x rescales power columns differently. It preserves their unregularized
    span, but can greatly change the empirical Gram's condition number. Regularized
    numerical fits need not be exactly invariant to such a column rescaling.
    """
    return float(stats.median_abs_deviation(x, scale="normal"))


def kernel_score(z, n_sub=4000):
    """Nonparametric score proxy psi_hat = -(d/dz) log f_hat from a Gaussian kernel fit.

    Bandwidth: Silverman's rule of thumb with the ROBUST spread min(sd, IQR/1.349) in place of
    the sample sd (Correction 6 in REAL_DATA_SPEC.md).  scipy's default Scott factor multiplies
    the sample sd, which on a heavy-tailed sample is set by the largest observations; on a
    synthetic t_1.5 control the resulting proxy gave projection coefficients unrelated to the
    true score (RMSE 0.0759 against 0.0681 with the exact score; the robust rule gives 0.0702).
    """
    zs = z[:: max(1, len(z) // n_sub)]
    spread = min(float(np.std(zs)), float(stats.iqr(zs) / 1.349))
    factor = 0.9 * spread * len(zs) ** (-1 / 5) / max(float(np.std(zs)), 1e-12)
    kde = stats.gaussian_kde(zs, bw_method=factor)
    h = 1e-3 * spread
    return -(np.log(kde(z + h) + 1e-300) - np.log(kde(z - h) + 1e-300)) / (2 * h)


def fit_score_coefs(x, alphas, scale=None):
    """Coefficients k solving F k = b on TRAIN, with b by the Stein form E[phi * psi] and
    psi the empirical score proxy.  We do NOT have a parametric score on real data, so we
    use the nonparametric estimate psi_hat = -(d/dx) log f_hat from a kernel fit on TRAIN;
    this is the honest plug-in a practitioner has, and it is fitted on TRAIN only."""
    sc = robust_scale(x) if scale is None else scale
    z = x / sc
    Phi, _ = frac_basis(alphas)
    psi = kernel_score(z)
    P = Phi(z)
    F = (P.T @ P) / len(z)
    b = (P.T @ psi) / len(z)
    k = np.linalg.solve(F + 1e-12 * np.eye(len(alphas)), b)
    return k, float(np.linalg.cond(F))


# --------------------------------------------------------------------------- #
def branch_testing(tr, Btr, Bte, alphas, delta, rng):
    sc = robust_scale(tr)
    Phi_f, _ = frac_basis(alphas)
    k_f, cond_f = fit_score_coefs(tr, alphas, sc)
    k_h, cond_h = fit_hermite_coefs(tr, scale=sc)

    def stat(b):
        return {
            "t-test": float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))),
            "sign": float(np.mean(np.sign(b))),
            "Wilcoxon": float(stats.wilcoxon(b, alternative="greater",
                                             zero_method="zsplit").statistic),
            "projected score (Hermite)": float((_hermite_cols(b / sc) @ k_h).mean()),
            "projected score (fractional)": float((Phi_f(b / sc) @ k_f).mean()),
        }

    names = list(stat(Bte[0]).keys())
    s_tr = {nm: [] for nm in names}
    for b in Btr:                                   # critical values from TRAIN only
        v = stat(b)
        for nm in names:
            s_tr[nm].append(v[nm])
    crit = {nm: float(np.quantile(s_tr[nm], 1 - ALPHA)) for nm in names}

    rej0 = {nm: [] for nm in names}
    rej1 = {nm: [] for nm in names}
    for b in Bte:
        v0, v1 = stat(b), stat(b + delta)
        for nm in names:
            rej0[nm].append(float(v0[nm] > crit[nm]))
            rej1[nm].append(float(v1[nm] > crit[nm]))
    size = {nm: float(np.mean(rej0[nm])) for nm in names}
    power = {nm: float(np.mean(rej1[nm])) for nm in names}

    # SECONDARY (Amendment 2): recalibrate on TEST null blocks, so every arm has size 0.05
    # on the data the power is measured on.  Reported beside the primary, never instead.
    s_te = {nm: [] for nm in names}
    for b in Bte:
        v = stat(b)
        for nm in names:
            s_te[nm].append(v[nm])
    crit2 = {nm: float(np.quantile(s_te[nm], 1 - ALPHA)) for nm in names}
    rej1b = {nm: [] for nm in names}
    for b in Bte:
        v1 = stat(b + delta)
        for nm in names:
            rej1b[nm].append(float(v1[nm] > crit2[nm]))
    power2 = {nm: float(np.mean(rej1b[nm])) for nm in names}
    return dict(size=size, power=power, power2=power2, nblocks=len(Bte))


def fit_hermite_coefs(x, order=4, scale=None):
    sc = robust_scale(x) if scale is None else scale
    z = x / sc
    psi = kernel_score(z)
    H = _hermite_cols(z, order)
    G = (H.T @ H) / len(z)
    b = (H.T @ psi) / len(z)
    k = np.linalg.solve(G + 1e-12 * np.eye(order), b)
    return k, float(np.linalg.cond(G))


# --------------------------------------------------------------------------- #
def branch_cls_location(tr, te, alphas, delta, rng, dates_te):
    """C-loc: the heavy tail is a SHARED nuisance; only location differs."""
    x0tr, x1tr = tr, tr + delta
    x0te, x1te = te, te + delta
    xtr = np.concatenate([x0tr, x1tr])
    ytr = np.concatenate([np.zeros(len(x0tr)), np.ones(len(x1tr))])
    xte = np.concatenate([x0te, x1te])
    yte = np.concatenate([np.zeros(len(x0te)), np.ones(len(x1te))])

    sc = robust_scale(tr)
    out = {}
    for nm, (Dtr, Dte) in {
        "label regression (fractional)":
            (_design_frac(xtr / sc, alphas), _design_frac(xte / sc, alphas)),
        "label regression (polynomial)":
            (_design_poly(xtr / sc, 3), _design_poly(xte / sc, 3)),
    }.items():
        coef, *_ = np.linalg.lstsq(Dtr, ytr - 0.5, rcond=None)
        out[nm] = Dte @ coef
    auc = {nm: _auc(s, yte) for nm, s in out.items()}

    # Paired calendar-block bootstrap keeps currencies and synthetic copies together.
    n = len(yte)
    dates = np.concatenate([dates_te, dates_te])
    diffs = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = calendar_boot_indices(dates, rng)
        diffs[i] = (_auc(out["label regression (fractional)"][idx], yte[idx])
                    - _auc(out["label regression (polynomial)"][idx], yte[idx]))
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    gap = auc["label regression (fractional)"] - auc["label regression (polynomial)"]
    c1 = (gap >= 0) and (lo > 0)
    return dict(auc=auc, diff=gap, ci=(lo, hi), c1=c1, n_test=n)


def branch_cls_scale(series, alphas, rng):
    """C-scale: the heavy tail IS the contrast.  Adverse pre-registered prediction C-2."""
    samples = scale_samples(series)
    xtr, ytr, xte, yte = [samples[k] for k in ('xtr', 'ytr', 'xte', 'yte')]

    ceil = max(alphas)
    dicts = {
        "A={1}": [1.0],
        "A={1,3}": [1.0, 3.0],
        "A={low}": [0.4 * ceil],
        "A={mid}": [0.7 * ceil],
        "A={high}": [ceil],
        "frac triple": list(alphas),
    }
    sc = float(stats.median_abs_deviation(xtr, scale="normal"))

    # Validation R-squared predicts centered labels, not an unknown log-ratio.
    # Dictionary exponents and regime labels use all TRAIN, so this inner holdout
    # is conditional on those choices, not a nested validation of the entire selector.
    fit = samples['fit']
    rows, scores = {}, {}
    for nm, A in dicts.items():
        Dfit = _design_frac(xtr[fit] / sc, A)
        Dhold = _design_frac(xtr[~fit] / sc, A)
        t_fit = ytr[fit] - ytr[fit].mean()
        t_hold = ytr[~fit] - ytr[fit].mean()
        coef, *_ = np.linalg.lstsq(Dfit, t_fit, rcond=None)
        kap_in = float(np.sum((Dfit @ coef) ** 2) / np.sum(t_fit ** 2))
        resid = t_hold - Dhold @ coef
        kap_out = float(1.0 - np.sum(resid ** 2) / np.sum((ytr[~fit] - ytr[~fit].mean()) ** 2))
        # test scores use coefficients refitted on the whole of TRAIN
        coef_full, *_ = np.linalg.lstsq(_design_frac(xtr / sc, A), ytr - ytr.mean(), rcond=None)
        sco = _design_frac(xte / sc, A) @ coef_full
        scores[nm] = sco
        rows[nm] = dict(r2_in=kap_in, r2_validation=kap_out, auc_test=_auc(sco, yte),
                        ncol=Dfit.shape[1])

    names = list(dicts)
    ki = np.array([rows[n]["r2_in"] for n in names])
    ko = np.array([rows[n]["r2_validation"] for n in names])
    au = np.array([rows[n]["auc_test"] for n in names])
    rho_in = float(stats.spearmanr(ki, au).statistic)
    rho = float(stats.spearmanr(ko, au).statistic)

    # Is there a ranking to recover at all?  Paired bootstrap of the AUC gap to the best arm.
    best = max(names, key=lambda k: rows[k]["auc_test"])
    n = len(yte)
    ndist = 0
    for nm in names:
        if nm == best:
            rows[nm]["ci"] = None
            continue
        d = np.empty(NBOOT)
        for i in range(NBOOT):
            idx = calendar_boot_indices(samples['dates_te'], rng)
            d[i] = _auc(scores[best][idx], yte[idx]) - _auc(scores[nm][idx], yte[idx])
        lo2, hi2 = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        rows[nm]["ci"] = (lo2, hi2)
        ndist += int(lo2 > 0)
    spread = float(au.max() - au.min())
    c2 = rho <= 0.0
    return dict(rows=rows, spearman=rho, spearman_in=rho_in, c2=c2, best=best,
                n_distinguishable=ndist, n_arms=len(names), spread=spread,
                n_train=len(xtr), n_test=len(xte), frac_class1=float(yte.mean()))


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    series, train_blocks, test_blocks, test_block_dates = [], [], [], []
    for name, path in resolve_data():
        dates, returns = load_returns(path)
        s = prepare_series(name, dates, returns)
        series.append(s)
        train_blocks.append(blocks_of(s["r"][s["train"]]))
        test = s["r"][~s["train"]]
        test_blocks.append(blocks_of(test))
        test_block_dates.append(dates[~s["train"]][:len(blocks_of(test))*NBLOCK])
    tr = np.concatenate([s["r"][s["train"]] for s in series])
    te = np.concatenate([s["r"][~s["train"]] for s in series])
    dates_te = np.concatenate([s["dates"][~s["train"]] for s in series])
    Btr, Bte = np.concatenate(train_blocks), np.concatenate(test_blocks)
    print("=" * 78)
    print("REAL-DATA STUDY -- corrected protocol 2026-09-21; seed =", SEED)
    print("Specification: REAL_DATA_SPEC.md (post-analysis correction recorded)")
    for s in series:
        print(f"{s['name']}: {len(s['r'])} returns; TRAIN {s['train'].sum()} / TEST {(~s['train']).sum()}")
    print(f"TOTAL {len(tr)+len(te)}; TRAIN {len(tr)} / TEST {len(te)}; blocks {len(Btr)} / {len(Bte)}")
    print("Per-series chronological split; not a global forward-forecasting split.")
    print("Cross-currency TRAIN/TEST calendar overlap remains; results are descriptive.")
    print("AUC intervals: paired 63-calendar-day clusters, 2000 replicates; conditional diagnostics.")
    ah = hill_alpha(tr)
    ceil = ah / 2
    alphas = list(np.linspace(0.35*ceil, 0.92*ceil, 3))
    print(f"R-0: TRAIN Hill {ah:.3f}; heuristic ceiling {ceil:.3f}; A={fmt_exponents(alphas)}")
    if ah >= 4:
        print("Study uninformative under the specified Hill gate.")
        return
    delta = 0.25 * robust_scale(tr)
    print(f"Shift delta={delta:.6f}; a Hill estimate does not prove moment existence.")

    def print_estimation(e):
        for name, value in e["rms"].items():
            print(f"  {name:<28} RMS {value:.6f}; ratio to minimum {value/e['best']:.3f}")
        print(f"  cond(F): fractional {e['cond_frac']:.3g}, polynomial {e['cond_poly']:.3g}, raw fractional {e['cond_frac_raw']:.3g}")
        print("  E-1/E-2: NOT IDENTIFIED as parameter-risk claims; descriptive RMS only.")

    print("\n[E] descriptive block estimates relative to TRAIN centering")
    print_estimation(branch_estimation(tr, Bte, alphas, delta, rng))
    print("\n[T] descriptive rejection rates; no iid z-test or WIN/TIE inference")
    t = branch_testing(tr, Btr, Bte, alphas, delta, rng)
    for name in t["power"]:
        print(f"  {name:<32} baseline {t['size'][name]:.4f}; shifted {t['power'][name]:.4f}; TEST-recalibrated {t['power2'][name]:.4f}")
    print("  T-1: no inferential verdict; TEST-recalibration is secondary and not held-out calibration.")

    def print_location(result):
        for name, value in result["auc"].items():
            print(f"  {name:<34} AUC {value:.4f}")
        print(f"  paired AUC gap {result['diff']:+.4f}; block CI [{result['ci'][0]:+.4f}, {result['ci'][1]:+.4f}]")
        print(f"  C-1 descriptive superiority criterion: {'met' if result['c1'] else 'not met'}")

    print("\n[C-loc] centered-label regression (not log-ratio projection)")
    print_location(branch_cls_location(tr, te, alphas, delta, rng, dates_te))
    print("\n[C-scale] within-series trailing windows; immutable split after filtering")
    print("  Each exponent set A generates an intercept and BOTH signed/absolute powers.")
    cs = branch_cls_scale(series, alphas, rng)
    print(f"  TRAIN {cs['n_train']} / TEST {cs['n_test']}; TEST class-1 share {cs['frac_class1']:.3f}")
    for name, row in cs["rows"].items():
        ci = "(best arm)" if row["ci"] is None else f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]"
        print(f"  {name:<14} cols {row['ncol']}; R2 fit {row['r2_in']:.4f}; R2 validation {row['r2_validation']:.4f}; AUC {row['auc_test']:.4f}; gap-to-best CI {ci}")
    print(f"  Spearman(validation label R2,AUC) {cs['spearman']:+.3f}; in-sample {cs['spearman_in']:+.3f}")
    print(f"  AUC spread {cs['spread']:.4f}; {cs['n_distinguishable']}/{cs['n_arms']-1} unadjusted intervals above zero.")
    print("  C-2: NOT TESTED as a claim about kappa(Lambda); label R2 is a different target.")
    print("  Best arm is selected on TEST; intervals are exploratory, not selection-adjusted.")

    print("\n[A3] exploratory block-scale diagnostic; reuses the observed TEST period")
    Bd = Bte / np.maximum(stats.median_abs_deviation(Bte, axis=1, scale="normal", keepdims=True), 1e-12)
    trd = np.concatenate([b/max(float(stats.median_abs_deviation(b, scale="normal")), 1e-12) for b in Btr])
    ahd = hill_alpha(trd)
    Ad = list(np.linspace(0.35*ahd/2, 0.92*ahd/2, 3))
    deld = 0.25*robust_scale(trd)
    print(f"  TRAIN Hill {ahd:.3f}; A={fmt_exponents(Ad)}")
    print_estimation(branch_estimation(trd, Bd, Ad, deld, rng))
    print_location(branch_cls_location(trd, Bd.ravel(), Ad, deld, rng, np.concatenate(test_block_dates)))
    print("  No parameter-risk advantage or general devolatilization recommendation established.")
    print("=" * 78)


if __name__ == "__main__":
    main()
