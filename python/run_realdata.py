"""
run_realdata.py -- the real-data study for the Statistical Papers revision (Reviewer 1).

Design, criteria and abort conditions were fixed in review/REAL_DATA_SPEC_2026-09-16.md
BEFORE this file was written.  Nothing here may move a threshold; every arm reports
whatever it produced.

Data: FRED DEXJPUS (yen per US dollar, daily, from 1971), from the dataset lake.
Split 60/40 by date: TRAIN fixes exponents, coefficients and critical values; TEST
carries every reported metric.

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
    px = px[ok]
    r = np.diff(np.log(px))
    return r[np.isfinite(r)]


def blocks_of(x, n=NBLOCK):
    k = len(x) // n
    return x[: k * n].reshape(k, n)


def fmt_exponents(a):
    """Exponent set as plain numbers.  numpy scalars repr as np.float64(0.491) under
    numpy >= 2, which is noise in output the supplement prints verbatim."""
    return "[" + ", ".join(f"{float(x):.3f}" for x in a) + "]"


def paired_boot(d, rng, nboot=NBOOT):
    """Bootstrap CI for the mean of a paired difference vector."""
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# --------------------------------------------------------------------------- #
def branch_estimation(tr, Bte, alphas, delta, rng):
    """Amendment 2.  Each series' TEST segment was centred once by its OWN grand median
    (one constant from ~5000 observations), so the true location of every block is 0 up to
    O(1/sqrt(5000)) -- negligible against a block's own O(1/sqrt(125)).  Each estimator is
    applied to each block and its error is the estimate itself.  No arm is privileged: the
    centring constant uses 40x more data than any block, so a block median is not exact.

    (Amendment 1's paired difference est(b+delta)-est(b) is withdrawn: it equals delta
    identically for any location-equivariant estimator and measures nothing.)"""
    sc = robust_scale(tr)
    Phi_f, _ = frac_basis(alphas)
    Phi_p, _ = frac_basis([1.0, 3.0])
    k_f, cond_f = fit_score_coefs(tr, alphas, sc)
    k_p, cond_p = fit_score_coefs(tr, [1.0, 3.0], sc)

    # cond(F) of the same fractional design applied to RAW returns, i.e. without the TRAIN
    # robust scale.  The Gram depends on the design alone, so this is the unscaled twin of
    # cond_f.  Printed so that the numerical-hygiene claim in `robust_scale`, which the
    # manuscript quotes, is reproduced rather than asserted.
    P_raw = Phi_f(tr)
    cond_f_raw = float(np.linalg.cond((P_raw.T @ P_raw) / len(tr)))

    arms = {
        "sample mean": lambda b: float(b.mean()),
        "median": lambda b: float(np.median(b)),
        "Huber(1.345)": lambda b: huber_location(b),
        "polynomial PMM {x,x^3}":
            lambda b: sc * _psi_root(b / sc, Phi_p, k_p, lo=-4 * delta / sc, hi=4 * delta / sc),
        "fractional PMM":
            lambda b: sc * _psi_root(b / sc, Phi_f, k_f, lo=-4 * delta / sc, hi=4 * delta / sc),
    }
    err = {nm: np.array([f(b) for b in Bte]) for nm, f in arms.items()}
    rmse = {nm: float(np.sqrt(np.mean(e ** 2))) for nm, e in err.items()}

    d = err["fractional PMM"] ** 2 - err["polynomial PMM {x,x^3}"] ** 2
    m, lo, hi = paired_boot(d, rng)
    e1 = (rmse["fractional PMM"] <= rmse["polynomial PMM {x,x^3}"]) and (hi < 0)
    best = min(rmse.values())
    e2 = rmse["fractional PMM"] <= 1.05 * best
    return dict(rmse=rmse, nblocks=len(Bte), delta=delta,
                e1=e1, e1_stat=(m, lo, hi), e2=e2, best=best,
                cond_frac=cond_f, cond_poly=cond_p, cond_frac_raw=cond_f_raw, scale=sc)


def robust_scale(x):
    """TRAIN-estimated robust scale.  The fractional basis is NOT scale-equivariant across
    different exponents: on raw FX returns (order 1e-3) the columns |x|^0.49 and |x|^1.29
    differ by ~300x in magnitude and cond(F) = 1.0e5, against 1.1e3 after standardizing.
    Every basis in this file is therefore applied to x / scale, with `scale` fixed on TRAIN.
    This is numerical hygiene applied identically to every arm; it changes no criterion."""
    return float(stats.median_abs_deviation(x, scale="normal"))


def fit_score_coefs(x, alphas, scale=None):
    """Coefficients k solving F k = b on TRAIN, with b by the Stein form E[phi * psi] and
    psi the empirical score proxy.  We do NOT have a parametric score on real data, so we
    use the nonparametric estimate psi_hat = -(d/dx) log f_hat from a kernel fit on TRAIN;
    this is the honest plug-in a practitioner has, and it is fitted on TRAIN only."""
    sc = robust_scale(x) if scale is None else scale
    z = x / sc
    Phi, _ = frac_basis(alphas)
    kde = stats.gaussian_kde(z[:: max(1, len(z) // 4000)])
    h = 1e-3 * np.std(z)
    psi = -(np.log(kde(z + h) + 1e-300) - np.log(kde(z - h) + 1e-300)) / (2 * h)
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

    ours = "projected score (fractional)"
    r_ours = np.array(rej1[ours])
    paired = {}
    for nm in names:
        if nm == ours:
            continue
        d = r_ours - np.array(rej1[nm])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        paired[nm] = (float(d.mean()), se, float(d.mean() / se) if se > 0 else 0.0)

    classical = [nm for nm in names if nm not in (ours, "projected score (Hermite)")]
    best_cl = max(classical, key=lambda nm: power[nm])
    z = paired[best_cl][2]
    verdict = "WIN" if z >= 2 else ("LOSS" if z <= -2 else "TIE")

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
    r2_ours = np.array(rej1b[ours])
    paired2 = {}
    for nm in names:
        if nm == ours:
            continue
        d = r2_ours - np.array(rej1b[nm])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        paired2[nm] = (float(d.mean()), se, float(d.mean() / se) if se > 0 else 0.0)
    best_cl2 = max(classical, key=lambda nm: power2[nm])
    z2 = paired2[best_cl2][2]
    verdict2 = "WIN" if z2 >= 2 else ("LOSS" if z2 <= -2 else "TIE")

    return dict(size=size, power=power, paired=paired, nblocks=len(Bte),
                best_classical=best_cl, z=z, verdict=verdict,
                power2=power2, paired2=paired2, best_classical2=best_cl2,
                z2=z2, verdict2=verdict2)


def fit_hermite_coefs(x, order=4, scale=None):
    sc = robust_scale(x) if scale is None else scale
    z = x / sc
    kde = stats.gaussian_kde(z[:: max(1, len(z) // 4000)])
    h = 1e-3 * np.std(z)
    psi = -(np.log(kde(z + h) + 1e-300) - np.log(kde(z - h) + 1e-300)) / (2 * h)
    H = _hermite_cols(z, order)
    G = (H.T @ H) / len(z)
    b = (H.T @ psi) / len(z)
    k = np.linalg.solve(G + 1e-12 * np.eye(order), b)
    return k, float(np.linalg.cond(G))


# --------------------------------------------------------------------------- #
def branch_cls_location(tr, te, alphas, delta, rng):
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
        "projected Lambda (fractional)":
            (_design_frac(xtr / sc, alphas), _design_frac(xte / sc, alphas)),
        "projected Lambda (polynomial)":
            (_design_poly(xtr / sc, 3), _design_poly(xte / sc, 3)),
    }.items():
        coef, *_ = np.linalg.lstsq(Dtr, ytr - 0.5, rcond=None)
        out[nm] = Dte @ coef
    auc = {nm: _auc(s, yte) for nm, s in out.items()}

    # paired bootstrap over TEST observations
    n = len(yte)
    diffs = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = (_auc(out["projected Lambda (fractional)"][idx], yte[idx])
                    - _auc(out["projected Lambda (polynomial)"][idx], yte[idx]))
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    c1 = (auc["projected Lambda (fractional)"] >= auc["projected Lambda (polynomial)"]) and (lo > 0)
    return dict(auc=auc, diff=float(diffs.mean()), ci=(lo, hi), c1=c1, n_test=n)


def branch_cls_scale(r, alphas, rng, train_frac=TRAIN_FRAC):
    """C-scale: the heavy tail IS the contrast.  Adverse pre-registered prediction C-2."""
    rv = np.array([np.std(r[max(0, i - VOL_WINDOW):i]) if i >= VOL_WINDOW else np.nan
                   for i in range(len(r))])
    ok = np.isfinite(rv)
    r, rv = r[ok], rv[ok]
    ntr = int(train_frac * len(r))
    lo_q, hi_q = np.quantile(rv[:ntr], [1 / 3, 2 / 3])
    lab = np.where(rv >= hi_q, 1, np.where(rv <= lo_q, 0, -1))
    keep = lab >= 0
    r, lab = r[keep], lab[keep]
    ntr = int(train_frac * len(r))
    xtr, ytr = r[:ntr], lab[:ntr].astype(float)
    xte, yte = r[ntr:], lab[ntr:].astype(float)

    ceil = max(alphas)
    dicts = {
        "{x}": [1.0],
        "{x, x^3}": [1.0, 3.0],
        "|x|^a low": [0.4 * ceil],
        "|x|^a mid": [0.7 * ceil],
        "|x|^a high": [ceil],
        "frac triple": list(alphas),
    }
    sc = float(stats.median_abs_deviation(xtr, scale="normal"))

    # CORRECTION 2026-09-16.  The first version of this branch scored each dictionary by an
    # IN-SAMPLE R^2 on the training data.  That is not the selector the paper proposes
    # (check S1 of the controlled study uses a HELD-OUT captured fraction), and it rewards
    # whichever dictionary has the most columns, so it cannot test a selection rule.  The
    # captured fraction is now computed out of sample: coefficients are fitted on the first
    # 70% of TRAIN and kappa is evaluated on the remaining 30%, which no fit has seen.
    # Both values are reported so the difference is visible.
    nfit = int(0.7 * len(xtr))
    rows, scores = {}, {}
    for nm, A in dicts.items():
        Dfit = _design_frac(xtr[:nfit] / sc, A)
        Dhold = _design_frac(xtr[nfit:] / sc, A)
        t_fit = ytr[:nfit] - ytr[:nfit].mean()
        t_hold = ytr[nfit:] - ytr[nfit:].mean()
        coef, *_ = np.linalg.lstsq(Dfit, t_fit, rcond=None)
        kap_in = float(np.sum((Dfit @ coef) ** 2) / np.sum(t_fit ** 2))
        resid = t_hold - Dhold @ coef
        kap_out = float(1.0 - np.sum(resid ** 2) / np.sum(t_hold ** 2))
        # test scores use coefficients refitted on the whole of TRAIN
        coef_full, *_ = np.linalg.lstsq(_design_frac(xtr / sc, A), ytr - ytr.mean(), rcond=None)
        sco = _design_frac(xte / sc, A) @ coef_full
        scores[nm] = sco
        rows[nm] = dict(kappa_in=kap_in, kappa_out=kap_out, auc_test=_auc(sco, yte),
                        ncol=Dfit.shape[1])

    names = list(dicts)
    ki = np.array([rows[n]["kappa_in"] for n in names])
    ko = np.array([rows[n]["kappa_out"] for n in names])
    au = np.array([rows[n]["auc_test"] for n in names])
    rho_in = float(stats.spearmanr(ki, au).statistic)
    rho = float(stats.spearmanr(ko, au).statistic)      # the selector the paper proposes

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
            idx = rng.integers(0, n, n)
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
    paths = resolve_data()
    tr_parts, te_parts, btr, bte, per = [], [], [], [], []
    for nm, p in paths:
        ri = load_returns(p)
        k = int(TRAIN_FRAC * len(ri))            # split each series by ITS OWN date range
        te_i = ri[k:] - np.median(ri[k:])      # Amendment 2: one constant per series
        tr_parts.append(ri[:k]); te_parts.append(te_i)
        btr.append(blocks_of(ri[:k])); bte.append(blocks_of(te_i))
        per.append((nm, len(ri), len(ri[:k]), len(ri[k:])))
    tr = np.concatenate(tr_parts); te = np.concatenate(te_parts)
    Btr = np.concatenate(btr, axis=0); Bte = np.concatenate(bte, axis=0)
    r = np.concatenate([np.concatenate(tr_parts), np.concatenate(te_parts)])

    print("=" * 78)
    print("REAL-DATA STUDY -- Ku-LSU, Statistical Papers revision.  seed =", SEED)
    print("pre-registration: review/REAL_DATA_SPEC_2026-09-16.md")
    print("=" * 78)
    print("\nsources  : " + ", ".join(n for n, _ in paths) +
          "   (fred-heavytail; SP500 and DCOILWTICO excluded by pre-registration)")
    for nm, n_all, n_tr, n_te in per:
        print(f"           {nm:<10} {n_all:>6} returns   train {n_tr:>6}  test {n_te:>6}")
    print(f"pooled   : {len(r)} daily log-returns; TRAIN {len(tr)} / TEST {len(te)} (60/40 by date)")
    print(f"blocks   : n={NBLOCK}; TRAIN {len(Btr)} blocks / TEST {len(Bte)} blocks")

    ah = hill_alpha(tr)
    print(f"\n[R-0] admissibility gate: Hill alpha_hat (TRAIN) = {ah:.3f}")
    if ah >= 4.0:
        print("      alpha_hat >= 4 -> the paper's heavy-tail regime does NOT apply.")
        print("      STUDY DECLARED UNINFORMATIVE (pre-registered abort).  No further arms run.")
        return
    ceil = ah / 2.0
    alphas = list(np.linspace(0.35 * ceil, 0.92 * ceil, 3))
    print(f"      alpha_hat < 4  -> PROCEED.  exponent ceiling = {ceil:.3f}, "
          f"A = {fmt_exponents(alphas)}")

    delta = 0.25 * float(stats.median_abs_deviation(tr, scale='normal'))
    print(f"      shift delta = 0.25 * MAD(train) = {delta:.6f}")

    print("\n[E] estimation on real noise with a known injected shift")
    e = branch_estimation(tr, Bte, alphas, delta, rng)
    print(f"    blocks of n={NBLOCK}: {e['nblocks']}")
    print(f"    {'estimator':<26}{'RMSE':>12}{'RMSE/best':>12}")
    for nm, v in e["rmse"].items():
        print(f"    {nm:<26}{v:>12.6f}{v/e['best']:>12.3f}")
    m, lo, hi = e["e1_stat"]
    print(f"    E-1 fractional <= polynomial (paired MSE diff {m:+.3e}, "
          f"95% CI [{lo:+.3e}, {hi:+.3e}]):  {'PASS' if e['e1'] else 'FAIL'}")
    print(f"    E-2 fractional within 5% of best RMSE:  {'PASS' if e['e2'] else 'FAIL'}")
    print(f"    cond(F): fractional {e['cond_frac']:.3g}, polynomial {e['cond_poly']:.3g} "
          f"(bases applied to x / {e['scale']:.3e}, the TRAIN robust scale)")
    print(f"             fractional on RAW returns, no scale: {e['cond_frac_raw']:.3g} "
          f"-- why the standardization is applied")

    print("\n[T] testing: critical values from TRAIN, size and power on TEST")
    t = branch_testing(tr, Btr, Bte, alphas, delta, rng)
    print(f"    blocks: {t['nblocks']}")
    print(f"    {'test':<32}{'size':>8}{'power':>9}{'paired vs ours':>20}{'z':>7}")
    for nm in t["power"]:
        if nm in t["paired"]:
            d, se, z = t["paired"][nm]
            tail = f"{d:>+13.4f} +/-{se:.4f}{z:>7.1f}"
        else:
            tail = f"{'(reference)':>20}{'':>7}"
        print(f"    {nm:<32}{t['size'][nm]:>8.3f}{t['power'][nm]:>9.4f}{tail}")
    print(f"    T-1 (PRIMARY, pre-registered: critical values from TRAIN)")
    print(f"        vs best classical arm ({t['best_classical']}): z = {t['z']:+.2f} "
          f"-> {t['verdict']}")
    print(f"    SECONDARY (Amendment 2: critical values recalibrated on TEST null blocks,")
    print(f"               so every arm has size 0.05 where the power is measured)")
    print(f"        {'test':<32}{'power':>9}{'paired vs ours':>20}{'z':>7}")
    for nm in t["power2"]:
        if nm in t["paired2"]:
            d, se, z = t["paired2"][nm]
            tail = f"{d:>+13.4f} +/-{se:.4f}{z:>7.1f}"
        else:
            tail = f"{'(reference)':>20}{'':>7}"
        print(f"        {nm:<32}{t['power2'][nm]:>9.4f}{tail}")
    print(f"        vs best classical arm ({t['best_classical2']}): z = {t['z2']:+.2f} "
          f"-> {t['verdict2']}  (secondary; does NOT replace the primary verdict)")

    print("\n[C-loc] classification, heavy tail is a SHARED nuisance (favourable regime)")
    cl = branch_cls_location(tr, te, alphas, delta, rng)
    for nm, v in cl["auc"].items():
        print(f"    {nm:<34}AUC {v:.4f}")
    print(f"    C-1 fractional >= polynomial (diff {cl['diff']:+.4f}, "
          f"95% CI [{cl['ci'][0]:+.4f}, {cl['ci'][1]:+.4f}]):  "
          f"{'PASS' if cl['c1'] else 'FAIL'}")

    print("\n[C-scale] classification, heavy tail IS the contrast (adverse regime)")
    cs = branch_cls_scale(r, alphas, rng)
    print(f"    labels from a DISJOINT trailing {VOL_WINDOW}-day realized-volatility window;")
    print(f"    TRAIN {cs['n_train']} / TEST {cs['n_test']}, class-1 share on TEST "
          f"{cs['frac_class1']:.3f}")
    print(f"    {'dictionary':<14}{'cols':>6}{'kappa in-samp':>15}{'kappa HELD-OUT':>16}"
          f"{'AUC (TEST)':>12}{'95% CI vs best':>22}")
    for nm, v in cs["rows"].items():
        ci = "  (best arm)" if v["ci"] is None else f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
        print(f"    {nm:<14}{v['ncol']:>6}{v['kappa_in']:>15.4f}{v['kappa_out']:>16.4f}"
              f"{v['auc_test']:>12.4f}{ci:>22}")
    print(f"    Spearman(kappa HELD-OUT, AUC) = {cs['spearman']:+.3f}   "
          f"[in-sample kappa would give {cs['spearman_in']:+.3f}]")
    print(f"    AUC spread across all {cs['n_arms']} dictionaries = {cs['spread']:.4f}; "
          f"{cs['n_distinguishable']} of {cs['n_arms']-1} arms separable from the best")
    print(f"    C-2 pre-registered prediction 'non-positive':  "
          f"{'CONFIRMED' if cs['c2'] else 'REFUTED'}")

    # ------------------------------------------------------------------ #
    # Amendment 3 (EXPLORATORY, written after the primary outcomes were seen):
    # is volatility clustering the cause?  Devolatilize each block by its OWN
    # robust scale and repeat E and C-loc.
    # ------------------------------------------------------------------ #
    print("\n[A3] DIAGNOSTIC (exploratory, not pre-registered): block-devolatilized returns")
    print("     each block divided by its own MAD before any basis is applied")
    Bd = Bte / np.maximum(stats.median_abs_deviation(Bte, axis=1, scale="normal",
                                                     keepdims=True), 1e-12)
    trd = np.concatenate([bb / max(float(stats.median_abs_deviation(bb, scale="normal")), 1e-12)
                          for bb in Btr])
    ahd = hill_alpha(trd)
    ceild = ahd / 2.0
    Ad = list(np.linspace(0.35 * ceild, 0.92 * ceild, 3))
    deld = 0.25 * float(stats.median_abs_deviation(trd, scale="normal"))
    print(f"     Hill alpha_hat after devolatilization = {ahd:.3f} "
          f"(was {ah:.3f})   A = {fmt_exponents(Ad)}")
    ed = branch_estimation(trd, Bd, Ad, deld, rng)
    print(f"     {'estimator':<26}{'RMSE':>12}{'RMSE/best':>12}")
    for nm, v in ed["rmse"].items():
        print(f"     {nm:<26}{v:>12.6f}{v/ed['best']:>12.3f}")
    m2, lo2, hi2 = ed["e1_stat"]
    print(f"     E-1' fractional <= polynomial (paired MSE diff {m2:+.3e}, "
          f"95% CI [{lo2:+.3e}, {hi2:+.3e}]):  {'PASS' if ed['e1'] else 'FAIL'}")
    print(f"     E-2' fractional within 5% of best:  {'PASS' if ed['e2'] else 'FAIL'}")

    cld = branch_cls_location(trd, np.concatenate(list(Bd)), Ad, deld, rng)
    for nm, v in cld["auc"].items():
        print(f"     {nm:<34}AUC {v:.4f}")
    print(f"     C-1' fractional >= polynomial (diff {cld['diff']:+.4f}, "
          f"95% CI [{cld['ci'][0]:+.4f}, {cld['ci'][1]:+.4f}]):  "
          f"{'PASS' if cld['c1'] else 'FAIL'}")
    verdict = "SUPPORTED" if (ed["e1"] or cld["c1"]) else "NOT SUPPORTED"
    print(f"     A-3 prediction (edge recovers on at least one of E-1', C-1'): {verdict}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
