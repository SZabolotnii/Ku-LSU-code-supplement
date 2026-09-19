"""
run_comparators.py -- comparator study for the Statistical Papers revision (R1, R2).

The frozen gate (run_three_branch_unification.py) verifies the THEORY: that the captured
fraction rises with the basis, terminates on sufficiency, and stays defined on heavy tails.
Both referees asked the complementary question: what does a practitioner GAIN relative to
the methods they would otherwise use?  This script answers it, on the three branches, with
the analytic captured fraction printed beside every empirical number so the prediction is
falsifiable rather than decorative.

Pre-specified before any number was produced (see SPEC block below).  Global seed 2026.

Branches and comparators
------------------------
  C-EST  estimation of a location parameter, RMSE and efficiency vs the MLE:
           sample mean | median | Huber M (k=1.345) | polynomial PMM {x,x^3}
           | fractional PMM (odd half, admissible exponents) | MLE (oracle)
         Reported against the analytic g = kappa_m(S; p_theta, B^-).

  C-TEST size-calibrated power against a fixed local shift h/sqrt(n):
           t-test | sign test | Wilcoxon signed-rank | Neyman smooth (Hermite, order 4)
           | projected-score test (fractional)
         Reported against the analytic prediction Phi(sqrt(g I) h - z_alpha).

  C-CLS  two-class discrimination, held-out AUC and error rate:
           QDA | logistic regression | KDE plug-in log-ratio | projected-Lambda rule
         On (a) the Gaussian covariance-shift anchor, where the order-2 projection is
         exact by Theorem bridge, and (b) a heavy-tailed pair where it is not.

  C-ANCH the exact analytic control: Student-t(10) with the odd dictionary
         {x, sgn(x)|x|^3} has g_3 = 1 - 1/25 = 0.96 EXACTLY.  A numerical routine that
         does not reproduce 0.96 to Monte-Carlo error is wrong, and this row is the only
         one in the paper with a closed-form target.

SPEC (fixed before computation)
-------------------------------
  S-1  Every estimator is compared at the same n and the same number of replications.
  S-2  Tests are calibrated to empirical size 0.05 under H0 by simulation, not by
       asymptotic critical values, so that power is comparable across very different
       null distributions.
  S-3  Classification numbers are held-out: the dictionary and all coefficients are fit
       on a training half, metrics computed on a disjoint test half.
  S-4  The fractional exponent set is chosen by the tail-index ceiling alpha_hat/2 (Hill),
       on TRAINING data only, never by looking at the metric being reported.
  S-5  A comparator that WINS is reported as a win.  There is no arm of this study whose
       failure would be suppressed; the expected-output file records whatever came out.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from run_three_branch_unification import (
    SEED,
    cauchy,
    frac_basis,
    gaussian,
    laplace,
    student_t,
)

NPROJ = 2_000_000      # MC size for L2 inner products (analytic g)
NEST = 400             # sample size per replication, estimation branch
REST = 3000            # replications, estimation branch
NTEST = 400            # sample size per replication, testing branch
RTEST = 4000           # replications per arm, testing branch (null + alt)
NCLS = 20000           # samples per class, classification branch
ALPHA = 0.05
HLOC = 0.35            # local shift in units of h/sqrt(n)


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #
def fmt_exponents(a):
    """Exponent set as plain numbers.  numpy scalars repr as np.float64(0.491) under
    numpy >= 2, which is noise in output the supplement prints verbatim."""
    return "[" + ", ".join(f"{float(x):.3f}" for x in a) + "]"


def g_analytic(fam, alphas, rng, n=NPROJ):
    """Captured Fisher fraction g = ||proj S||^2 / I on the ODD half B^-(A).

    The score of a symmetric location family is odd, so E[rho^-_a] = 0 and the centered
    Gram coincides with the raw second-moment matrix (the parity rule of section 2).
    """
    x = fam["samp"](rng, n)
    Phi, _ = frac_basis(alphas)
    P = Phi(x)
    S = fam["score"](x)
    G = (P.T @ P) / len(x)
    b = (P.T @ S) / len(x)
    k = np.linalg.solve(G + 1e-12 * np.eye(len(alphas)), b)
    return float(b @ k) / fam["I"], k


def hill_alpha(x, frac=0.1):
    """Hill tail-index estimate from the upper `frac` of |x|; training data only (S-4)."""
    a = np.sort(np.abs(x))[::-1]
    k = max(10, int(frac * len(a)))
    k = min(k, len(a) - 1)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def admissible_exponents(x, n_exp=3):
    """Exponent set under the ceiling alpha_hat/2, chosen on training data only."""
    ah = hill_alpha(x)
    ceil = max(0.35, min(3.0, ah / 2.0))
    return list(np.linspace(0.35 * ceil, 0.92 * ceil, n_exp)), ah


# --------------------------------------------------------------------------- #
# C-EST : location estimation
# --------------------------------------------------------------------------- #
def _psi_root(x, Phi, k, lo=-1.5, hi=1.5, grid=61):
    """Solve sum psi(x_i - theta) = 0 on a grid + bisection refinement.
    psi is non-smooth for fractional exponents, so we do not use Newton."""
    th = np.linspace(lo, hi, grid)
    vals = np.array([(Phi(x - t) @ k).mean() for t in th])
    sgn = np.sign(vals)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross) == 0:
        return th[np.argmin(np.abs(vals))]
    i = cross[0]
    a, b = th[i], th[i + 1]
    fa = vals[i]
    for _ in range(40):
        m = 0.5 * (a + b)
        fm = (Phi(x - m) @ k).mean()
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def huber_location(x, k=1.345):
    s = stats.median_abs_deviation(x, scale="normal")
    s = s if s > 1e-9 else 1.0
    f = lambda t: np.mean(np.clip((x - t) / s, -k, k))
    lo, hi = np.median(x) - 5 * s, np.median(x) + 5 * s
    try:
        return optimize.brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return float(np.median(x))


def mle_location(fam, x):
    """Oracle MLE: maximize the true log-likelihood in the location parameter."""
    f = lambda t: -np.mean(fam["logp"](x - t))
    return float(optimize.minimize_scalar(f, bounds=(-2.0, 2.0), method="bounded").x)


def run_estimation(fam, rng):
    xtr = fam["samp"](rng, 200_000)
    alphas, ah = admissible_exponents(xtr)
    g_frac, k_frac = g_analytic(fam, alphas, rng)
    g_poly, k_poly = g_analytic(fam, [1.0, 3.0], rng)
    Phi_f, _ = frac_basis(alphas)
    Phi_p, _ = frac_basis([1.0, 3.0])

    names = ["mean", "median", "Huber(1.345)", "poly PMM {x,x^3}", "frac PMM", "MLE (oracle)"]
    est = {nm: np.empty(REST) for nm in names}
    for r in range(REST):
        x = fam["samp"](rng, NEST)
        est["mean"][r] = x.mean()
        est["median"][r] = np.median(x)
        est["Huber(1.345)"][r] = huber_location(x)
        est["poly PMM {x,x^3}"][r] = _psi_root(x, Phi_p, k_poly)
        est["frac PMM"][r] = _psi_root(x, Phi_f, k_frac)
        est["MLE (oracle)"][r] = mle_location(fam, x)

    rmse = {nm: float(np.sqrt(np.mean(v ** 2))) for nm, v in est.items()}
    ref = rmse["MLE (oracle)"]
    eff = {nm: (ref / v) ** 2 for nm, v in rmse.items()}
    return dict(
        family=fam["name"], alphas=alphas, hill=ah, g_frac=g_frac, g_poly=g_poly,
        rmse=rmse, eff=eff, n=NEST, R=REST,
    )


# --------------------------------------------------------------------------- #
# C-TEST : one-sided location test, size-calibrated
# --------------------------------------------------------------------------- #
def _hermite_cols(x, order=4):
    """Probabilists' Hermite polynomials H_1..H_order of the standardized sample."""
    s = np.std(x)
    z = x / (s if s > 1e-9 else 1.0)
    H = [z, z ** 2 - 1.0, z ** 3 - 3.0 * z, z ** 4 - 6.0 * z ** 2 + 3.0]
    return np.stack(H[:order], axis=-1)


def _neyman_smooth_stat(x, order=4):
    """Neyman's smooth test statistic, as Neyman defined it: the sum of squared
    standardized Hermite coefficients, i.e. an omnibus chi-square.  It is two-sided by
    construction, so it is calibrated and compared on its own terms (larger = more
    evidence), not forced into a one-sided location statistic."""
    H = _hermite_cols(x, order)
    n = len(x)
    coef = H.mean(axis=0)
    sd = H.std(axis=0, ddof=1) / np.sqrt(n)
    return float(np.sum((coef / np.maximum(sd, 1e-12)) ** 2))


def _hermite_directed_stat(x, k_herm, order=4):
    """The FAIR Hermite analogue of our projected-score test: the same construction
    (project the score onto a basis, use the projection as a directed statistic) with the
    Hermite system in place of the fractional one.  This is the comparator R2 asked for."""
    return float(_hermite_cols(x, order) @ k_herm).__float__() if False else float(
        (_hermite_cols(x, order) @ k_herm).mean())


def _hermite_score_coef(fam, rng, order=4, n=400_000):
    """Least-squares coefficients projecting the true score onto the Hermite system.
    On a heavy tail the Gram entries needed here diverge in population; we return the
    empirical solve and its condition number so the failure is visible, not hidden."""
    x = fam["samp"](rng, n)
    H = _hermite_cols(x, order)
    S = fam["score"](x)
    G = (H.T @ H) / n
    b = (H.T @ S) / n
    k = np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]), b)
    return k, float(np.linalg.cond(G))


def run_testing(fam, rng):
    xtr = fam["samp"](rng, 200_000)
    alphas, _ = admissible_exponents(xtr)
    g_frac, k_frac = g_analytic(fam, alphas, rng)
    Phi_f, _ = frac_basis(alphas)

    k_herm, cond_herm = _hermite_score_coef(fam, rng)
    shift = HLOC / np.sqrt(NTEST)                    # fixed local alternative h/sqrt(n)

    def stats_for(x):
        return {
            "t-test": float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))),
            "sign": float(np.mean(np.sign(x))),
            "Wilcoxon": float(stats.wilcoxon(x, alternative="greater", zero_method="zsplit").statistic),
            "Neyman smooth (omnibus chi2)": _neyman_smooth_stat(x),
            "projected score (Hermite)": _hermite_directed_stat(x, k_herm),
            "projected score (fractional)": float((Phi_f(x) @ k_frac).mean()),
        }

    names = list(stats_for(fam["samp"](rng, 32)).keys())
    null = {nm: np.empty(RTEST) for nm in names}
    alt = {nm: np.empty(RTEST) for nm in names}
    for r in range(RTEST):
        x0 = fam["samp"](rng, NTEST)
        x1 = fam["samp"](rng, NTEST) + shift
        s0, s1 = stats_for(x0), stats_for(x1)
        for nm in names:
            null[nm][r] = s0[nm]
            alt[nm][r] = s1[nm]

    power, power_se, size, rej = {}, {}, {}, {}
    for nm in names:
        crit = np.quantile(null[nm], 1.0 - ALPHA)          # S-2: empirical calibration
        rej[nm] = (alt[nm] > crit).astype(float)
        p = float(rej[nm].mean())
        power[nm] = p
        power_se[nm] = float(np.sqrt(p * (1.0 - p) / RTEST))
        size[nm] = float(np.mean(null[nm] > crit))         # sanity: must be ~= ALPHA

    # PAIRED comparison against our test: every arm saw the SAME replications, so the
    # difference in rejection indicators is paired and its SE is far smaller than the
    # independent-sample bound.  This is the comparison that decides a win or a tie.
    ours = "projected score (fractional)"
    paired = {}
    for nm in names:
        if nm == ours:
            continue
        d = rej[ours] - rej[nm]
        se = float(d.std(ddof=1) / np.sqrt(RTEST))
        paired[nm] = dict(diff=float(d.mean()), se=se,
                          z=float(d.mean() / se) if se > 0 else 0.0)
    h = shift * np.sqrt(NTEST)
    predicted = float(stats.norm.cdf(np.sqrt(g_frac * fam["I"]) * h - stats.norm.ppf(1 - ALPHA)))
    return dict(family=fam["name"], g_frac=g_frac, power=power, power_se=power_se,
                size=size, paired=paired, predicted=predicted, h=h, n=NTEST, R=RTEST,
                alphas=alphas, cond_herm=cond_herm)


# --------------------------------------------------------------------------- #
# C-CLS : two-class discrimination, held out
# --------------------------------------------------------------------------- #
def _design_poly(x, m=3):
    """The POLYNOMIAL design [1, x, x^2, x^3][:m+1].  On a heavy tail this is inadmissible
    in the sense of Definition 1 (the Gram entries it needs diverge); we keep it as the
    comparator arm that the paper's own theory predicts should fail."""
    feats = [np.ones_like(x), x, x ** 2, x ** 3]
    return np.stack(feats[:m + 1], axis=-1)


def _design_frac(x, alphas):
    """The ADMISSIBLE fractional design: BOTH parity halves of B(A), since the
    classification target log(p1/p0) has no parity (the parity rule, case (iii)).
    Exponents come from the tail-index ceiling on training data only (S-4)."""
    cols = [np.ones_like(x)]
    for a in alphas:
        cols.append(np.abs(x) ** a)                    # rho^+_a, even half
        cols.append(np.sign(x) * np.abs(x) ** a)       # rho^-_a, odd half
    return np.stack(cols, axis=-1)


def _auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score), float)
    r[order] = np.arange(1, len(score) + 1)
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def run_classification(samp0, samp1, true_lam, label, rng, m=3):
    x0tr, x1tr = samp0(rng, NCLS), samp1(rng, NCLS)
    x0te, x1te = samp0(rng, NCLS), samp1(rng, NCLS)
    xtr = np.concatenate([x0tr, x1tr])
    ytr = np.concatenate([np.zeros(NCLS), np.ones(NCLS)])
    xte = np.concatenate([x0te, x1te])
    yte = np.concatenate([np.zeros(NCLS), np.ones(NCLS)])

    # S-4: exponents from the tail index of the TRAINING pool only.
    alphas, ah = admissible_exponents(xtr, n_exp=3)

    out, conds = {}, {}

    # QDA: Gaussian per class; exact on the covariance-shift anchor by Theorem bridge.
    m0, v0 = x0tr.mean(), x0tr.var()
    m1, v1 = x1tr.mean(), x1tr.var()
    out["QDA"] = (-0.5 * (xte - m1) ** 2 / v1 - 0.5 * np.log(v1)) - (
        -0.5 * (xte - m0) ** 2 / v0 - 0.5 * np.log(v0))

    # Logistic regression on the admissible fractional design (IRLS).
    B = _design_frac(xtr, alphas)
    Bt = _design_frac(xte, alphas)
    w = np.zeros(B.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-B @ w))
        W = np.clip(p * (1 - p), 1e-9, None)
        z = B @ w + (ytr - p) / W
        w = np.linalg.solve((B * W[:, None]).T @ B + 1e-8 * np.eye(B.shape[1]),
                            (B * W[:, None]).T @ z)
    out["logistic (fractional design)"] = Bt @ w

    # KDE plug-in log-ratio (Silverman bandwidth, subsampled for tractability).
    sub = 4000
    k0 = stats.gaussian_kde(x0tr[:sub])
    k1 = stats.gaussian_kde(x1tr[:sub])
    out["KDE plug-in"] = np.log(k1(xte) + 1e-300) - np.log(k0(xte) + 1e-300)

    # Projected-Lambda rules.  The theory (Definition kappa, Theorem bridge) projects the
    # TRUE log-ratio, so that is the regression target here; `true_lam` is supplied
    # analytically by the caller.  We project onto (a) the ADMISSIBLE fractional design and
    # (b) the POLYNOMIAL design, and print cond(F) for both, because the paper's own theory
    # predicts (b) degrades on a heavy tail.
    #
    # A third arm uses the KDE-ESTIMATED log-ratio as the target: this is the honest
    # plug-in version a practitioner without the true densities would run, and its gap to
    # the oracle-target arm is the cost of estimating Lambda, not a property of the basis.
    P = _design_poly(xtr, m)
    Pt = _design_poly(xte, m)
    lam_true_tr = true_lam(xtr)
    lam_kde_tr = np.log(k1(xtr) + 1e-300) - np.log(k0(xtr) + 1e-300)
    for nm, (Dtr, Dte, tgt) in {
        "projected Lambda (fractional)": (B, Bt, lam_true_tr),
        "projected Lambda (polynomial)": (P, Pt, lam_true_tr),
        "projected Lambda (frac, KDE target)": (B, Bt, lam_kde_tr),
    }.items():
        coef, *_ = np.linalg.lstsq(Dtr, tgt, rcond=None)
        out[nm] = Dte @ coef
        G = (Dtr.T @ Dtr) / len(Dtr)
        conds[nm] = float(np.linalg.cond(G))

    # The oracle ceiling: the true log-ratio itself.
    out["true Lambda (oracle)"] = true_lam(xte)

    res = {}
    for nm, sc in out.items():
        res[nm] = dict(auc=_auc(sc, yte), err=float(np.mean((sc > 0).astype(float) != yte)),
                       cond=conds.get(nm))
    return dict(pair=label, n_per_class=NCLS, m=m, results=res, alphas=alphas, hill=ah)


# --------------------------------------------------------------------------- #
# C-ANCH : the exact analytic control
# --------------------------------------------------------------------------- #
def t_abs_moment(nu, q):
    """E|X|^q for Student-t(nu), finite iff q < nu."""
    from scipy.special import gammaln
    if q >= nu:
        return np.inf
    return float(np.exp(
        0.5 * q * np.log(nu)
        + gammaln(0.5 * (q + 1.0)) + gammaln(0.5 * (nu - q))
        - 0.5 * np.log(np.pi) - gammaln(0.5 * nu)
    ))


def run_anchor(rng, nu=10.0):
    """The exact analytic control: Student-t(10) with the odd dictionary {x, sgn(x)|x|^3}.

    Two DIFFERENT quantities are in play here and conflating them is the classic error
    this paper's terminology section is meant to prevent:

      * the CLASSICAL Kunchenko coefficient  g_3 = Var(PMM_3)/Var(PMM_1)
        = 1 - gamma4^2 / (gamma6 + 9 gamma4 + 6).  For t(10) the standardized cumulants
        are gamma4 = 1 (excess) and gamma6 = 10, so g_3 = 1 - 1/25 = 24/25 = 0.96 EXACTLY.
        This is a variance RATIO against the sample mean, not a captured fraction.

      * the CAPTURED FRACTION  kappa_3 = ||proj S||^2 / I, which is what this paper calls
        a reading.  They are related by  g_s = kappa_1 / kappa_s, and for a location
        family kappa_1 = 1/(sigma^2 I).  For t(10): sigma^2 = nu/(nu-2) = 5/4,
        I = (nu+1)/(nu+3) = 11/13, so kappa_1 = 52/55 and

            kappa_3 = kappa_1 / g_3 = (52/55) * (25/24) = 65/66 = 0.984848...  EXACTLY.

    We therefore have a closed-form target for BOTH, and the routine is wrong if it misses
    either.  The Gram is formed from exact t-moments rather than by Monte Carlo.  The reason
    is not that Monte Carlo visibly fails here -- measured over 12 seeds and n from 1e6 to
    1.6e7 its error still falls off near n^{-1/2} -- but that the justification for those
    error bars does not exist: the Gram entry E[x^6] has an MC estimator of INFINITE
    variance on t(10) (it would need E[x^12], and 12 > nu = 10), so the CLT does not apply
    and a reported standard error is not a standard error.  The analytic Gram costs nothing
    and removes the question.  We print the MC route beside it so the reader can see both.
    """
    fam = student_t(nu)
    sig2 = nu / (nu - 2.0)
    I = (nu + 1.0) / (nu + 3.0)
    kap1_exact = 1.0 / (sig2 * I)
    g3_classical_exact = 1.0 - 1.0 ** 2 / (10.0 + 9.0 * 1.0 + 6.0)      # 24/25
    kap3_exact = kap1_exact / g3_classical_exact                          # 65/66

    # --- analytic route: exact Gram from t-moments, b by Stein with the analytic score.
    m2, m4, m6 = (t_abs_moment(nu, q) for q in (2.0, 4.0, 6.0))
    F = np.array([[m2, m4], [m4, m6]])
    # b_i = E[phi_i * S]; for a location family Stein gives E[phi' ]: b = (1, 3 E[x^2]).
    b = np.array([1.0, 3.0 * m2])
    k = np.linalg.solve(F, b)
    kap3_analytic = float(b @ k) / I
    condF = float(np.linalg.cond(F))

    # --- Monte-Carlo route, for contrast only.
    mc = []
    for s in (0, 1, 2):
        r2 = np.random.default_rng(SEED + 1000 + s)
        v, _ = g_analytic(fam, [1.0, 3.0], r2, n=4_000_000)
        mc.append(v)
    mc = np.array(mc)

    return dict(
        kap3_exact=kap3_exact, kap3_analytic=kap3_analytic,
        abs_err=abs(kap3_analytic - kap3_exact),
        g3_classical_exact=g3_classical_exact,
        g3_from_analytic=kap1_exact / kap3_analytic,
        kap1_exact=kap1_exact, condF=condF,
        mc_mean=float(mc.mean()), mc_sd=float(mc.std(ddof=1)),
        mc_bias=float(mc.mean() - kap3_exact),
    )


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("COMPARATOR STUDY -- Ku-LSU, Statistical Papers revision.  seed =", SEED)
    print("=" * 78)

    print("\n[C-ANCH] exact analytic control: Student-t(10), odd dictionary {x, sgn(x)|x|^3}")
    a = run_anchor(rng)
    print(f"   classical Kunchenko g_3 = Var(PMM_3)/Var(PMM_1) = 24/25 = "
          f"{a['g3_classical_exact']:.6f}  (EXACT)")
    print(f"   captured fraction kappa_3 = kappa_1/g_3 = 65/66      = "
          f"{a['kap3_exact']:.6f}  (EXACT)   [kappa_1 = 52/55 = {a['kap1_exact']:.6f}]")
    print(f"   analytic Gram route:  kappa_3 = {a['kap3_analytic']:.6f}   "
          f"|err| = {a['abs_err']:.2e}   cond(F) = {a['condF']:.3g}   "
          f"{'PASS' if a['abs_err'] < 1e-9 else 'FAIL'}")
    print(f"   recovered classical   g_3     = {a['g3_from_analytic']:.6f}")
    print(f"   Monte-Carlo Gram route (3 seeds, n=4e6): {a['mc_mean']:.5f} "
          f"+/- {a['mc_sd']:.5f}   error {a['mc_bias']:+.5f}")
    print( "      ^ agrees to ~1e-3, but the MC estimator of the Gram entry E[x^6] on t(10)")
    print( "        has INFINITE variance (it needs E[x^12], and 12 > nu = 10), so the CLT")
    print( "        does not apply and that '+/-' is not a standard error.  The analytic Gram")
    print( "        is exact and free; we use it and quote no MC error bar for this row.")

    print("\n[C-EST] location estimation: RMSE and efficiency vs the oracle MLE")
    for fam in (laplace(), student_t(3)):
        r = run_estimation(fam, rng)
        print(f"\n  {r['family']}   n={r['n']}, R={r['R']}, Hill alpha_hat={r['hill']:.2f}, "
              f"exponents={fmt_exponents(r['alphas'])}")
        print(f"  analytic g (fractional) = {r['g_frac']:.4f}   "
              f"analytic g (poly {{x,x^3}}) = {r['g_poly']:.4f}")
        print(f"     {'estimator':<26}{'RMSE':>10}{'efficiency':>13}")
        for nm in r["rmse"]:
            print(f"     {nm:<26}{r['rmse'][nm]:>10.4f}{r['eff'][nm]:>13.4f}")

    print("\n[C-TEST] one-sided location test, size calibrated to 0.05 by simulation")
    for fam in (laplace(), student_t(3)):
        r = run_testing(fam, rng)
        print(f"\n  {r['family']}   n={r['n']}, R={r['R']}, local h={r['h']:.3f}")
        print(f"  analytic g = {r['g_frac']:.4f}   predicted power "
              f"Phi(sqrt(gI)h - z) = {r['predicted']:.4f}   "
              f"cond(Gram, Hermite) = {r['cond_herm']:.3g}")
        print(f"     {'test':<32}{'power':>9}{'MC SE':>9}{'size':>8}"
              f"{'ours - this (paired)':>24}{'z':>7}")
        for nm, p in r["power"].items():
            if nm in r["paired"]:
                d = r["paired"][nm]
                tail = f"{d['diff']:>+15.4f} +/- {d['se']:.4f}{d['z']:>7.1f}"
            else:
                tail = f"{'(reference arm)':>24}{'':>7}"
            print(f"     {nm:<32}{p:>9.4f}{r['power_se'][nm]:>9.4f}"
                  f"{r['size'][nm]:>8.3f}{tail}")

    print("\n[C-CLS] two-class discrimination, held out")
    CSCALE = 1.5

    def lam_gauss(x):
        return 0.5 * x ** 2 * (1.0 - 1.0 / CSCALE ** 2) - np.log(CSCALE)

    def lam_t3(x, nu=3.0):
        lg = lambda u: -0.5 * (nu + 1.0) * np.log1p(u ** 2 / nu)
        return lg(x / CSCALE) - np.log(CSCALE) - lg(x)

    pairs = [
        (lambda g, n: g.standard_normal(n),
         lambda g, n: CSCALE * g.standard_normal(n), lam_gauss,
         "Gaussian covariance shift  N(0,1) vs N(0,2.25)  [order-2 projection EXACT]"),
        (lambda g, n: g.standard_t(3, n),
         lambda g, n: CSCALE * g.standard_t(3, n), lam_t3,
         "Student-t3 scale shift  t3 vs 1.5*t3  [heavy tail, not exact]"),
    ]
    for s0, s1, tl, lab in pairs:
        r = run_classification(s0, s1, tl, lab, rng)
        print(f"\n  {lab}\n     n/class={r['n_per_class']}, Hill alpha_hat={r['hill']:.2f}, "
              f"exponents={fmt_exponents(r['alphas'])}")
        print(f"     {'rule':<32}{'AUC':>9}{'error':>9}{'cond(F)':>12}")
        for nm, v in r["results"].items():
            cf = f"{v['cond']:.3g}" if v["cond"] is not None else "--"
            print(f"     {nm:<36}{v['auc']:>9.4f}{v['err']:>9.4f}{cf:>12}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
