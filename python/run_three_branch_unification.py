"""Verifying gate for the three-branch unification (see Spec-Unification-LikelihoodSeries.md).

Demonstrates that PMM (estimation), detection (LLR), and DSGE (decomposition) are three
FUNCTIONALS of ONE object: the truncated expansion of log p(x;θ) in a Kunchenko basis.

Two families x three branches, all closed-form checkable (Monte-Carlo projections, seed 2026):

  ANCHOR  -- Gaussian (the basis SPANS the truth; the series terminates):
    A1  estimation: score S=x is linear -> projection onto {x} captures it -> g = 1.
    A2  detection : H0=N(0,1), H1=N(δ,1) -> LLR is linear in x; the per-class Gaussian
        DSGE/QDA discriminant EQUALS the true LLR to machine precision (Theorem A), c = 1.

  TRUNCATION -- Laplace (the basis APPROXIMATES; residual R_m shrinks with order m):
    T1  estimation: score S=sign(x)/b; projected onto a nested fractional basis sgn(x)|x|^a,
        the captured Fisher fraction g(m) rises monotonically with m.
    T2  reading identity: the analytic g(m) = ‖proj S‖²/I equals the EMPIRICAL relative
        efficiency (CRB / Var of the one-step PMM estimator) to <0.05 -> g really IS the
        estimator efficiency (ties oPMMα's g(α;C) to the master object).
    T3  detection : H0=Lap(0), H1=Lap(δ); LLR is non-linear (|x| kink), captured fraction
        c(m) rises with m.

  GSA ANCHOR -- Gaussian COVARIANCE shift (sequential FORM of detection: change-point):
    A3  H0=N(0,1) vs H1=N(0,2.25).  The true LLR is the explicit degree-2 polynomial of
        Lean `gauss_llr_eq_quadratic`, so:
          a) the order-2 projection is exact (c=1, GSA@S=2 = true LLR);
          b) the Mahalanobis-affine (Ghosh) head equals the LLR identically;
          c) the sign of the projected statistic is the Bayes decision;
          d) DESIGNED BLINDNESS: order-1 captures ~nothing (pure even signal, odd basis) --
             the S=1->S=2 ladder predicted a priori, not post hoc;
          e) the GSA normal system F·K=Y (F=Cov0+Cov1, Y=E1-E0, theoretical moments, the
             KuYuPe convention) yields a statistic collinear with the true LLR.
  T4  reading identity for the GSA branch: the information functional J(s) (partial
      Parseval sum over the orthonormalized basis, the InfoFunctional.lean object) equals
      the captured-LLR fraction c(m) on the SAME sample -- J(s)/‖z‖² IS the detection
      reading; saturates at 1 exactly where the series terminates (Gaussian quadratic),
      stays <1 and rises elsewhere (Laplace).

PASS iff A1 ∧ A2 ∧ T1 ∧ T2 ∧ T3 ∧ A3 ∧ T4.   Self-contained numpy/scipy.
Run: python run_three_branch_unification.py
"""

from __future__ import annotations

import numpy as np

SEED = 2026
NPROJ = 4_000_000      # MC sample size for L2 projections (inner products)


# --------------------------------------------------------------------------- #
# Densities, scores, samplers (standardized to unit variance)
# --------------------------------------------------------------------------- #
def gaussian():
    s = lambda x: x                                  # score of N(0,1):  -d/dx log φ = x
    logp = lambda x: -0.5 * x ** 2 - 0.5 * np.log(2 * np.pi)
    samp = lambda rng, n: rng.standard_normal(n)
    return dict(name="Gaussian", score=s, logp=logp, samp=samp, I=1.0)


def laplace():
    b = 1.0 / np.sqrt(2.0)                            # scale for unit variance (Var=2b²=1)
    s = lambda x: np.sign(x) / b                      # score: -d/dx log f = sign(x)/b
    logp = lambda x: -np.abs(x) / b - np.log(2 * b)
    samp = lambda rng, n: rng.laplace(0.0, b, n)
    return dict(name="Laplace", score=s, logp=logp, samp=samp, I=1.0 / b ** 2)


def student_t(df):
    """Heavy-tailed Student-t(df): E[X^q] finite iff q < df, so the fractional element
    sgn(x)|x|^a stays in L^2 only while 2a < df — exactly where a monomial/Hermite basis
    (needing E[x^4], E[x^6]) diverges.  Location score (ν+1)x/(ν+x²); I = (ν+1)/(ν+3)."""
    nu = float(df)
    s = lambda x: (nu + 1.0) * x / (nu + x ** 2)               # -d/dx log f_t
    logp = lambda x: -(nu + 1.0) / 2.0 * np.log1p(x ** 2 / nu)  # up to const (cancels, centered R²)
    samp = lambda rng, n: rng.standard_t(nu, n)
    return dict(name=f"Student-t(df={nu:g})", score=s, logp=logp, samp=samp, I=(nu + 1.0) / (nu + 3.0))


def cauchy():
    """The extreme heavy tail (= t with df=1): NO polynomial moment exists, yet the
    fractional element with 2a < 1 remains L².  Score 2x/(1+x²); I = 1/2."""
    s = lambda x: 2.0 * x / (1.0 + x ** 2)
    logp = lambda x: -np.log1p(x ** 2)
    samp = lambda rng, n: rng.standard_cauchy(n)
    return dict(name="Cauchy", score=s, logp=logp, samp=samp, I=0.5)


# --------------------------------------------------------------------------- #
# Basis families (identity element handled per-branch)
# --------------------------------------------------------------------------- #
def frac_basis(alphas):
    """Odd fractional/PATP basis rows ρ_a(x)=sgn(x)|x|^a (for the symmetric score)."""
    def Phi(x):
        x = np.asarray(x, float)
        return np.stack([np.sign(x) * np.abs(x) ** a for a in alphas], axis=-1)
    def dPhi(x):
        x = np.asarray(x, float)
        ax = np.maximum(np.abs(x), 1e-12)
        return np.stack([a * ax ** (a - 1) for a in alphas], axis=-1)   # d/dx sgn|x|^a = a|x|^{a-1}
    return Phi, dPhi


def detect_basis(x, m):
    """Nested even+odd basis prefix for projecting the LLR: [1, x, x², |x|, x³, |x|³][:m]."""
    feats = [np.ones_like(x), x, x ** 2, np.abs(x), x ** 3, np.abs(x) ** 3]
    return np.stack(feats[:m], axis=-1)


# --------------------------------------------------------------------------- #
# Branch 1 — PMM/estimation: analytic g = ‖proj S‖²/I
# --------------------------------------------------------------------------- #
def g_analytic(fam, alphas, rng, n=NPROJ):
    x = fam["samp"](rng, n)
    Phi, _ = frac_basis(alphas)
    P = Phi(x); S = fam["score"](x)
    G = (P.T @ P) / len(x)                            # E[φ_i φ_j]
    b = (P.T @ S) / len(x)                            # E[φ_i S]
    k = np.linalg.solve(G + 1e-9 * np.eye(len(alphas)), b)
    proj_norm2 = float(b @ k)                         # ‖proj S‖²
    return proj_norm2 / fam["I"], k


def _pmm_estimates(fam, alphas, k, rng, n=1500, M=2500):
    """M one-step PMM location estimates θ̂ (true θ=0) via a grid root of the Z-equation
    Σ ψ(x_k−θ)=0 (ψ=Σ k_i ρ_{a_i}) — robust to the non-smooth fractional ψ'."""
    Phi, _ = frac_basis(alphas)
    grid = np.linspace(-1.0, 1.0, 41)
    th = np.empty(M)
    for r in range(M):
        x = fam["samp"](rng, n)
        Gv = (Phi(x[:, None] - grid[None, :]) @ k).mean(0)      # mean ψ(x−θ) over the grid
        cross = np.where(np.diff(np.sign(Gv)) != 0)[0]
        if len(cross) == 0:
            th[r] = grid[np.argmin(np.abs(Gv))]
            continue
        j = cross[np.argmin(np.abs(grid[cross]))]               # crossing nearest 0
        t0, t1, g0, g1 = grid[j], grid[j + 1], Gv[j], Gv[j + 1]
        th[r] = t0 - g0 * (t1 - t0) / (g1 - g0)                 # linear interp to the root
    return th


def g_empirical(fam, alphas, k, rng, n=1500, M=2500):
    """ARE = CRB / Var(PMM estimator), should equal g_analytic."""
    th = _pmm_estimates(fam, alphas, k, rng, n=n, M=M)
    return (1.0 / (n * fam["I"])) / np.var(th)


# --------------------------------------------------------------------------- #
# Branch 2 — detection: captured fraction of the true LLR, + DSGE/QDA exactness
# --------------------------------------------------------------------------- #
def true_llr(fam, x, delta):
    return fam["logp"](x - delta) - fam["logp"](x)   # location shift H1: p(x-δ)


def c_captured(fam, delta, m, rng, n=NPROJ):
    """R² of projecting the true LLR onto the order-m basis under p_mix=½(p0+p1)."""
    nh = n // 2
    x0 = fam["samp"](rng, nh); x1 = fam["samp"](rng, nh) + delta
    x = np.concatenate([x0, x1])                     # samples from the mixture
    L = true_llr(fam, x, delta)
    B = detect_basis(x, m)
    beta, *_ = np.linalg.lstsq(B, L, rcond=None)
    resid = L - B @ beta
    return 1.0 - np.var(resid) / np.var(L)


def qda_vs_llr(fam, delta, mom0, mom1):
    """Per-class Gaussian DSGE (2nd-order truncation) -> QDA discriminant; compare to the
    true LLR using the EXACT population moments mom_c=(μ_c, σ_c²).  Theorem A: for Gaussian
    classes this equals the LLR identically; for Laplace it leaves the truncation residual."""
    xt = np.linspace(-5 + delta / 2, 5 + delta / 2, 4000)
    gauss_ll = lambda x, mu, v: -0.5 * (x - mu) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    qda = gauss_ll(xt, mom1[0], mom1[1]) - gauss_ll(xt, mom0[0], mom0[1])
    llr = true_llr(fam, xt, delta)
    return float(np.max(np.abs(qda - llr)))


# --------------------------------------------------------------------------- #
# Sequential form of detection — GSA / change-point: anchor GSA@S=2 = Ghosh = Bayes (A3),
# and the reading identity J(s) = c (T4)
# --------------------------------------------------------------------------- #
def gauss_ll(x, mu, v):
    return -0.5 * (x - mu) ** 2 / v - 0.5 * np.log(2 * np.pi * v)


def gsa_anchor_covshift(rng, v1=2.25):
    """A3: H0=N(0,1) vs H1=N(0,v1) — a pure covariance shift, the §6 anchor case.
    Returns (c_S1, c_S2, mahala_affine_err, sign_agreement, rho_FKY, x, L)."""
    n = NPROJ // 2
    x = np.concatenate([rng.standard_normal(n), np.sqrt(v1) * rng.standard_normal(n)])
    L = gauss_ll(x, 0.0, v1) - gauss_ll(x, 0.0, 1.0)          # true LLR on the mixture

    def proj(m):
        B = detect_basis(x, m)
        beta, *_ = np.linalg.lstsq(B, L, rcond=None)
        fit = B @ beta
        return 1.0 - np.var(L - fit) / np.var(L), fit

    c1, _ = proj(2)                                            # S=1: basis [1, x]
    c2, Lhat = proj(3)                                         # S=2: basis [1, x, x²]

    # Ghosh leg: Λ = ½D₀ − ½D₁ − ½(log v₁ − log v₀)  (Lean gauss_llr_eq_mahala_affine)
    xt = np.linspace(-6.0, 6.0, 4001)
    mahala_affine = 0.5 * xt ** 2 / 1.0 - 0.5 * xt ** 2 / v1 - 0.5 * np.log(v1 / 1.0)
    qerr = float(np.max(np.abs((gauss_ll(xt, 0.0, v1) - gauss_ll(xt, 0.0, 1.0))
                               - mahala_affine)))

    # Bayes leg: the sign of the order-2 statistic is the Bayes decision
    agree = float(np.mean(np.sign(Lhat) == np.sign(L)))

    # GSA normal system F·K=Y over φ=(x, x²), THEORETICAL moments (KuYuPe convention):
    # E0=(0,1), E1=(0,v1); Cov0=diag(1,2), Cov1=diag(v1, 2v1²)  (Gaussian moments).
    F = np.diag([1.0 + v1, 2.0 + 2.0 * v1 ** 2])
    Y = np.array([0.0, v1 - 1.0])
    K = np.linalg.solve(F, Y)
    stat = np.stack([x, x ** 2], axis=-1) @ K
    rho = float(np.corrcoef(stat, L)[0, 1])
    return c1, c2, qerr, agree, rho, x, L


def j_vs_c(x, L, ms=(2, 3, 4, 6)):
    """T4: J(s)/‖z‖² (partial Parseval sum over the QR-orthonormalized centered basis —
    the InfoFunctional.lean object, computed on the sample measure) vs the captured
    fraction c(m) (lstsq R²) on the SAME sample.  Equal in exact arithmetic — the GSA
    information functional IS the detection reading."""
    out = []
    z = L - L.mean()
    for m in ms:
        B = detect_basis(x, m)
        beta, *_ = np.linalg.lstsq(B, L, rcond=None)
        c = 1.0 - np.var(L - B @ beta) / np.var(L)
        Bc = B[:, 1:] - B[:, 1:].mean(axis=0)                  # drop 1, center -> L²₀
        Q, _ = np.linalg.qr(Bc)
        coef = Q.T @ z
        jfrac = float((coef @ coef) / (z @ z))
        out.append((m, float(c), jfrac))
    return out


# --------------------------------------------------------------------------- #
# Branch 3 — DECOMPOSITION: captured fraction R²=κ_m(log p) of the master series itself
# --------------------------------------------------------------------------- #
def r2_decomposition(fam, m, rng, n=NPROJ):
    """The decomposition reading: the order-m captured fraction of the CENTERED log-density
    (the master-series target y=log p), reconstructed by least squares on the basis prefix
    [1,x,x²,|x|,x³,|x|³][:m].  This is the residual-norm (DSGE log-MSED) leg — a THIRD object,
    distinct from the score S (estimation g) and the LLR Λ (detection c).  Terminates (R²=1)
    when the basis spans log p (Gaussian, at x²); rises with the fractional element on Laplace."""
    x = fam["samp"](rng, n)
    y = fam["logp"](x)
    B = detect_basis(x, m)                                # includes the intercept column
    beta, *_ = np.linalg.lstsq(B, y, rcond=None)
    return 1.0 - np.var(y - B @ beta) / np.var(y)


# --------------------------------------------------------------------------- #
# Heavy-tail demonstration: the fractional element earns its keep where Hermite diverges
# --------------------------------------------------------------------------- #
def heavy_tail_g(fam, alphas, rng):
    """Captured-Fisher fraction g for the odd fractional basis on a heavy-tailed law.
    Admissible iff 2·max(alphas) < tail index (else ρ_a ∉ L²)."""
    g, _ = g_analytic(fam, alphas, rng)
    return g


def monomial_gram_blowup(fam, rng, n=NPROJ):
    """Evidence that the POLYNOMIAL PMM basis {x, x³} is not L² on a heavy tail: the Gram entry
    E[x⁴] (needed for x³·x) diverges (df ≤ 4), so its empirical value is enormous and unstable —
    the projection is undefined — whereas the fractional Gram E[|x|^{2a}] (2a<df) is finite."""
    x = fam["samp"](rng, n)
    return float(np.mean(x ** 4))                        # → ∞ in population for df ≤ 4


# --------------------------------------------------------------------------- #
# Local cross-branch identity: c(δ) → g as δ → 0 for a location family (Λ_δ = δ·S + o(δ))
# --------------------------------------------------------------------------- #
def local_identity(fam, deltas, rng):
    """Detection captured fraction c(δ)=κ_2(Λ_δ; p₀) onto the matched centered basis {1,x},
    computed under the null p₀, vs the estimation fraction g=κ_1(S; p₀) onto {x}.  Theorem:
    c(δ) → g as δ → 0 because Λ_δ = δ·S + o(δ) in L²(p₀)."""
    x0 = fam["samp"](rng, NPROJ)                          # null sample (single reference measure p₀)
    S = fam["score"](x0)
    # g = κ₁(S;p₀): project the (centered) score onto centered {x}
    xc = x0 - x0.mean()
    g = float((np.dot(xc, S - S.mean()) ** 2) / (np.dot(xc, xc) * np.dot(S - S.mean(), S - S.mean())))
    B = np.stack([np.ones_like(x0), x0], axis=-1)         # {1, x}
    out = []
    for d in deltas:
        L = fam["logp"](x0 - d) - fam["logp"](x0)         # Λ_δ on the SAME null sample
        beta, *_ = np.linalg.lstsq(B, L, rcond=None)
        c = 1.0 - np.var(L - B @ beta) / np.var(L)
        out.append((d, float(c)))
    return g, out


# --------------------------------------------------------------------------- #
# Operating characteristics, uncertainty, selection, completeness  (revision)
#   U1  uncertainty quantification (mean ± SE, 95% CI) on the headline readings
#   P1  power / ROC of the order-m projected-score test, vs the analytic g (Prop. power)
#   P2  estimator RMSE & empirical efficiency vs order, tracking g     (Prop. power (i))
#   S1  data-driven selection: held-out κ for order m; Hill tail index → exponent ceiling
#   C1  completeness: κ_m(generic z) → 1 as the fractional basis densifies (Thm complete)
# numpy-only; make_figures.py recomputes these for the plots — nothing hardcoded.
# --------------------------------------------------------------------------- #
from math import erf, sqrt

Z95 = 1.6448536269514722        # one-sided standard-normal 0.95 quantile (level 0.05)
_erf = np.vectorize(erf)


def Phi(t):
    """Standard-normal CDF (scalar/array), numpy-only via math.erf."""
    return 0.5 * (1.0 + _erf(np.asarray(t, float) / sqrt(2.0)))


def replicate(fn, R, base_seed, n):
    """Estimand fn(rng, n) over R independent seeds -> (mean, se, lo95, hi95)."""
    v = np.array([fn(np.random.default_rng(base_seed + 7919 * (r + 1)), n) for r in range(R)])
    m = float(v.mean()); se = float(v.std(ddof=1) / sqrt(R))
    return m, se, m - 1.96 * se, m + 1.96 * se


def power_vs_order(fam, theta, nested, rng, n=400, M=2500):
    """Empirical level (calibrated to 0.05 under H0) and power at shift theta of the test
    T=Σ ψ_m(x_i), ψ_m the projected score in V_m.  Returns (m, g, level, power, power_theory)
    with power_theory=Φ(√(gI)·h−z) the Prop.\\ power local-power curve, h=θ√n."""
    out = []
    for al in nested:
        ga, k = g_analytic(fam, al, rng, n=400_000)
        Phf, _ = frac_basis(al)
        T0 = np.array([(Phf(fam["samp"](rng, n)) @ k).sum() for _ in range(M)])
        T1 = np.array([(Phf(fam["samp"](rng, n) + theta) @ k).sum() for _ in range(M)])
        thr = np.quantile(T0, 0.95)
        h = theta * sqrt(n)
        out.append((len(al), float(ga), float(np.mean(T0 > thr)), float(np.mean(T1 > thr)),
                    float(Phi(sqrt(ga * fam["I"]) * h - Z95))))
    return out


def roc_points(fam, theta, al, rng, n=400, M=4000):
    """ROC (fpr, tpr) of the order-m projected-score test by sweeping the threshold."""
    ga, k = g_analytic(fam, al, rng, n=400_000)
    Phf, _ = frac_basis(al)
    T0 = np.array([(Phf(fam["samp"](rng, n)) @ k).sum() for _ in range(M)])
    T1 = np.array([(Phf(fam["samp"](rng, n) + theta) @ k).sum() for _ in range(M)])
    ths = np.quantile(np.concatenate([T0, T1]), np.linspace(0.0, 1.0, 100))
    return (np.array([np.mean(T0 > t) for t in ths]),
            np.array([np.mean(T1 > t) for t in ths]), float(ga))


def rmse_vs_order(fam, nested, rng, n=500, M=2500):
    """PMM one-step estimator RMSE and empirical efficiency CRB/MSE vs order (true θ=0),
    against the analytic captured fraction g.  Returns (m, g, rmse, eff_emp)."""
    out = []
    for al in nested:
        ga, k = g_analytic(fam, al, rng, n=400_000)
        th = _pmm_estimates(fam, al, k, rng, n=n, M=M)
        mse = float(np.mean(th ** 2))
        out.append((len(al), float(ga), sqrt(mse), (1.0 / (n * fam["I"])) / mse))
    return out


def hill_alpha(x, frac=0.1):
    """Hill tail-index estimate from the top fraction of |x|."""
    a = np.sort(np.abs(x)); k = max(50, int(frac * len(a)))
    return float(1.0 / np.mean(np.log(a[-k:] / a[-k - 1])))


def select_order_cv(fam, delta, max_m, rng, n=400_000):
    """Held-out captured fraction of the LLR: fit β on a train split, score κ on a test split,
    per order m.  Returns (list of (m, κ_test), argmax m)."""
    nh = n // 2
    x = np.concatenate([fam["samp"](rng, nh), fam["samp"](rng, nh) + delta])
    L = true_llr(fam, x, delta)
    idx = rng.permutation(len(x)); tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]
    out = []
    for m in range(2, max_m + 1):
        B = detect_basis(x, m)
        beta, *_ = np.linalg.lstsq(B[tr], L[tr], rcond=None)
        out.append((m, float(1.0 - np.var(L[te] - B[te] @ beta) / np.var(L[te]))))
    return out, max(out, key=lambda t: t[1])[0]


def select_exponents(fam, rng, n=400_000):
    """Data-driven exponent ceiling a_max=α̂/2 (Hill), a grid in (0,a_max), and the resulting
    captured-Fisher fraction (admissible by construction).  Returns (α̂, a_max, grid, g)."""
    x = fam["samp"](rng, n)
    alpha_hat = hill_alpha(x); a_max = alpha_hat / 2.0
    grid = list(np.linspace(0.3 * a_max, 0.9 * a_max, 4))
    g, _ = g_analytic(fam, grid, rng, n=n)
    return alpha_hat, a_max, grid, float(g)


def completeness_densify(fam, grids, rng, n=400_000):
    """κ_m of a generic centered smooth target z=x·e^{−x²/2} as the even+odd fractional system
    densifies its exponents in (0, s0).  Confirms Thm complete / Prop complete-light: κ_m(z)→1.
    Returns list of (#exponents, κ)."""
    x = fam["samp"](rng, n)
    z = x * np.exp(-0.5 * x ** 2); z = z - z.mean()
    out = []
    for grid in grids:
        cols = [np.ones_like(x)]
        for a in grid:
            cols.append(np.abs(x) ** a)
            cols.append(np.sign(x) * np.abs(x) ** a)
        B = np.stack(cols, axis=-1)
        beta, *_ = np.linalg.lstsq(B, z, rcond=None)
        out.append((len(grid), float(1.0 - np.var(z - B @ beta) / np.var(z))))
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    GAU, LAP = gaussian(), laplace()
    delta = 1.0
    print("=" * 74)
    print("KUNCHENKO LIKELIHOOD-SERIES UNIFICATION")
    print("  three branches  PMM(∂) · detection(Δ) · DSGE(residual)  + sequential form  GSA(Δ∘time)")
    print(f"  seed {SEED}, MC={NPROJ:,}, location shift δ={delta}")
    print("=" * 74)

    # ----- ANCHOR: Gaussian -----
    print("\n[ANCHOR — Gaussian: basis spans the truth, the series terminates]")
    gA1, _ = g_analytic(GAU, [1.0], rng)                       # score=x ∈ span{x}
    print(f"  A1 estimation: g(order-1 {{x}})         = {gA1:.6f}   (score linear -> full Fisher)")
    cA = c_captured(GAU, delta, 2, rng)                        # basis [1,x]
    qerr = qda_vs_llr(GAU, delta, (0.0, 1.0), (delta, 1.0))    # exact N(0,1) vs N(δ,1) moments
    print(f"  A2 detection : c(order [1,x])           = {cA:.6f}   |QDA-disc − true LLR|max = {qerr:.2e}")
    a1 = gA1 >= 0.999
    a2 = (cA >= 0.999) and (qerr < 1e-9)
    print(f"     A1 {'PASS' if a1 else 'FAIL'} · A2 {'PASS' if a2 else 'FAIL'}")

    # ----- TRUNCATION: Laplace -----
    print("\n[TRUNCATION — Laplace: basis approximates, residual R_m shrinks with order m]")
    nested = [[1.0], [0.5, 1.0], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5, 2.5]]
    print("  T1/T2 estimation (fractional basis sgn(x)|x|^a):")
    print(f"     {'basis α':<28}{'g_analytic':>12}{'g_empirical(ARE)':>18}")
    g_an_list, g_emp_full = [], None
    for al in nested:
        ga, k = g_analytic(LAP, al, rng)
        ge = g_empirical(LAP, al, k, rng)
        g_an_list.append(ga); g_emp_full = (ge, ga)
        print(f"     {str(al):<28}{ga:>12.4f}{ge:>18.4f}")
    t1 = all(g_an_list[i + 1] >= g_an_list[i] - 1e-6 for i in range(len(g_an_list) - 1)) \
        and (g_an_list[-1] > g_an_list[0] + 0.02)
    t2 = abs(g_emp_full[0] - g_emp_full[1]) < 0.05
    print(f"     T1 (g rises with order) {'PASS' if t1 else 'FAIL'}   "
          f"· T2 (|g_an−g_emp|={abs(g_emp_full[0]-g_emp_full[1]):.3f}<0.05) {'PASS' if t2 else 'FAIL'}")

    print("\n  T3 detection (captured LLR fraction vs basis order):")
    c_list = [c_captured(LAP, delta, m, rng) for m in (2, 3, 4, 6)]
    for m, c in zip((2, 3, 4, 6), c_list):
        print(f"     c(order {m})  = {c:.4f}")
    t3 = all(c_list[i + 1] >= c_list[i] - 1e-3 for i in range(len(c_list) - 1)) \
        and (c_list[-1] > c_list[0] + 0.05)
    print(f"     T3 (c rises with order) {'PASS' if t3 else 'FAIL'}")

    # ----- A3: GSA anchor — Gaussian covariance shift (sequential form of detection) -----
    print("\n[A3 — GSA anchor: H0=N(0,1) vs H1=N(0,2.25), pure covariance shift]")
    c1, c2, qerr3, agree, rho, xg, Lg = gsa_anchor_covshift(rng)
    print(f"  c(order [1,x])    (S=1)   = {c1:.6f}   (designed-blind: even signal ⊥ odd basis)")
    print(f"  c(order [1,x,x²]) (S=2)   = {c2:.6f}   (LLR quadratic -> projection exact)")
    print(f"  |mahala-affine − LLR|max  = {qerr3:.2e}  (Ghosh head ≡ LLR, Lean gauss_llr_eq_mahala_affine)")
    print(f"  sign(GSA@S=2) = Bayes     : agreement {agree:.6f}")
    print(f"  F·K=Y (theor. moments)    : corr(K'φ, LLR) = {rho:.9f}")
    a3 = (c2 >= 0.999) and (qerr3 < 1e-9) and (agree >= 0.9999) and (c1 < 0.05) \
        and (rho >= 1 - 1e-6)
    print(f"     A3 {'PASS' if a3 else 'FAIL'}  "
          f"(S=2 exact ∧ Ghosh≡LLR ∧ sign=Bayes ∧ S=1 blind ∧ F·K=Y collinear)")

    # ----- T4: reading identity J(s) = c for the GSA branch -----
    print("\n[T4 — GSA reading: J(s)/‖z‖² (InfoFunctional) vs captured c(m), same sample]")
    rows_g = j_vs_c(xg, Lg)
    x_lap = np.concatenate([LAP["samp"](rng, NPROJ // 2),
                            LAP["samp"](rng, NPROJ // 2) + delta])
    rows_l = j_vs_c(x_lap, true_llr(LAP, x_lap, delta))
    print(f"     {'family':<22}{'m':>3}{'c(m)':>10}{'J/‖z‖²':>10}{'|Δ|':>11}")
    dmax = 0.0
    for fam_name, rows in (("Gaussian cov-shift", rows_g), ("Laplace shift", rows_l)):
        for m, c, j in rows:
            dmax = max(dmax, abs(c - j))
            print(f"     {fam_name:<22}{m:>3}{c:>10.6f}{j:>10.6f}{abs(c - j):>11.2e}")
    j_sat = rows_g[1][2]                                       # Gaussian, m=3
    t4 = (dmax < 1e-6) and (j_sat >= 0.999)
    print(f"     T4 (max|c−J|={dmax:.2e}<1e-6 ∧ Gaussian J saturates at m=3: {j_sat:.6f}) "
          f"{'PASS' if t4 else 'FAIL'}")

    # ----- D1: DECOMPOSITION leg — R²=κ_m(log p) terminates (Gaussian) and rises (Laplace) -----
    print("\n[D1 — decomposition: R²=κ_m(log p), captured fraction of the log-density itself]")
    rg2, rg3 = r2_decomposition(GAU, 2, rng), r2_decomposition(GAU, 3, rng)
    rl = [r2_decomposition(LAP, m, rng) for m in (2, 3, 4, 6)]
    print(f"  Gaussian R²(m=2 [1,x]) = {rg2:.6f}   R²(m=3 [1,x,x²]) = {rg3:.6f}  (terminates at x²)")
    for m, r in zip((2, 3, 4, 6), rl):
        print(f"  Laplace  R²(m={m})            = {r:.4f}")
    d1 = (rg3 >= 0.999) and (rg2 < 0.999) and (rl[-1] > rl[0] + 0.02) \
        and all(rl[i + 1] >= rl[i] - 1e-3 for i in range(len(rl) - 1))
    print(f"     D1 {'PASS' if d1 else 'FAIL'}  (Gaussian terminates at x²; Laplace R² rises with the fractional basis)")

    # ----- H1: HEAVY TAIL — fractional element captures Fisher where the monomial Gram diverges -----
    print("\n[H1 — heavy tail: fractional g rises where the polynomial/Hermite basis is not L²]")
    T3, CAU = student_t(3.0), cauchy()
    t3_nested = [[0.5], [0.5, 1.0], [0.5, 1.0, 1.25]]     # 2·max a = 2.5 < df=3  (admissible)
    cau_nested = [[0.25], [0.25, 0.4]]                    # 2·max a = 0.8 < df=1  (admissible)
    gt3 = [heavy_tail_g(T3, al, rng) for al in t3_nested]
    gca = [heavy_tail_g(CAU, al, rng) for al in cau_nested]
    ex4 = monomial_gram_blowup(T3, rng)
    for al, g in zip(t3_nested, gt3):
        print(f"  Student-t(3)  fractional {str(al):<22} g = {g:.4f}")
    print(f"     monomial {{x,x³}} needs E[x⁴]: empirical = {ex4:.1f} (df=3 -> diverges in population) -> not L²")
    for al, g in zip(cau_nested, gca):
        print(f"  Cauchy        fractional {str(al):<22} g = {g:.4f}")
    h1 = (gt3[-1] > gt3[0] + 0.05) and all(gt3[i + 1] >= gt3[i] - 1e-3 for i in range(len(gt3) - 1)) \
        and (gca[-1] > gca[0] - 1e-3) and (ex4 > 50.0)
    print(f"     H1 {'PASS' if h1 else 'FAIL'}  (fractional g rises on t₃; Cauchy captured; monomial E[x⁴] diverges)")

    # ----- L1: LOCAL identity — c(δ) → g as δ → 0 on a location family -----
    print("\n[L1 — local identity: c(δ) → g as δ → 0 (Λ_δ = δ·S + o(δ)), Laplace]")
    gloc, seq = local_identity(LAP, (0.5, 0.2, 0.1, 0.05, 0.02), rng)
    print(f"  g = κ₁(S;p₀) onto {{x}}      = {gloc:.4f}")
    for d, c in seq:
        print(f"  c(δ={d:<5}) = κ₂(Λ_δ;p₀)  = {c:.4f}   |c−g| = {abs(c - gloc):.4f}")
    l1 = abs(seq[-1][1] - gloc) < 0.02 and seq[0][1] >= seq[-1][1] - 1e-3
    print(f"     L1 {'PASS' if l1 else 'FAIL'}  (c(δ) decreases to g as δ→0)")

    # ----- U: unification co-movement -----
    print("\n[U — one reading] Gaussian: all branches complete (g=c=1; GSA J saturates).")
    print(f"   Laplace: estimation g {g_an_list[0]:.3f}→{g_an_list[-1]:.3f} and detection "
          f"c {c_list[0]:.3f}→{c_list[-1]:.3f} both rise with truncation order -> same series.")
    print("   GSA: J(s)/‖z‖² ≡ c(m) on the same sample -> the change-point branch spends")
    print("   the same reading; CUSUM on GSA@S=2 increments = CUSUM on the true LLR (A3).")

    # ----- U1: UNCERTAINTY — headline readings as mean ± SE with 95% CI over R seeds -----
    print("\n[U1 — uncertainty: headline readings as mean ± SE over R independent seeds]")
    Rrep, nrep = 30, 200_000
    gm, gse, glo, ghi = replicate(lambda r, n: g_analytic(LAP, [0.5, 1.0, 1.5, 2.5], r, n)[0], Rrep, SEED, nrep)
    cm, cse, clo, chi = replicate(lambda r, n: c_captured(LAP, delta, 6, r, n), Rrep, SEED, nrep)
    rm2, rse, rlo, rhi = replicate(lambda r, n: r2_decomposition(LAP, 6, r, n), Rrep, SEED, nrep)
    print(f"  Laplace  g(order-4 frac) = {gm:.4f} ± {gse:.4f}   95% CI [{glo:.4f}, {ghi:.4f}]")
    print(f"  Laplace  c(order 6)      = {cm:.4f} ± {cse:.4f}   95% CI [{clo:.4f}, {chi:.4f}]")
    print(f"  Laplace  R^2(order 6)    = {rm2:.4f} ± {rse:.4f}   95% CI [{rlo:.4f}, {rhi:.4f}]")
    print(f"     (R={Rrep} independent seeds, n={nrep:,} each)")
    u1 = all(0.0 <= se < 0.05 for se in (gse, cse, rse)) and min(glo, clo, rlo) > 0.75
    print(f"     U1 {'PASS' if u1 else 'FAIL'}  (finite MC standard error; 95% CIs reported)")

    # ----- P1: POWER — order-m projected-score test vs the analytic g (Prop. power) -----
    print("\n[P1 — power: order-m projected-score test, level calibrated to 0.05 under H0]")
    print(f"     {'family':<14}{'m':>3}{'g':>8}{'level':>8}{'power':>8}{'power(thy)':>12}")
    pw_lap = power_vs_order(LAP, 0.09, nested, rng, n=400, M=2000)
    for m, ga, lv, pw, pth in pw_lap:
        print(f"     {'Laplace':<14}{m:>3}{ga:>8.3f}{lv:>8.3f}{pw:>8.3f}{pth:>12.3f}")
    pw_t3 = power_vs_order(T3, 0.16, t3_nested, rng, n=400, M=2000)
    for m, ga, lv, pw, pth in pw_t3:
        print(f"     {'Student-t(3)':<14}{m:>3}{ga:>8.3f}{lv:>8.3f}{pw:>8.3f}{pth:>12.3f}")
    p1 = (pw_lap[-1][3] > pw_lap[0][3] + 0.03) \
        and all(abs(r[2] - 0.05) < 0.03 for r in pw_lap) \
        and all(abs(r[3] - r[4]) < 0.08 for r in pw_lap)
    print(f"     P1 {'PASS' if p1 else 'FAIL'}  (power rises with order, ~tracks Φ(√(gI)h−z), level≈0.05)")

    # ----- P2: estimator RMSE & empirical efficiency vs order (tracks g) -----
    print("\n[P2 — estimation: PMM one-step RMSE & empirical efficiency vs order (true θ=0)]")
    print(f"     {'family':<14}{'m':>3}{'g(analytic)':>13}{'RMSE':>10}{'eff(emp)':>10}")
    rm_lap = rmse_vs_order(LAP, nested, rng, n=500, M=2000)
    for m, ga, rmse, eff in rm_lap:
        print(f"     {'Laplace':<14}{m:>3}{ga:>13.3f}{rmse:>10.4f}{eff:>10.3f}")
    p2 = (rm_lap[-1][3] > rm_lap[0][3] + 0.02) and (rm_lap[-1][2] < rm_lap[0][2]) \
        and all(abs(r[3] - r[1]) < 0.08 for r in rm_lap)
    print(f"     P2 {'PASS' if p2 else 'FAIL'}  (efficiency rises with order and tracks g; RMSE falls)")

    # ----- S1: data-driven selection of order m and exponents (Q2) -----
    print("\n[S1 — selection: held-out κ for the order m; Hill tail index → exponent ceiling]")
    cv, m_star = select_order_cv(LAP, delta, 6, rng)
    print("  held-out κ(LLR) by order (Laplace):  " + "  ".join(f"m{m}:{k:.3f}" for m, k in cv))
    print(f"     selected order m* = {m_star}  (max held-out captured fraction)")
    ah, amax, grid, gsel = select_exponents(T3, rng)
    print(f"  Student-t(3): Hill α̂ = {ah:.2f} → ceiling a_max = α̂/2 = {amax:.2f}")
    print(f"     grid {[round(float(a), 2) for a in grid]} admissible (2·max a = {2 * max(grid):.2f} < α̂); g = {gsel:.3f}")
    s1 = (m_star >= 3) and (1.5 <= ah <= 6.0) and (2 * max(grid) < ah) and (gsel > 0.75)
    print(f"     S1 {'PASS' if s1 else 'FAIL'}  (CV picks a sensible order; data-driven ceiling admissible)")

    # ----- C1: completeness — κ_m(generic z) → 1 as the fractional basis densifies -----
    print("\n[C1 — completeness: κ_m(generic z=x·e^{−x²/2}) as the fractional basis densifies]")
    lap_grids = [list(np.linspace(0.4, 2.4, j)) for j in (1, 2, 4, 8)]
    t3_grids = [list(np.linspace(0.25, 1.4, j)) for j in (1, 2, 4, 8)]    # all < s0 = df/2 = 1.5
    cl = completeness_densify(LAP, lap_grids, rng)
    ct = completeness_densify(T3, t3_grids, rng)
    print("  Laplace (light tail):          " + "  ".join(f"{q}exp:{k:.3f}" for q, k in cl))
    print("  Student-t(3) (heavy, exps<1.5): " + "  ".join(f"{q}exp:{k:.3f}" for q, k in ct))
    c1 = (cl[-1][1] > 0.9) and (ct[-1][1] > 0.9) \
        and (cl[-1][1] > cl[0][1] + 0.05) and (ct[-1][1] > ct[0][1] + 0.05)
    print(f"     C1 {'PASS' if c1 else 'FAIL'}  (κ→1 as exponents densify, both light & heavy tails)")

    ok = a1 and a2 and t1 and t2 and t3 and a3 and t4 and d1 and h1 and l1 \
        and u1 and p1 and p2 and s1 and c1
    print("\n" + "=" * 74)
    print(f"UNIFICATION VERIFICATION : {'PASS' if ok else 'FAIL'}  (15 checks)")
    print(f"  A1 {'✓' if a1 else '✗'}  A2 {'✓' if a2 else '✗'}  T1 {'✓' if t1 else '✗'}  "
          f"T2 {'✓' if t2 else '✗'}  T3 {'✓' if t3 else '✗'}  A3 {'✓' if a3 else '✗'}  "
          f"T4 {'✓' if t4 else '✗'}  D1 {'✓' if d1 else '✗'}  H1 {'✓' if h1 else '✗'}  "
          f"L1 {'✓' if l1 else '✗'}")
    print(f"  U1 {'✓' if u1 else '✗'}  P1 {'✓' if p1 else '✗'}  P2 {'✓' if p2 else '✗'}  "
          f"S1 {'✓' if s1 else '✗'}  C1 {'✓' if c1 else '✗'}")
    if ok:
        print("  ONE functional κ_m (captured fraction) read on FOUR objects: score S (PMM g),")
        print("  LLR Λ (detection c; sequential GSA reads J(s)/‖z‖²), and log-density (DSGE R²).")
        print("  These are NOT one number: each is κ_m on a different (object, measure, basis).")
        print("  They reach 1 SIMULTANEOUSLY only on the terminating anchor (Gaussian; GSA@S=2 =")
        print("  Ghosh = Bayes); off it all are <1 and rise with truncation order (Laplace, t₃,")
        print("  Cauchy — where the fractional element stays L² and the Hermite/monomial basis")
        print("  does not). The genuine cross-branch EQUALITY is the local limit c(δ)→g (L1).")
    print("=" * 74)


if __name__ == "__main__":
    main()
