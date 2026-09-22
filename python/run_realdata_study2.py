"""
run_realdata_study2.py -- Study 2 of the real-data section: a real noise law in the
second-moment-failure regime.

Pre-registration: REAL_DATA_SPEC.md, section "Study 2", committed before this file existed.
Data: EEGdenoiseNet muscular-artefact epochs (data/EMG_all_epochs.npy, CC0), chosen by the
outcome-blind selection rule recorded there.

Design in one paragraph.  Block = epoch.  Inside each epoch the pair differences
d_t = x_{2t} - x_{2t-1} have location 0 without any centering, so an injected shift delta is
a genuine estimand and block RMSE about it is estimator risk -- the identification the FX
study could not achieve.  Each block is divided by its own MAD (shift-invariant), so all arms
see the same standardised block.  Epochs in file order: first 60 % TRAIN (exponents, score
coefficients, fitted t law, critical values), last 40 % TEST (every reported metric).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import integrate, optimize, stats

from run_comparators import _auc, _design_frac, _design_poly, _hermite_cols, _psi_root, huber_location, hill_alpha
from run_realdata import fit_hermite_coefs, fit_score_coefs, fmt_exponents
from run_three_branch_unification import SEED, frac_basis

TRAIN_FRAC = 0.60
NBOOT = 2000
ALPHA = 0.05
DELTAS = (0.25, 1.0)          # primary, secondary (MAD units)
ROOT_LO, ROOT_HI = -3.0, 3.0  # search interval of the projected estimating equation


def resolve_data():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "data", "EMG_all_epochs.npy")
    if not os.path.exists(p):
        sys.exit(f"dataset not found: {p}")
    return p


def pair_differences(epochs):
    """d_t = x_{2t} - x_{2t-1} inside every epoch; (n_epochs, 256)."""
    e = np.asarray(epochs, float)
    return e[:, 1::2] - e[:, 0::2]


def standardise_blocks(D):
    """Divide every block by its own normal-consistent MAD about the block median.
    Shift-invariant, hence computed before any injection without loss of blindness."""
    mad = stats.median_abs_deviation(D, axis=1, scale="normal", keepdims=True)
    return D / np.maximum(mad, 1e-12)


def split_epochs(n):
    train = np.arange(n) < int(TRAIN_FRAC * n)
    return train


# --------------------------------------------------------------------------- #
def fit_t_law(z):
    """Student-t (df, scale) fitted once on standardised TRAIN; loc fixed at 0 by the pairing."""
    df, loc, sc = stats.t.fit(z, floc=0.0)
    return float(df), float(sc)


def t_mle_location(b, df, sc):
    f = lambda th: -np.sum(stats.t.logpdf((b - th) / sc, df))
    return float(optimize.minimize_scalar(f, bounds=(ROOT_LO, ROOT_HI), method="bounded").x)


def kappa_under_t(alphas, df, sc):
    """Captured Fisher fraction of the odd dictionary {sgn|x|^a} (or of sgn x for alphas=[0])
    against the score of t(df, scale=sc), by quadrature. kappa = b' F^{-1} b / I."""
    pdf = lambda x: stats.t.pdf(x / sc, df) / sc
    score = lambda x: (df + 1) * (x / sc**2) / (df + (x / sc) ** 2)   # -d/dx log pdf
    def rho(x, a):
        return np.sign(x) * np.abs(x) ** a if a > 0 else np.sign(x)
    m = len(alphas)
    F = np.zeros((m, m)); b = np.zeros(m)
    q = lambda f: 2 * integrate.quad(f, 0, np.inf, limit=400)[0]     # even integrands, half-line
    for i, a in enumerate(alphas):
        b[i] = q(lambda x: rho(x, a) * score(x) * pdf(x))
        for j, c in enumerate(alphas):
            if j < i:
                F[i, j] = F[j, i]; continue
            F[i, j] = q(lambda x: rho(x, a) * rho(x, c) * pdf(x))
    I = q(lambda x: score(x) ** 2 * pdf(x))
    return float(b @ np.linalg.solve(F, b) / I)


# --------------------------------------------------------------------------- #
def branch_estimation(ztr, Zte, alphas, delta, tlaw, rng):
    Phi_f, _ = frac_basis(alphas)
    Phi_p, _ = frac_basis([1.0, 3.0])
    k_f, cond_f = fit_score_coefs(ztr, alphas, scale=1.0)
    k_p, cond_p = fit_score_coefs(ztr, [1.0, 3.0], scale=1.0)
    k_h, cond_h = fit_hermite_coefs(ztr, scale=1.0)
    df, sc = tlaw
    arms = {
        "sample mean": lambda b: float(b.mean()),
        "median": lambda b: float(np.median(b)),
        "Huber(1.345)": lambda b: huber_location(b),
        "Student-t MLE (df,scale from TRAIN)": lambda b: t_mle_location(b, df, sc),
        "polynomial PMM {x,x^3}": lambda b: _psi_root(b, Phi_p, k_p, lo=ROOT_LO, hi=ROOT_HI),
        "Hermite-projected score": lambda b: _psi_root(b, lambda u: _hermite_cols(u), k_h, lo=ROOT_LO, hi=ROOT_HI),
        "fractional PMM": lambda b: _psi_root(b, Phi_f, k_f, lo=ROOT_LO, hi=ROOT_HI),
    }
    err = {nm: np.array([f(b + delta) - delta for b in Zte]) for nm, f in arms.items()}
    rmse = {nm: float(np.sqrt(np.mean(v ** 2))) for nm, v in err.items()}
    # paired epoch bootstrap of RMSE differences against the fractional arm
    n = len(Zte); ci = {}
    for nm in err:
        if nm == "fractional PMM":
            continue
        d = np.empty(NBOOT)
        for i in range(NBOOT):
            idx = rng.integers(0, n, n)
            d[i] = np.sqrt(np.mean(err[nm][idx] ** 2)) - np.sqrt(np.mean(err["fractional PMM"][idx] ** 2))
        ci[nm] = (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
    return dict(rmse=rmse, ci=ci, cond=dict(frac=cond_f, poly=cond_p, herm=cond_h), nblocks=n)


def branch_testing(ztr, Ztr, Zte, alphas, delta):
    Phi_f, _ = frac_basis(alphas)
    k_f, _ = fit_score_coefs(ztr, alphas, scale=1.0)
    k_h, _ = fit_hermite_coefs(ztr, scale=1.0)

    def stat(b):
        return {
            "t-test": float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))),
            "sign": float(np.mean(np.sign(b))),
            "Wilcoxon": float(stats.wilcoxon(b, alternative="greater", zero_method="zsplit").statistic),
            "projected score (Hermite)": float((_hermite_cols(b) @ k_h).mean()),
            "projected score (fractional)": float((Phi_f(b) @ k_f).mean()),
        }

    names = list(stat(Zte[0]).keys())
    s_tr = {nm: np.array([stat(b)[nm] for b in Ztr]) for nm in names}
    crit = {nm: float(np.quantile(s_tr[nm], 1 - ALPHA)) for nm in names}
    rej0 = {nm: np.array([float(stat(b)[nm] > crit[nm]) for b in Zte]) for nm in names}
    rej1 = {nm: np.array([float(stat(b + delta)[nm] > crit[nm]) for b in Zte]) for nm in names}
    size = {nm: float(rej0[nm].mean()) for nm in names}
    power = {nm: float(rej1[nm].mean()) for nm in names}
    classical = ["t-test", "sign", "Wilcoxon"]
    best = max(classical, key=lambda nm: power[nm])
    diff = rej1["projected score (fractional)"] - rej1[best]
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    z = float(diff.mean() / se) if se > 0 else 0.0
    verdict = "WIN" if z >= 2 else ("LOSS" if z <= -2 else "TIE")
    return dict(size=size, power=power, best=best, gap=float(diff.mean()), se=se, z=z, verdict=verdict)


def branch_classification(ztr, Zte, alphas, delta, rng):
    xtr = np.concatenate([ztr, ztr + delta])
    ytr = np.concatenate([np.zeros(len(ztr)), np.ones(len(ztr))])
    n_ep, n_in = Zte.shape
    xte = np.concatenate([Zte.ravel(), Zte.ravel() + delta])
    yte = np.concatenate([np.zeros(Zte.size), np.ones(Zte.size)])
    ep = np.concatenate([np.repeat(np.arange(n_ep), n_in)] * 2)     # epoch id of every row
    out = {}
    for nm, (Dtr, Dte) in {
        "label regression (fractional)": (_design_frac(xtr, alphas), _design_frac(xte, alphas)),
        "label regression (polynomial)": (_design_poly(xtr, 3), _design_poly(xte, 3)),
    }.items():
        coef, *_ = np.linalg.lstsq(Dtr, ytr - 0.5, rcond=None)
        out[nm] = Dte @ coef
    auc = {nm: _auc(s, yte) for nm, s in out.items()}
    order = np.argsort(ep, kind="stable"); cuts = np.r_[0, np.cumsum(np.bincount(ep))]
    diffs = np.empty(NBOOT)
    for i in range(NBOOT):
        draw = rng.integers(0, n_ep, n_ep)
        idx = np.concatenate([order[cuts[j]:cuts[j + 1]] for j in draw])
        diffs[i] = _auc(out["label regression (fractional)"][idx], yte[idx]) - _auc(out["label regression (polynomial)"][idx], yte[idx])
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    gap = auc["label regression (fractional)"] - auc["label regression (polynomial)"]
    return dict(auc=auc, gap=gap, ci=(lo, hi), c1=(gap >= 0) and (lo > 0))


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    epochs = np.load(resolve_data())
    D = pair_differences(epochs)
    train = split_epochs(len(D))
    Z = standardise_blocks(D)
    Ztr, Zte = Z[train], Z[~train]
    ztr = Ztr.ravel()
    print("=" * 78)
    print("REAL-DATA STUDY 2 -- pre-registered 2026-09-22; seed =", SEED)
    print("Specification: REAL_DATA_SPEC.md, section 'Study 2'")
    print(f"EMG epochs {len(D)} x {D.shape[1]} pair differences; TRAIN {train.sum()} / TEST {(~train).sum()} epochs")
    print(f"TRAIN sign balance P(d>0) = {np.mean(ztr > 0):.4f}; within-block lag-1 acf of d = {np.mean([np.corrcoef(b[:-1], b[1:])[0,1] for b in Ztr]):+.3f}")
    ah = hill_alpha(ztr)
    ceil = ah / 2
    alphas = list(np.linspace(0.35 * ceil, 0.92 * ceil, 3))
    per_block = np.median([hill_alpha(b) for b in Ztr])
    print(f"R-0: TRAIN Hill (pooled standardised) {ah:.3f}; per-block median {per_block:.3f}; ceiling {ceil:.3f}; A={fmt_exponents(alphas)}")
    if ah >= 2:
        print("Study uninformative under the pre-registered Hill gate (alpha_hat >= 2).")
        return
    # Diagnostic only (never used by any arm): how the TEST tail compares with TRAIN.
    print(f"Tail diagnostic: TEST pooled standardised Hill {hill_alpha(Zte.ravel()):.3f}; per-block median {np.median([hill_alpha(b) for b in Zte]):.3f}")
    half = len(Ztr) // 2
    m6a, m6b = float(np.mean(Ztr[:half] ** 6)), float(np.mean(Ztr[half:] ** 6))
    print(f"E-3 diagnostic: TRAIN mean(d^6) first half {m6a:.3g}, second half {m6b:.3g}, ratio {max(m6a, m6b)/min(m6a, m6b):.2f}")
    tlaw = fit_t_law(ztr)
    print(f"Fitted Student-t on TRAIN: df {tlaw[0]:.3f}, scale {tlaw[1]:.3f}")

    est = {}
    for delta in DELTAS:
        tag = "primary" if delta == DELTAS[0] else "secondary"
        print(f"\n[E] block RMSE about the known shift delta={delta} ({tag}); {(~train).sum()} TEST blocks")
        e = branch_estimation(ztr, Zte, alphas, delta, tlaw, rng); est[delta] = e
        best_rob = min(e["rmse"][k] for k in ("median", "Huber(1.345)", "Student-t MLE (df,scale from TRAIN)"))
        for nm, v in e["rmse"].items():
            ci = "" if nm not in e["ci"] else f"; RMSE minus fractional CI [{e['ci'][nm][0]:+.4f}, {e['ci'][nm][1]:+.4f}]"
            print(f"  {nm:<36} RMSE {v:.4f}; ratio to fractional {v / e['rmse']['fractional PMM']:.3f}{ci}")
        print(f"  cond(F): fractional {e['cond']['frac']:.3g}, polynomial {e['cond']['poly']:.3g}, Hermite {e['cond']['herm']:.3g}")
        e1 = e["rmse"]["fractional PMM"] < e["rmse"]["polynomial PMM {x,x^3}"] and e["ci"]["polynomial PMM {x,x^3}"][0] > 0
        e2 = e["rmse"]["fractional PMM"] <= 1.10 * best_rob
        print(f"  E-1 (fractional < polynomial, paired CI excl. 0): {'PASS' if e1 else 'FAIL'}")
        print(f"  E-2 (fractional <= 1.10 x best robust arm {best_rob:.4f}): {'PASS' if e2 else 'FAIL'}")
        print(f"  E-3 ratio mean-RMSE / fractional-RMSE: {e['rmse']['sample mean'] / e['rmse']['fractional PMM']:.2f}")

    print(f"\n[T] size 0.05 calibrated on TRAIN blocks; H1 shift delta={DELTAS[0]}")
    t = branch_testing(ztr, Ztr, Zte, alphas, DELTAS[0])
    for nm in t["power"]:
        print(f"  {nm:<32} TEST size {t['size'][nm]:.4f}; power {t['power'][nm]:.4f}")
    print(f"  T-1: fractional minus best classical ({t['best']}) = {t['gap']:+.4f}, paired SE {t['se']:.4f}, z = {t['z']:+.2f}: {t['verdict']}")

    print(f"\n[C-loc] centered-label regression, shift delta={DELTAS[0]}; paired epoch bootstrap")
    c = branch_classification(ztr, Zte, alphas, DELTAS[0], rng)
    for nm, v in c["auc"].items():
        print(f"  {nm:<34} AUC {v:.4f}")
    print(f"  paired AUC gap {c['gap']:+.4f}; epoch CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
    print(f"  C-1 (fractional >= polynomial, CI excl. 0): {'PASS' if c['c1'] else 'FAIL'}")

    print("\n[kappa] prediction under the fitted t law vs realisation at delta=0.25 (numbers, not a criterion)")
    kf = kappa_under_t(alphas, *tlaw); ks = kappa_under_t([0.0], *tlaw)
    pred = kf / ks
    e = est[DELTAS[0]]
    real = (e["rmse"]["median"] / e["rmse"]["fractional PMM"]) ** 2
    print(f"  kappa(fractional) {kf:.4f}; kappa(sign) {ks:.4f}; predicted efficiency ratio fractional/median {pred:.3f}")
    print(f"  realised TEST RMSE^2(median)/RMSE^2(fractional) {real:.3f}; gap (realised - predicted) {real - pred:+.3f}")
    print("  The fitted law is a proxy for an unknown score and blocks are dependent; the gap is reported, not tested.")
    print("=" * 78)


if __name__ == "__main__":
    main()
