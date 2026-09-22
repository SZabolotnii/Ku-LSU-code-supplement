"""Closed-form asymptotic efficiencies of the robust arms against the Student-t MLE.

Used by Section 8.2 ("Reading") to explain why the fitted-t law predicted the fractional
projection ahead of the Huber estimator (kappa = 0.910 against 0.83-0.85 for the robust arms)
while the realised order is the reverse: within one epoch the tail is milder than in the
pooled law, and at the per-epoch tail (median Hill 3.3-3.6) the robust arms are already
efficient to within a few percent.

No data is read.  Each value is a deterministic quadrature of the textbook formulas
    median     4 f(0)^2 / I
    Huber      (E psi')^2 / (E psi^2) / I,  psi = clip(x, -k, k), k = 1.345 * MAD/0.6745
    Wilcoxon   12 (int f^2)^2 / I
with I = (nu + 1) / (nu + 3) the Fisher information of the standard t_nu location family.
The Huber constant is applied on the MAD scale, as in run_realdata_study2.py.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats

NUS = (1.855, 3.0, 3.3, 3.56, 5.0)
HUBER_K = 1.345


def efficiencies(nu: float) -> dict[str, float]:
    f = lambda x: stats.t.pdf(x, nu)
    fisher = (nu + 1.0) / (nu + 3.0)
    f2 = integrate.quad(lambda x: f(x) ** 2, -np.inf, np.inf)[0]
    scale = stats.t.ppf(0.75, nu) / 0.6745
    k = HUBER_K * scale
    e_psi2 = integrate.quad(lambda x: min(max(x, -k), k) ** 2 * f(x), -np.inf, np.inf)[0]
    e_dpsi = stats.t.cdf(k, nu) - stats.t.cdf(-k, nu)
    return {
        "median": 4.0 * f(0.0) ** 2 / fisher,
        "Huber(1.345, MAD scale)": (e_dpsi ** 2 / e_psi2) / fisher,
        "Wilcoxon": 12.0 * f2 ** 2 / fisher,
    }


def main() -> None:
    print("=" * 78)
    print("Reference efficiencies of the robust arms against the t_nu MLE (closed form)")
    print("nu = 1.855 is the law fitted on TRAIN in Study 2; 3.3 and 3.56 bracket the")
    print("per-epoch median Hill estimates (TEST 3.29, TRAIN 3.56).")
    print("=" * 78)
    for nu in NUS:
        e = efficiencies(nu)
        row = "  ".join(f"{k} {v:.3f}" for k, v in e.items())
        print(f"nu = {nu:<6} {row}")
    print("=" * 78)


if __name__ == "__main__":
    main()
