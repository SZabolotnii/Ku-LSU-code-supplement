"""Publication figures for Paper 4 (three-branch likelihood-series unification).

Self-contained, seed 2026.  Imports the verifying gate module
(run_three_branch_unification.py) as the SINGLE SOURCE OF TRUTH and RECOMPUTES
every number plotted — nothing is hardcoded.  Three figures are produced into
verification/figs/ (each as .png and .pdf):

  fig1_reading_comovement       — Laplace family: estimation g(m) and detection c(m)
                                  rise together vs. basis size; Gaussian anchor (g=c=1).
  fig2_schematic                — one log-likelihood series feeding three functionals.
  fig3_operating_characteristics— the operating-characteristic & completeness panel:
                                  (a) power vs order (empirical vs Φ(√(gI)h−z)),
                                  (b) ROC of the order-m projected-score test,
                                  (c) estimator efficiency/RMSE vs order tracking g,
                                  (d) completeness: κ_m(generic z)→1 as exponents densify.

Run:  python make_figures.py   (uses smaller Monte-Carlo than the gate; a few minutes)
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / no display
# Embed real Type 1/TrueType outlines rather than the Type 3 bitmap fonts matplotlib
# writes by default: Springer's figure guidance requires embedded fonts and many
# portals reject Type 3 outright.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- import the verifying gate as the single source of truth ----------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE_DIR = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)   # the gate module lives beside this file

import run_three_branch_unification as gate  # noqa: E402

SEED = gate.SEED  # 2026
FIGS = os.path.join(_GATE_DIR, "figs")   # verification/figs/ (matches README layout)
os.makedirs(FIGS, exist_ok=True)


def _save(fig, stem):
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Recompute the reading curves (exactly the families/criteria the gate uses)
# --------------------------------------------------------------------------- #
def compute_curves():
    rng = np.random.default_rng(SEED)
    LAP = gate.laplace()
    delta = 1.0

    nested = [[1.0], [0.5, 1.0], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5, 2.5]]
    g_sizes = [len(al) for al in nested]
    g_vals = [float(gate.g_analytic(LAP, al, rng)[0]) for al in nested]

    c_orders = [2, 3, 4, 6]
    c_vals = [float(gate.c_captured(LAP, delta, m, rng)) for m in c_orders]

    return dict(nested=nested, g_sizes=g_sizes, g_vals=g_vals,
                c_orders=c_orders, c_vals=c_vals, delta=delta)


# --------------------------------------------------------------------------- #
# Figure 1 — reading co-movement
# --------------------------------------------------------------------------- #
def fig1_reading_comovement(curves):
    g_sizes, g_vals = curves["g_sizes"], curves["g_vals"]
    c_orders, c_vals = curves["c_orders"], curves["c_vals"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.plot(g_sizes, g_vals, "o-", color="#1f77b4", lw=2.2, ms=8,
            label=r"estimation $g(m)=\|\mathrm{proj}\,S\|^2/I$  (PMM)")
    ax.plot(c_orders, c_vals, "s--", color="#d62728", lw=2.2, ms=8,
            label=r"detection $c(m)=R^2$ of true LLR  (LLR)")

    ax.axhline(1.0, color="0.45", lw=1.2, ls=":")
    ax.annotate(
        "Gaussian terminating anchor (separate experiment):\n"
        r"all readings $=1$ when the basis is sufficient",
        xy=(2.5, 1.0), xycoords="data",
        xytext=(2.7, 0.66), textcoords="data",
        fontsize=9.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="0.6"),
    )

    ax.set_xlabel("number of basis terms", fontsize=12)
    ax.set_ylabel("captured fraction  $\\kappa_m$", fontsize=12)
    ax.set_title("Each captured fraction rises toward 1 as the basis is enriched\n"
                 "Laplace family; estimation $g$ and detection $c$ use two distinct nested bases",
                 fontsize=12)

    all_x = sorted(set(g_sizes) | set(c_orders))
    ax.set_xticks(all_x)
    ax.set_xlim(min(all_x) - 0.4, max(all_x) + 0.4)
    ax.set_ylim(min(g_vals + c_vals) - 0.06, 1.06)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)

    return _save(fig, "fig1_reading_comovement")


# --------------------------------------------------------------------------- #
# Figure 2 — schematic: one series, three functionals
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, ec, fontsize=11, weight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=3)
    return (x, y, w, h)


def fig2_schematic():
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.7, 9)
    ax.axis("off")

    # Central object.
    cx, cy, cw, ch = 2.7, 4.6, 4.6, 1.7
    _box(ax, (cx, cy), cw, ch,
         r"$\log p(x;\theta)$" "\n" "truncated series\nin a Kunchenko basis",
         fc="#e8f0fb", ec="#1f3b66", fontsize=12.5, weight="bold")

    # Three target functionals (right column).
    tx, tw, th = 9.5, 3.4, 1.3
    ys = [7.2, 4.6, 1.0]
    targets = [
        ("PMM / estimation\n" r"$g=\kappa_m(S)$", "#dcefe0", "#1d6b34"),
        ("Detection / LLR\n" r"$c=\kappa_m(\Lambda)$", "#fde2e2", "#8a1f1f"),
        ("DSGE / decomposition\n" r"$R^2=\kappa_m(\log p)$", "#fff0d6", "#8a5a00"),
    ]
    tboxes = []
    for y, (txt, fc, ec) in zip(ys, targets):
        tboxes.append(_box(ax, (tx, y), tw, th, txt, fc=fc, ec=ec, fontsize=12.5))

    op_labels = [r"$\partial_\theta$  (score)", r"$\Delta$  between hypotheses", r"residual norm"]
    edge_x = cx + cw / 2          # right edge of the central box (= 4.6)
    target_left = tx - tw / 2     # left edge of the target column (= 7.8)
    gap_mid = (edge_x + target_left) / 2  # = 6.2; the label corridor between the columns
    for (tboxx, tboxy, twi, thi), lab in zip(tboxes, op_labels):
        start = (edge_x, cy + (tboxy - cy) * 0.15)
        end = (tboxx - twi / 2, tboxy)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                     lw=2.0, color="#444444", zorder=1))
        # Label centred in the inter-column corridor, lifted just above the arrow line.
        frac = (gap_mid - start[0]) / (end[0] - start[0])
        my = start[1] + (end[1] - start[1]) * frac + 0.30
        ax.text(gap_mid, my, lab, ha="center", va="bottom", fontsize=11.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=4)

    # Sequential / change-point sub-branch hanging off the Detection box (Δ∘time).
    dbx, dby, dbw, dbh = tboxes[1]
    seq_cx, seq_cy, seq_w, seq_h = tx + 0.3, 2.5, 3.4, 1.1
    _box(ax, (seq_cx, seq_cy), seq_w, seq_h,
         "sequential / change-point\n" r"$J(s)/\|z\|^2$", fc="#fce8f3", ec="#7a2150",
         fontsize=11.5)
    ax.add_patch(FancyArrowPatch((tx, dby - dbh / 2), (seq_cx, seq_cy + seq_h / 2),
                                 arrowstyle="-|>", mutation_scale=16, lw=1.8,
                                 color="#7a2150", zorder=1))
    # Label kept to the LEFT of the connector so it never runs past the canvas edge.
    ax.text(tx - 0.15, (dby - dbh / 2 + seq_cy + seq_h / 2) / 2,
            r"$\Delta\!\circ\!$time $\to$ stopping rule", ha="right", va="center",
            fontsize=10.5, color="#7a2150",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=4)

    # Bottom banner — the CORRECT statement (not "g=c=R^2").
    ax.text(6.3, -0.35,
            r"one functional $\kappa_m(z;\nu)=\|\mathrm{proj}_{V_m}\mathring z\|^2/\|\mathring z\|^2$"
            " on different objects $z$;  all $=1$ together iff the basis is sufficient",
            ha="center", va="center", fontsize=11.5, style="italic", color="#1f3b66")

    return _save(fig, "fig2_schematic")


# --------------------------------------------------------------------------- #
# Figure 3 — operating characteristics & completeness (the revision panel)
# --------------------------------------------------------------------------- #
def compute_oc():
    rng = np.random.default_rng(SEED)
    LAP = gate.laplace()
    T3 = gate.student_t(3.0)
    nested = [[1.0], [0.5, 1.0], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5, 2.5]]
    t3_nested = [[0.5], [0.5, 1.0], [0.5, 1.0, 1.25]]
    delta = 1.0

    pw_lap = gate.power_vs_order(LAP, 0.09, nested, rng, n=400, M=1200)
    pw_t3 = gate.power_vs_order(T3, 0.16, t3_nested, rng, n=400, M=1200)
    roc_lo = gate.roc_points(LAP, 0.13, [0.5, 1.0], rng, n=400, M=2000)
    roc_hi = gate.roc_points(LAP, 0.13, [0.5, 1.0, 1.5, 2.5], rng, n=400, M=2000)
    rm_lap = gate.rmse_vs_order(LAP, nested, rng, n=500, M=1500)
    lap_grids = [list(np.linspace(0.4, 2.4, j)) for j in (1, 2, 4, 8)]
    t3_grids = [list(np.linspace(0.25, 1.4, j)) for j in (1, 2, 4, 8)]
    cl = gate.completeness_densify(LAP, lap_grids, rng)
    ct = gate.completeness_densify(T3, t3_grids, rng)
    return dict(pw_lap=pw_lap, pw_t3=pw_t3, roc_lo=roc_lo, roc_hi=roc_hi,
                rm_lap=rm_lap, cl=cl, ct=ct)


def fig3_operating_characteristics(oc):
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.4))

    # (a) power vs order — empirical markers vs theory line
    ax = axs[0, 0]
    for rows, col, lab in ((oc["pw_lap"], "#1f77b4", "Laplace"),
                           (oc["pw_t3"], "#d62728", "Student-$t_3$")):
        m = [r[0] for r in rows]
        ax.plot(m, [r[3] for r in rows], "o", color=col, ms=8, label=f"{lab} (empirical)")
        ax.plot(m, [r[4] for r in rows], "-", color=col, lw=1.8, alpha=0.8,
                label=f"{lab}  $\\Phi(\\sqrt{{gI}}\\,h-z_\\alpha)$")
    ax.set_xlabel("order $m$ (basis terms)"); ax.set_ylabel("power (level $0.05$)")
    ax.set_title("(a) Power rises with order, tracking the captured-fraction theory")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8.5, loc="lower right")

    # (b) ROC for two orders (Laplace)
    ax = axs[0, 1]
    for (fpr, tpr, ga), col, lab in ((oc["roc_lo"], "#9ecae1", "order 2"),
                                     (oc["roc_hi"], "#08519c", "order 4")):
        ax.plot(fpr, tpr, "-", color=col, lw=2.2, label=f"{lab}  ($g={ga:.2f}$)")
    ax.plot([0, 1], [0, 1], ":", color="0.6", lw=1.2)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("(b) ROC of the order-$m$ projected-score test (Laplace)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9, loc="lower right")

    # (c) estimator efficiency vs order, tracking g
    ax = axs[1, 0]
    rows = oc["rm_lap"]
    m = [r[0] for r in rows]
    ax.plot(m, [r[1] for r in rows], "-", color="#1d6b34", lw=1.8, label=r"analytic $g=\kappa_m(S)$")
    ax.plot(m, [r[3] for r in rows], "o", color="#1d6b34", ms=8, label="empirical efficiency (CRB/MSE)")
    ax.set_xlabel("order $m$ (basis terms)"); ax.set_ylabel("relative efficiency")
    ax.set_title("(c) Estimator efficiency tracks the captured fraction $g$ (Laplace)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9, loc="lower right")

    # (d) completeness — kappa_m(generic z) -> 1 as exponents densify
    ax = axs[1, 1]
    for rows, col, lab in ((oc["cl"], "#1f77b4", "Laplace (light tail)"),
                           (oc["ct"], "#d62728", "Student-$t_3$ (heavy, $a<1.5$)")):
        q = [r[0] for r in rows]
        ax.plot(q, [r[1] for r in rows], "o-", color=col, lw=2.0, ms=7, label=lab)
    ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel("number of fractional exponents"); ax.set_ylabel(r"$\kappa_m(z)$, generic $z$")
    ax.set_title("(d) Completeness: $\\kappa_m(z)\\to1$ as exponents densify in $(0,\\alpha/2)$")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Operating characteristics and completeness of the captured-fraction framework",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return _save(fig, "fig3_operating_characteristics")


# --------------------------------------------------------------------------- #
def main():
    curves = compute_curves()

    print("=" * 70)
    print(f"make_figures.py  (seed {SEED}) — recomputed reading curves (Laplace)")
    print("=" * 70)
    print("  estimation g(m):")
    for s, g in zip(curves["g_sizes"], curves["g_vals"]):
        print(f"     basis size {s} : g = {g:.4f}")
    print("  detection c(m):")
    for m, c in zip(curves["c_orders"], curves["c_vals"]):
        print(f"     order {m}      : c = {c:.4f}")
    print("  Gaussian anchor : g = c = R^2 = 1.0 (series terminates)")

    p1 = fig1_reading_comovement(curves)
    p2 = fig2_schematic()
    print("\n  computing operating-characteristic & completeness panel (Monte-Carlo)...")
    oc = compute_oc()
    p3 = fig3_operating_characteristics(oc)

    print("\n  wrote:")
    for p in p1 + p2 + p3:
        sz = os.path.getsize(p)
        print(f"     {p}  ({sz} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
