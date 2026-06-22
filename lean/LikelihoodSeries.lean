import Mathlib

/-!
# Detection branch of the master series — LLR = Δ(½D²) (machine-checked target)

Backs `Spec-Unification-LikelihoodSeries.md` §2 (rigorous core, detection branch) and
generalizes `../../DSGE/papers/dsge-spectral/lean/DescriptorBasis/BayesBound.lean`.

The master object is the truncated expansion of `log p(x;θ)` in a Kunchenko basis; the
**detection branch** is `Δ` between hypotheses, i.e. the log-likelihood ratio (LLR).  When
the per-class model is Gaussian-in-the-basis (the 2nd-order truncation realized by DSGE),
its log-density at a descriptor point `s` is

    log p_c(s) = -½·D_c(s) - ½·log|F_c| - (d/2)·log 2π,        D_c(s) = (s-μ_c)ᵀ F_c⁻¹ (s-μ_c)

so the LLR reduces to an **affine function of the difference of the DSGE Mahalanobis
("decomposition") scores**:

    Λ(s) = log p₁/p₀ = -(½D₁ - ½D₀)(s) - ½(log|F₁| - log|F₀|).

This is the formal content of "the detection functional is `Δ` of the decomposition
functional".  Two consequences are mechanized:
  * `llr_eq_score_difference` — the identity above (and its equal-`log|F|` corollary,
    which is exactly the gate's Gaussian anchor A2: QDA discriminant = true LLR);
  * `argmax_logDensity_iff_argmin_penalty` + `exists_argmax_logDensity` — the Bayes
    (max-posterior) decision **is** the DSGE min-penalized-score decision, so
    `BayesBound.bayes_expected_optimal` transfers verbatim: the DSGE rule is Bayes-optimal
    in this regime.  This is the generalization of `BayesBound` (which proved only that the
    argmax-posterior rule is optimal; here we identify *which* statistic that rule is).

We take the Gaussian-in-basis log-density *form* as the hypothesis (= the 2nd-order
truncation assumption); we do not re-derive it from matrix algebra.  The reductions are
then pure ring/linear-arithmetic facts, hence robust.

**GSA anchor (fourth branch — sequential change-point detection).**  The second half of the
file mechanizes the anchor `GSA@S=2 = Ghosh = Bayes` (gate A3; `../../DSGE/papers/dsge-multivariate/`
CONCLUSIONS §6): for two *concrete scalar Gaussians* the LLR is an explicit polynomial of
degree ≤ 2 (`gauss_llr_eq_quadratic`), hence lies in `span{1, x, x²}`
(`gauss_llr_mem_span_quadratics`) — so the order-2 GSA projection returns the LLR itself
and the information functional saturates: the partial Parseval sum over any orthonormal
system spanning the truth equals the full energy (`sum_inner_sq_eq_norm_sq_of_combination`,
`sum_inner_sq_eq_norm_sq_of_mem_span`, and the `HilbertBasis` form `J_saturates`, which is
the saturation of `J(s)` from KuYuPe `InfoFunctional.lean` — there `J(s) ≤ ‖z‖²` always,
here `=` when the series terminates at order `s`).  The same LLR is an affine function of
the two Mahalanobis heads (`gauss_llr_eq_mahala_affine` — the Ghosh-GAM architecture for
the Gaussian case, a concrete instance of `llr_eq_score_difference`), and its sign is the
Bayes decision (`llr_nonneg_iff_penalty_le`, `gauss_bayes_iff_min_penalty`).  Equal
variances kill the quadratic term (`gauss_llr_linear_of_eq_var`) — the `S=1` case of
KuYuPe `GaussianLimit.lean`, so the two anchors form one ladder.  CUSUM optimality for the
exact LLR (Lorden/Moustakides) is *cited*, not re-proven: the mechanized content is that
GSA@S=2 *is* the exact LLR in this regime.

NOTE: written for Mathlib `v4.26.0`.  **Machine-checked 2026-06-12**: the full file (all
16 theorems, GSA-anchor section included) builds with zero errors/warnings via the
documented drop-in procedure — copy to `../../DSGE/papers/dsge-spectral/lean/DescriptorBasis/LikelihoodSeries.lean`
and run `lake build DescriptorBasis.LikelihoodSeries` (toolchain `leanprover/lean4:v4.26.0`,
the Paper-1/2 project's Mathlib cache; 7743 jobs OK).  The drop-in copy is removed after
verification — the canonical source stays here to preserve Paper 3 / Paper 4 separation.
Self-contained (`import Mathlib`).
-/

namespace LikelihoodSeries

variable {S : Type*}

/-- A Gaussian-in-basis per-class model: the Mahalanobis score field `D s = (s-μ)ᵀF⁻¹(s-μ)`
and the (point-independent) log-determinant `ld = log|F|`.  This is the 2nd-order
truncation of `-log p_c` in the descriptor basis (the DSGE/C2 realization). -/
structure GaussBasis (S : Type*) where
  D  : S → ℝ
  ld : ℝ

/-- Gaussian-in-basis log-density at a point (shared constant `κ = (d/2)·log 2π`). -/
noncomputable def logDensity (m : GaussBasis S) (κ : ℝ) (s : S) : ℝ :=
  -(1 / 2) * m.D s - (1 / 2) * m.ld - κ

/-- DSGE **decomposition** score: the half-Mahalanobis `½·D_c(s)` (the per-class
reconstruction penalty; `log-MSED` is its empirical stand-in). -/
noncomputable def score (m : GaussBasis S) (s : S) : ℝ := (1 / 2) * m.D s

/-- The full per-class penalty `½D_c + ½log|F_c|` whose argmin is the classifier's choice. -/
noncomputable def penalty (m : GaussBasis S) (s : S) : ℝ := (1 / 2) * m.D s + (1 / 2) * m.ld

/-- The **detection functional**: the log-likelihood ratio between two class models. -/
noncomputable def llr (m₁ m₀ : GaussBasis S) (κ : ℝ) (s : S) : ℝ :=
  logDensity m₁ κ s - logDensity m₀ κ s

/-- **Detection bridge (the identity).** The LLR equals the negative difference of the DSGE
decomposition scores plus the log-determinant offset:
`Λ = -(½D₁ - ½D₀) - ½(log|F₁| - log|F₀|)`.  Detection = `Δ` of the decomposition branch. -/
theorem llr_eq_score_difference (m₁ m₀ : GaussBasis S) (κ : ℝ) (s : S) :
    llr m₁ m₀ κ s = -(score m₁ s - score m₀ s) - (1 / 2) * (m₁.ld - m₀.ld) := by
  unfold llr logDensity score
  ring

/-- **Gaussian anchor (gate A2).** When the class covariances share a log-determinant
(`log|F₁| = log|F₀|`, e.g. equal covariance), the offset vanishes and the LLR is exactly the
difference of the DSGE scores — the QDA discriminant equals the true LLR with no constant. -/
theorem llr_eq_score_difference_of_equal_logdet (m₁ m₀ : GaussBasis S) (κ : ℝ) (s : S)
    (h : m₁.ld = m₀.ld) :
    llr m₁ m₀ κ s = -(score m₁ s - score m₀ s) := by
  rw [llr_eq_score_difference, h]; ring

/-- Posterior (log-density) is the negated penalty up to the shared constant. -/
theorem logDensity_eq_neg_penalty (m : GaussBasis S) (κ : ℝ) (s : S) :
    logDensity m κ s = -(penalty m s) - κ := by
  unfold logDensity penalty; ring

/-- **Two-class decision equivalence.** Choosing the higher-posterior class is the same as
choosing the lower DSGE penalized score — the LLR sign is the decision. -/
theorem ge_logDensity_iff_le_penalty (m₁ m₀ : GaussBasis S) (κ : ℝ) (s : S) :
    logDensity m₀ κ s ≤ logDensity m₁ κ s ↔ penalty m₁ s ≤ penalty m₀ s := by
  unfold logDensity penalty
  constructor <;> intro h <;> linarith

/-- **Multi-class: Bayes decision = DSGE min-penalty decision.** A label `c` maximizes the
posterior over all labels iff it minimizes the DSGE penalty over all labels. -/
theorem argmax_logDensity_iff_argmin_penalty {ι : Type*}
    (m : ι → GaussBasis S) (κ : ℝ) (s : S) (c : ι) :
    (∀ g, logDensity (m g) κ s ≤ logDensity (m c) κ s) ↔
      (∀ g, penalty (m c) s ≤ penalty (m g) s) := by
  constructor
  · intro h g
    have hg := h g
    simp only [logDensity, penalty] at hg ⊢
    linarith
  · intro h g
    have hg := h g
    simp only [logDensity, penalty] at hg ⊢
    linarith

/-- A Bayes (max-posterior) decision exists for finitely many classes — the same statistic
that `BayesBound.bayes_expected_optimal` proves optimal.  Via
`argmax_logDensity_iff_argmin_penalty` this decision is realized by the DSGE min-penalty
rule, transferring optimality to DSGE in the Gaussian-in-basis regime. -/
theorem exists_argmax_logDensity {ι : Type*} [Fintype ι] [Nonempty ι]
    (m : ι → GaussBasis S) (κ : ℝ) (s : S) :
    ∃ c, ∀ g, logDensity (m g) κ s ≤ logDensity (m c) κ s :=
  Finite.exists_max (fun g => logDensity (m g) κ s)

/-- **Detection functional sign = Bayes decision.**  The two-hypothesis Bayes rule is the
sign of the detection functional: `Λ ≥ 0` iff the `H₁` penalty is the smaller one.  This is
the statement that feeds CUSUM: accumulating increments of `Λ` accumulates Bayes evidence. -/
theorem llr_nonneg_iff_penalty_le (m₁ m₀ : GaussBasis S) (κ : ℝ) (s : S) :
    0 ≤ llr m₁ m₀ κ s ↔ penalty m₁ s ≤ penalty m₀ s := by
  unfold llr logDensity penalty
  constructor <;> intro h <;> linarith

/-! ## GSA anchor — sequential-detection branch: `GSA@S=2 = Ghosh = Bayes`

Concrete scalar-Gaussian instantiation of the detection bridge, plus the Hilbert-space
saturation lemma behind «`J(s)` = captured divergence».  See the module docstring. -/

section GsaAnchor

/-! ### Saturation of the information functional

`J(s)` of KuYuPe `InfoFunctional.lean` is the partial Parseval sum
`∑_{i<s} (b.repr z i)²`; Theorem 2a there gives `J(s) ≤ ‖z‖²`.  Here: **equality** holds
the moment `z` is a combination of the first `s` basis vectors — the truncated series
*terminates*, the order-`s` GSA projection is exact, no divergence is lost. -/

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

open scoped RealInnerProductSpace

/-- Partial Parseval saturation over any finite orthonormal family: the sum of squared
inner products of a *member of the family's span* against the family recovers the full
squared norm.  (For a non-member, `≤` with strict loss — that is the truncation residual.) -/
theorem sum_inner_sq_eq_norm_sq_of_combination
    {ι : Type*} {e : ι → H} (he : Orthonormal ℝ e) (c : ι → ℝ) (s : Finset ι) :
    ∑ i ∈ s, ⟪e i, ∑ j ∈ s, c j • e j⟫ ^ 2 = ‖∑ j ∈ s, c j • e j‖ ^ 2 := by
  calc ∑ i ∈ s, ⟪e i, ∑ j ∈ s, c j • e j⟫ ^ 2
      = ∑ i ∈ s, c i * c i :=
        Finset.sum_congr rfl fun i hi => by rw [he.inner_right_sum c hi]; ring
    _ = ⟪∑ i ∈ s, c i • e i, ∑ j ∈ s, c j • e j⟫ := by
        rw [he.inner_sum c c s]
        exact Finset.sum_congr rfl fun i _ => by simp
    _ = ‖∑ j ∈ s, c j • e j‖ ^ 2 := real_inner_self_eq_norm_sq _

/-- Span form of the saturation: any `z` in the span of a finite orthonormal family has its
whole energy captured by the family's coefficients.  Applied to the GSA anchor: the
Gaussian LLR lies in the degree-2 span, so the order-2 system captures **all** of it. -/
theorem sum_inner_sq_eq_norm_sq_of_mem_span
    {n : ℕ} {e : Fin n → H} (he : Orthonormal ℝ e) {z : H}
    (hz : z ∈ Submodule.span ℝ (Set.range e)) :
    ∑ i, ⟪e i, z⟫ ^ 2 = ‖z‖ ^ 2 := by
  obtain ⟨c, rfl⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).mp hz
  simpa using sum_inner_sq_eq_norm_sq_of_combination he c Finset.univ

/-- **`J(s)` saturates when the series terminates at order `s`** (`HilbertBasis` form,
matching `GSA.Part2.InfoFunctional.J` verbatim): if `z` is a combination of the first `s`
Hilbert-basis vectors, the partial sum `∑_{i<s} (b.repr z i)²` *equals* `‖z‖²`.
Together with Theorem 2a/2c of `InfoFunctional.lean` (`J(s) ≤ ‖z‖²`, `J(s) → ‖z‖²`) this
pins the GSA currency: `J(s)/‖z‖²` is the captured-divergence fraction `c`, and it hits `1`
at finite order exactly in the terminating (e.g. Gaussian-quadratic) regime. -/
theorem J_saturates (b : HilbertBasis ℕ ℝ H) (s : ℕ) (c : ℕ → ℝ) :
    ∑ i ∈ Finset.range s, (b.repr (∑ j ∈ Finset.range s, c j • b j) i) ^ 2
      = ‖∑ j ∈ Finset.range s, c j • b j‖ ^ 2 := by
  have h := sum_inner_sq_eq_norm_sq_of_combination b.orthonormal c (Finset.range s)
  refine Eq.trans (Finset.sum_congr rfl fun i _ => ?_) h
  rw [HilbertBasis.repr_apply_apply]

/-! ### Concrete scalar-Gaussian anchor

`H₀ = N(μ₀, v₀)`, `H₁ = N(μ₁, v₁)` (variances `v_c`, shared constant `κ = ½·log 2π`).
The Mahalanobis field is `(x−μ)²/v`, the log-determinant is `log v` — a literal
`GaussBasis` over `S := ℝ`, so every abstract theorem above instantiates. -/

variable {μ₀ μ₁ v₀ v₁ v : ℝ}

/-- Scalar Gaussian log-density (shared constant `κ = ½·log 2π`). -/
noncomputable def gaussLogDensity (μ v κ x : ℝ) : ℝ :=
  -(x - μ) ^ 2 / (2 * v) - (1 / 2) * Real.log v - κ

/-- Scalar Mahalanobis score `(x−μ)²/v`. -/
noncomputable def mahala (μ v x : ℝ) : ℝ := (x - μ) ^ 2 / v

/-- Package a scalar Gaussian as a `GaussBasis` over `ℝ`: `D := mahala`, `ld := log v`. -/
noncomputable def toGaussBasis (μ v : ℝ) : GaussBasis ℝ :=
  ⟨mahala μ v, Real.log v⟩

/-- The concrete scalar Gaussian log-density **is** the abstract Gaussian-in-basis
log-density of its `GaussBasis` package — the hypothesis of every abstract theorem above
is *satisfied*, not assumed, in the scalar Gaussian case. -/
theorem gaussLogDensity_eq_logDensity (μ v κ x : ℝ) :
    gaussLogDensity μ v κ x = logDensity (toGaussBasis μ v) κ x := by
  simp only [gaussLogDensity, logDensity, toGaussBasis, mahala]
  ring

/-- **GSA side of the anchor: the Gaussian LLR is an explicit polynomial of degree ≤ 2.**
A covariance shift makes the leading coefficient `1/(2v₀) − 1/(2v₁)` nonzero — exactly the
component an order-1 (`S=1`) system cannot represent and an order-2 (`S=2`) system
represents exactly. -/
theorem gauss_llr_eq_quadratic (hv₀ : v₀ ≠ 0) (hv₁ : v₁ ≠ 0) (κ x : ℝ) :
    llr (toGaussBasis μ₁ v₁) (toGaussBasis μ₀ v₀) κ x =
      (1 / (2 * v₀) - 1 / (2 * v₁)) * x ^ 2
        + (μ₁ / v₁ - μ₀ / v₀) * x
        + ((μ₀ ^ 2 / (2 * v₀) - μ₁ ^ 2 / (2 * v₁))
            - (1 / 2) * (Real.log v₁ - Real.log v₀)) := by
  simp only [llr, logDensity, toGaussBasis, mahala]
  field_simp
  ring

/-- Equal variances (`S=1` rung of the ladder, KuYuPe `GaussianLimit.lean`): the quadratic
term vanishes and the LLR is affine — the order-1 projection is already exact. -/
theorem gauss_llr_linear_of_eq_var (hv : v ≠ 0) (κ x : ℝ) :
    llr (toGaussBasis μ₁ v) (toGaussBasis μ₀ v) κ x =
      ((μ₁ - μ₀) / v) * x + (μ₀ ^ 2 - μ₁ ^ 2) / (2 * v) := by
  simp only [llr, logDensity, toGaussBasis, mahala]
  field_simp
  ring

/-- **The Gaussian LLR lies in `span{1, x, x²}`** — the formal content of «GSA@S=2 is
exact»: the orthogonal projection onto the degree-2 subspace fixes the LLR (projection of
a member is the member), so by `sum_inner_sq_eq_norm_sq_of_mem_span`/`J_saturates` the
order-2 GSA system captures the full divergence, in any sampling measure. -/
theorem gauss_llr_mem_span_quadratics (hv₀ : v₀ ≠ 0) (hv₁ : v₁ ≠ 0) (κ : ℝ) :
    llr (toGaussBasis μ₁ v₁) (toGaussBasis μ₀ v₀) κ ∈
      Submodule.span ℝ
        ({fun _ => 1, fun x => x, fun x => x ^ 2} : Set (ℝ → ℝ)) := by
  have hrep : llr (toGaussBasis μ₁ v₁) (toGaussBasis μ₀ v₀) κ =
      ((μ₀ ^ 2 / (2 * v₀) - μ₁ ^ 2 / (2 * v₁))
          - (1 / 2) * (Real.log v₁ - Real.log v₀)) • (fun _ : ℝ => (1 : ℝ))
        + (μ₁ / v₁ - μ₀ / v₀) • (fun x : ℝ => x)
        + (1 / (2 * v₀) - 1 / (2 * v₁)) • (fun x : ℝ => x ^ 2) := by
    funext x
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [gauss_llr_eq_quadratic hv₀ hv₁ κ x]
    ring
  rw [hrep]
  refine Submodule.add_mem _ (Submodule.add_mem _ ?_ ?_) ?_ <;>
    refine Submodule.smul_mem _ _ (Submodule.subset_span ?_)
  · exact Set.mem_insert _ _
  · exact Set.mem_insert_of_mem _ (Set.mem_insert _ _)
  · exact Set.mem_insert_of_mem _ (Set.mem_insert_of_mem _ rfl)

/-- **Ghosh side of the anchor:** the Gaussian LLR is an *affine function of the two
Mahalanobis heads* `D²₀, D²₁` — the Ghosh-GAM architecture with the identity link, and the
concrete instance of `llr_eq_score_difference`.  GSA@S=2, the Mahalanobis-head classifier,
and the exact LLR are one statistic in this regime. -/
theorem gauss_llr_eq_mahala_affine (κ x : ℝ) :
    llr (toGaussBasis μ₁ v₁) (toGaussBasis μ₀ v₀) κ x =
      (1 / 2) * mahala μ₀ v₀ x - (1 / 2) * mahala μ₁ v₁ x
        - (1 / 2) * (Real.log v₁ - Real.log v₀) := by
  simp only [llr, logDensity, toGaussBasis, mahala]
  ring

/-- **Bayes side of the anchor:** for the concrete scalar Gaussians, choosing the
higher-density hypothesis is choosing the smaller DSGE penalty (instantiation of
`ge_logDensity_iff_le_penalty`); with `llr_nonneg_iff_penalty_le`, the sign of the
(GSA@S=2-exact) LLR **is** the Bayes decision. -/
theorem gauss_bayes_iff_min_penalty (κ x : ℝ) :
    gaussLogDensity μ₀ v₀ κ x ≤ gaussLogDensity μ₁ v₁ κ x ↔
      penalty (toGaussBasis μ₁ v₁) x ≤ penalty (toGaussBasis μ₀ v₀) x := by
  rw [gaussLogDensity_eq_logDensity, gaussLogDensity_eq_logDensity]
  exact ge_logDensity_iff_le_penalty _ _ κ x

end GsaAnchor

end LikelihoodSeries
