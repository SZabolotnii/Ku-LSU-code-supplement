import Lake
open Lake DSL

/-! Standalone reproduction of the Paper 4 Lean core (`LikelihoodSeries.lean`).
    Pinned to Lean v4.26.0 + Mathlib v4.26.0 (see `lean-toolchain` and the committed
    `lake-manifest.json`). Build with:

        lake exe cache get      -- fetch prebuilt Mathlib oleans (recommended)
        lake build LikelihoodSeries

    Expected: `Build completed successfully`, 0 `sorry`. -/

package «paper4-verification»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.26.0"

@[default_target]
lean_lib LikelihoodSeries where
  roots := #[`LikelihoodSeries]
