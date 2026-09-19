#!/usr/bin/env bash
# One-command reproduction of the Paper 4 verification package. Seed 2026 throughout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

echo "== [1/5] dependencies =="
"$PY" -m pip install -r "$HERE/requirements.txt"

echo "== [2/5] Section 6: the fifteen pre-specified checks (a few minutes) =="
"$PY" "$HERE/python/run_three_branch_unification.py"

echo "== [3/5] Section 7: comparators on all three branches (a few minutes) =="
"$PY" "$HERE/python/run_comparators.py"

echo "== [4/5] Section 8: the pre-registered real-data study (reads data/) =="
"$PY" "$HERE/python/run_realdata.py"

echo "== [5/5] figures =="
"$PY" "$HERE/python/make_figures.py"

echo
echo "Numerical reproduction complete. For the machine-checked core run:"
echo "    cd '$HERE/lean' && lake exe cache get && lake build   # 0 sorry"
