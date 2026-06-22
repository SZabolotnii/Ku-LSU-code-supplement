#!/usr/bin/env bash
# One-command reproduction of the Paper 4 verification package. Seed 2026 throughout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

echo "== [1/3] dependencies =="
"$PY" -m pip install -r "$HERE/requirements.txt"

echo "== [2/3] numerical verification (a few minutes) =="
"$PY" "$HERE/python/run_three_branch_unification.py"

echo "== [3/3] figures =="
"$PY" "$HERE/python/make_figures.py"

echo
echo "Numerical reproduction complete. For the machine-checked core run:"
echo "    cd '$HERE/lean' && lake exe cache get && lake build   # 0 sorry"
