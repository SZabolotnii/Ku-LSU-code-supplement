#!/usr/bin/env bash
# One-command reproduction of the Paper 4 verification package. Seed 2026 throughout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

echo "== [1/8] dependencies =="
"$PY" -m pip install -r "$HERE/requirements.txt"

echo "== [2/8] Section 6: the fifteen pre-specified checks (a few minutes) =="
"$PY" "$HERE/python/run_three_branch_unification.py"

echo "== [3/8] Section 7: comparators on all three branches (a few minutes) =="
"$PY" "$HERE/python/run_comparators.py"

echo "== [4/8] Section 8.1: the pre-registered exchange-rate study (reads data/) =="
"$PY" "$HERE/python/run_realdata.py"

echo "== [5/8] Section 8.2: Study 2 on EEGdenoiseNet EMG epochs (reads data/) =="
"$PY" "$HERE/python/run_realdata_study2.py"

echo "== [6/8] Section 8.2: reference efficiencies of the robust arms (closed form) =="
"$PY" "$HERE/python/reference_efficiencies.py"

echo "== [7/8] regression tests =="
"$PY" -m unittest discover -s "$HERE/python" -p 'test_*.py'

echo "== [8/8] figures =="
"$PY" "$HERE/python/make_figures.py"

echo
echo "Numerical reproduction complete. For the machine-checked core run:"
echo "    cd '$HERE/lean' && lake exe cache get && lake build   # 0 sorry"
