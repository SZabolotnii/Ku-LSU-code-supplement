# Reproduction targets for the Paper 4 verification package.
PY ?= python3

.PHONY: all gate figures lean check clean
all: gate figures lean

gate:            ## run the 15-check numerical verification (seed 2026)
	$(PY) python/run_three_branch_unification.py

figures:         ## regenerate the figures from the gate (nothing hardcoded)
	$(PY) python/make_figures.py

lean:            ## build the machine-checked core (needs elan/lake; fetches Mathlib cache)
	cd lean && lake exe cache get && lake build

check: gate      ## alias: confirm PASS
	@echo "compare stdout above against python/expected_output.txt"

clean:
	rm -rf python/__pycache__ lean/.lake figs/*.png figs/*.pdf
