# Reproduction targets for the Paper 4 verification package.
PY ?= python3

.PHONY: all gate comparators realdata realdata2 refeff figures lean check test clean
all: gate comparators realdata realdata2 figures lean

gate:            ## Section 6: the 15-check numerical verification (seed 2026)
	$(PY) python/run_three_branch_unification.py

comparators:     ## Section 7: comparators on all three branches (seed 2026)
	$(PY) python/run_comparators.py

realdata:        ## Section 8.1: the pre-registered FRED study (seed 2026; reads data/)
	$(PY) python/run_realdata.py

realdata2:       ## Section 8.2: Study 2 on EEGdenoiseNet EMG epochs (seed 2026; reads data/)
	$(PY) python/run_realdata_study2.py

refeff:          ## Section 8.2 "Reading": closed-form efficiencies of the robust arms vs the t MLE
	$(PY) python/reference_efficiencies.py

figures:         ## regenerate the figures from the gate (nothing hardcoded)
	$(PY) python/make_figures.py

lean:            ## build the machine-checked core (needs elan/lake; fetches Mathlib cache)
	cd lean && lake exe cache get && lake build

check: gate comparators realdata realdata2  ## alias: confirm PASS
	@echo "compare the four outputs above against python/expected_output.txt,"
	@echo "python/expected_output_comparators.txt, python/expected_output_realdata.txt"
	@echo "and python/expected_output_realdata_study2.txt"
	@echo "(seed 2026; allow platform-dependent floating-point rounding)"

test:           ## focused admissibility and data-protocol regression checks
	$(PY) -m unittest discover -s python -p 'test_*.py'

clean:
	rm -rf python/__pycache__ lean/.lake figs/*.png figs/*.pdf
