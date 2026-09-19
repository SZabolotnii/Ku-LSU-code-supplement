# Reproduction targets for the Paper 4 verification package.
PY ?= python3

.PHONY: all gate comparators realdata figures lean check clean
all: gate comparators realdata figures lean

gate:            ## Section 6: the 15-check numerical verification (seed 2026)
	$(PY) python/run_three_branch_unification.py

comparators:     ## Section 7: comparators on all three branches (seed 2026)
	$(PY) python/run_comparators.py

realdata:        ## Section 8: the pre-registered FRED study (seed 2026; reads data/)
	$(PY) python/run_realdata.py

figures:         ## regenerate the figures from the gate (nothing hardcoded)
	$(PY) python/make_figures.py

lean:            ## build the machine-checked core (needs elan/lake; fetches Mathlib cache)
	cd lean && lake exe cache get && lake build

check: gate comparators realdata  ## alias: confirm PASS
	@echo "compare the three outputs above against python/expected_output.txt,"
	@echo "python/expected_output_comparators.txt and python/expected_output_realdata.txt"
	@echo "(each reproduces byte-for-byte at seed 2026)"

clean:
	rm -rf python/__pycache__ lean/.lake figs/*.png figs/*.pdf
