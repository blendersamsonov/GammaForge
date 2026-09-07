.PHONY: check

PYTHON ?= python

check:
	$(PYTHON) -m pytest
