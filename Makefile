# Common developer entry points. GPU targets assume CUDA_VISIBLE_DEVICES is set by the caller.
PY ?= python

.PHONY: lint test test-gpu test-slow fmt

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

test:            ## CPU-only tests (what CI runs)
	$(PY) -m pytest

test-gpu:        ## adds tests that need a CUDA device
	$(PY) -m pytest --gpu

test-slow:       ## adds tests that download small models/datasets
	$(PY) -m pytest --gpu --slow
