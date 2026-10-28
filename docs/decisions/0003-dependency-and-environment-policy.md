# ADR-0003: Light core package, heavy dependencies as extras, pinned environments per published run

**Status.** Accepted (2026-10-29)

## Context
The simulators, calculators, metrics and analysis code should run in a CPU-only CI job in seconds.
The GPU labs need torch, vLLM, TRL, Ray and friends, whose version constraints conflict with each other
over time.

## Decision
* `llm_systems_zoo` (the harness) depends only on numpy, pyyaml, matplotlib, pandas, rich.
* Track-specific dependencies are `pyproject.toml` extras: `inference`, `training`, `ray`, `apps`, `torch`, `dev`.
* CI installs core + dev + CPU torch and runs lint and CPU tests. GPU tests run manually with `pytest --gpu`.
* The exact versions used for a published number are recorded in the run manifest (automatic), and the
  reference environment is documented in `docs/hardware.md`.
* Model weights, checkpoints and large datasets never enter git.

## Consequences
* `pip install -e ".[inference]"` may need a specific torch/CUDA wheel on a given machine; the README
  points to the engine's own install docs instead of promising a universal lock file.
* A reader can always see which versions produced a number even when the extras drift.
