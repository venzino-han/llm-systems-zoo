# Contributing / working in this repository

## Setup

```bash
uv venv --python 3.10 .venv            # add --system-site-packages to reuse a machine-wide torch/vLLM
uv pip install -e ".[dev]"             # harness + tests (CPU)
uv pip install -e ".[inference]"       # vLLM-based labs (needs a matching torch/CUDA wheel; see vLLM docs)
uv pip install -e ".[training]"        # L05 / L06
uv pip install -e ".[ray]"             # L07
uv pip install -e ".[apps]"            # L09
```

## Test levels

```bash
make lint          # ruff
make test          # CPU tests only (what CI runs)
make test-gpu      # + tests marked `gpu`
make test-slow     # + tests that download small models/datasets
```

## Adding an experiment

1. Create the script under the lab directory with an argparse CLI and a `--smoke` mode.
2. Wrap the run in `llm_systems_zoo.Run(...)` so the manifest, config hash and environment are recorded.
3. Log raw records (`records.jsonl`) and the aggregate (`summary.json`); produce figures with `llm_systems_zoo.plotting`.
4. Copy the curated summary and figure to `reports/<lab>/results/` and update the lab README and REPORT.
5. Add CPU tests for anything that can be tested without a GPU.

## Definition of done (per experiment)

- [ ] Question and baseline are explicit.
- [ ] Environment and model revision are fixed in the manifest.
- [ ] Correctness or quality check passed.
- [ ] Reproduction commands and required hardware are stated.
- [ ] Raw metrics and the code that produced the summary are present.
- [ ] At least one independent re-run confirmed the trend (or the result is labelled pilot).
- [ ] Failures, OOMs, constraints and negative results are recorded.
- [ ] The cause of an improvement is explained with measurements, not assumptions.
- [ ] The root README's map links to the evidence.

## Style

* Python 3.10+, ruff (line length 100). Type hints on public functions.
* English for all documentation and code comments.
* Never commit weights, checkpoints, large datasets or raw run directories.
