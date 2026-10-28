# benchmarks/

Shared pieces that labs reuse so comparisons stay fair:

* `workloads/` – YAML workload definitions (length distributions, arrival processes, prefix sharing).
  Serving (L03), multi-GPU (L04) and Ray Serve (L07) experiments load the same files.
* `analysis/aggregate_runs.py` – walks `runs/` and prints/exports one row per run (lab, name, status,
  config hash, duration, key summary fields).

The metric definitions live in `llm_systems_zoo/metrics.py` and are documented in
`docs/benchmark_protocol.md`.
