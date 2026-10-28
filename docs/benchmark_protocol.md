# Benchmark protocol

Every number published in this repository was produced under the rules below. The rules exist
so that a reader can tell *what* was measured, *where*, and *how fairly* before trusting it.

## 1. What every run records

The harness (`llm_systems_zoo.manifest.Run`) writes a `manifest.json` next to every result.

| Category | Fields | Collected by |
|---|---|---|
| Hardware | GPU model / count / VRAM, interconnect (NVLink vs PCIe), CPU, RAM, node count | `hardware.hardware_inventory()` (automatic) |
| Software | OS, driver, CUDA, framework/engine versions, package versions | `hardware.software_inventory()` (automatic) |
| Model | HF id, revision, tokenizer, dtype, quantisation, parallelism (TP/PP/replicas) | the lab script (`Run(model=...)`) |
| Workload | dataset, seed, input/output length distribution, concurrency or arrival rate | the lab script (`Run(workload=...)`) |
| Execution | warm-up, repeats, timeout, cache state (cold/warm), start/end time | the lab script (`Run(execution=...)`) |
| Quality | eval set version, scoring method, judge settings if any | the lab script |
| Performance | raw per-request / per-step records, tokens, errors, memory, GPU counters | `records.jsonl`, `metrics.jsonl` |
| Cost | GPU-hours and the price assumption and its date, if a cost is quoted | the report |
| Provenance | config hash, run id, git commit, dirty flag, exact command | `manifest.git_provenance()` (automatic) |

## 2. Fair-comparison rules

1. **One variable at a time.** A combined optimisation is decomposed into an ablation.
2. **Same model, same workload, same quality bar** across the compared configurations.
3. **Server-internal tokens/s and client end-to-end throughput are different numbers** and are labelled as such.
4. **Input-token and output-token throughput are reported separately.** Prefill-heavy and decode-heavy workloads saturate different resources.
5. **Failures count.** Time-outs, OOMs and rejected requests stay in the denominator. `slo_attainment` is computed over *submitted* requests.
6. **Report the spread.** Repeated runs report mean and standard deviation; quality differences come with sample sizes.
7. **Synthetic and application workloads are labelled** and never mixed in one curve.
8. **Only measured improvements are quoted.** Targets, plan numbers and examples are never formatted like results.
9. **p99 from a short pilot is flagged** as under-sampled (fewer than ~100 requests per condition).

## 3. Metric definitions

Defined once in `llm_systems_zoo/metrics.py` and reused by every lab:

* `TTFT = t_first_token - t_submit` (includes queueing)
* `TPOT = (t_end - t_first_token) / (output_tokens - 1)`
* `E2E = t_end - t_submit`
* Throughput is computed over the observed window `[min t_submit, max t_end]`.
* `goodput_rps = requests meeting every SLO bound / window`
* `slo_attainment = requests meeting every SLO bound / submitted requests`
* Closed-loop load fixes concurrency (each client waits for its previous reply); open-loop load fixes the
  arrival rate (Poisson by default) regardless of the server's state. Only open-loop load can show queue build-up.

## 4. Execution ladder

`smoke → pilot → measurement`. A smoke run proves the script works (CPU or tiny model). A pilot finds
the variables that matter at low cost. A measurement run fixes the grid, the repeat count and the
budget (conditions × repeats × expected duration × GPUs = GPU-hours) before it starts.

## 5. Result directories

```
runs/<lab>/<timestamp>_<name>_<cfghash>/   raw, git-ignored
reports/<lab>/results/                     curated summary.json + figures, committed
reports/<lab>/REPORT.md                    question / environment / hypothesis / baseline / change /
                                           method / results / bottleneck analysis / limits /
                                           reproduction commands / next experiments
```

Large traces and checkpoints are kept out of git and referenced by path and checksum.
