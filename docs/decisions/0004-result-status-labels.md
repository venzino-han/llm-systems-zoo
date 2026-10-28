# ADR-0004: Status labels for techniques and for numbers

**Status.** Accepted (2026-10-29)

## Context
A technology map that lists everything looks like a wrapper collection unless each item says how far
it was actually taken. Numbers measured on a shared, noisy GPU must not be confused with measurements.

## Decision
Every lab and sub-experiment carries one of:

| Label | Meaning |
|---|---|
| Planned | Described in the docs; no code yet |
| Implemented | Code exists, CPU tests pass, `--smoke` ran |
| Measured | A pilot or measurement run produced numbers that are in `reports/<lab>/results/` |
| Reproduced | The trend was confirmed by at least one independent re-run on a stated environment |

Every number additionally carries a measurement grade:

* **smoke**: proves the script runs; not a performance claim.
* **pilot**: small grid, possibly a shared GPU, few repeats; p99 flagged as under-sampled below 100 requests.
* **measurement**: fixed grid, exclusive hardware, repeats with spread reported.

## Consequences
* The root README's technique map shows the label per item; claims in a CV follow the label, not the plan.
* "Local load test", "single node multi-GPU", "multi-node" and "production operation" are distinct claims.
