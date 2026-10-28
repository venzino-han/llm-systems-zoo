# Closed-loop concurrency sweep: short_qa

- workload: benchmarks/workloads/short_qa.yaml
- backend: external (pilot_baseline)
- SLO: ttft=0.5,tpot=0.05
- tag: pilot, shared GPU, 0.10 util, A6000, enforce-eager, max_model_len 2048

| label | mode | offered_load | n_success/n_submitted | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | E2E p99 (s) | err | goodput req/s | SLO attain. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_baseline | closed | 1 | 32/32 * | 0.87 | 120 | 18/215 | 6/38 | 3.95 | 0.00 | 0.87 | 1.00 |
| pilot_baseline | closed | 4 | 32/32 * | 3.83 | 530 | 23/69 | 7/8 | 1.85 | 0.00 | 3.83 | 1.00 |
| pilot_baseline | closed | 16 | 64/64 * | 16.90 | 2095 | 25/62 | 7/8 | 1.73 | 0.00 | 16.90 | 1.00 |
| pilot_baseline | closed | 64 | 256/256 | 49.63 | 6342 | 36/202 | 8/10 | 2.07 | 0.00 | 49.63 | 1.00 |
| pilot_baseline | closed | 128 | 512/512 | 75.48 | 9914 | 85/280 | 10/12 | 2.61 | 0.00 | 75.48 | 1.00 |
| pilot_baseline | closed | 256 | 1024/1024 | 75.09 | 9937 | 381/916 | 17/39 | 6.43 | 0.00 | 59.40 | 0.79 |

\* fewer than 100 successful requests: p99 is under-sampled.

SLO: {'ttft_s': 0.5, 'tpot_s': 0.05, 'e2e_s': None}. Attainment is over *submitted* requests (failures and rejections count).
