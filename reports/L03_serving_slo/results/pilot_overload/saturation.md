# Overload at 140 req/s (2.0× 70): admission control

- workload: benchmarks/workloads/short_qa.yaml
- backend: external (pilot_baseline)
- SLO: ttft=0.5,tpot=0.05
- tag: pilot, shared GPU, 0.10 util, A6000, enforce-eager, max_model_len 2048

| label | mode | offered_load | n_success/n_submitted | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | E2E p99 (s) | err | goodput req/s | SLO attain. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| max_in_flight_128 | open | 140 | 554/1680 | 35.30 | 4716 | 201/582 | 21/26 | 5.63 | 0.67 (1126 rej) | 34.59 | 0.32 |
| max_in_flight_64 | open | 140 | 427/1680 | 28.54 | 3666 | 55/117 | 15/16 | 3.36 | 0.75 (1253 rej) | 28.54 | 0.25 |
| no_admission | open | 140 | 1680/1680 | 45.33 | 6023 | 7686/19260 | 39/44 | 24.31 | 0.00 | 7.61 | 0.17 |

SLO: {'ttft_s': 0.5, 'tpot_s': 0.05, 'e2e_s': None}. Attainment is over *submitted* requests (failures and rejections count).
