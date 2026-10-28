# Open-loop rate sweep: short_qa

- workload: benchmarks/workloads/short_qa.yaml
- backend: external (pilot_baseline)
- SLO: ttft=0.5,tpot=0.05
- tag: pilot, shared GPU, 0.10 util, A6000, enforce-eager, max_model_len 2048

| label | mode | offered_load | n_success/n_submitted | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | E2E p99 (s) | err | goodput req/s | SLO attain. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_baseline | open | 10 | 120/120 | 9.50 | 1221 | 26/42 | 7/7 | 1.58 | 0.00 | 9.50 | 1.00 |
| pilot_baseline | open | 20 | 240/240 | 18.15 | 2335 | 27/47 | 7/8 | 1.75 | 0.00 | 18.15 | 1.00 |
| pilot_baseline | open | 40 | 480/480 | 36.47 | 4802 | 33/69 | 8/11 | 2.20 | 0.00 | 36.47 | 1.00 |
| pilot_baseline | open | 80 | 960/960 | 58.82 | 7794 | 77/497 | 11/28 | 5.30 | 0.00 | 58.39 | 0.99 |
| pilot_baseline | open | 120 | 1440/1440 | 66.84 | 8899 | 3787/6235 | 23/36 | 9.81 | 0.00 | 8.49 | 0.13 |
| pilot_baseline | open | 160 | 1920/1920 | 57.44 | 7621 | 3308/16129 | 25/47 | 20.68 | 0.00 | 11.88 | 0.21 |

SLO: {'ttft_s': 0.5, 'tpot_s': 0.05, 'e2e_s': None}. Attainment is over *submitted* requests (failures and rejections count).
