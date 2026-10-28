# Server config comparison: short_qa @ open 16.0

- workload: benchmarks/workloads/short_qa.yaml
- backend: vllm (max_num_seqs_8)
- SLO: ttft=0.5,tpot=0.05
- tag: pilot, shared GPU, 0.10 util, A6000, enforce-eager, max_model_len 2048

| label | mode | offered_load | n_success/n_submitted | req/s | out tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | E2E p99 (s) | err | goodput req/s | SLO attain. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | open | 16 | 192/192 | 15.01 | 1924 | 43/69 | 11/12 | 2.81 | 0.00 | 15.01 | 1.00 |
| max_num_seqs_8 | open | 16 | 192/192 | 7.96 | 1020 | 5225/11688 | 7/8 | 12.73 | 0.00 | 0.37 | 0.05 |

SLO: {'ttft_s': 0.5, 'tpot_s': 0.05, 'e2e_s': None}. Attainment is over *submitted* requests (failures and rejections count).
