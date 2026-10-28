| strategy | scaling | world_size | global_batch | aggregate_tok_s | step_ms | peak_alloc_gib_max | speedup | efficiency | exposed_comm_s | comm_kernel_s_per_step |
|---|---|---|---|---|---|---|---|---|---|---|
| ddp | strong | 1 | 8 | 1.57e+04 | 108 | 3.76 | 1 | 1 | 0 | 0 |
| ddp | strong | 2 | 8 | 2.34e+04 | 73.3 | 2.09 | 1.47 | 0.735 | 0.0045 | 0.01 |
| fsdp2 | strong | 1 | 8 | 1.64e+04 | 103 | 3.41 | 1 | 1 |  | 0 |
| fsdp2 | strong | 2 | 8 | 9.11e+03 | 188 | 2.01 | 0.548 | 0.274 |  | 0.109 |
