### Task granularity vs overhead (`runs/L07_ray_systems/20261029_020812_task_overhead_f61c6d612dc4`, 8 workers)

| g (s) | units | backend | wall (s) | speed-up | overhead/task (ms) |
|---|---|---|---|---|---|
| 1e-05 | 4000 | sequential | 0.0405 | 1 | - |
| 1e-05 | 4000 | multiprocessing | 0.0518 | 0.782 | 0.0117 |
| 1e-05 | 4000 | ray_tasks | 0.29 | 0.14 | 0.0713 |
| 1e-05 | 4000 | ray_chunked | 0.0143 | 2.83 | 0.148 |
| 0.0001 | 4000 | sequential | 0.401 | 1 | - |
| 0.0001 | 4000 | multiprocessing | 0.0609 | 6.58 | 0.00272 |
| 0.0001 | 4000 | ray_tasks | 0.377 | 1.06 | 0.0818 |
| 0.0001 | 4000 | ray_chunked | 0.0626 | 6.4 | 0.2 |
| 0.001 | 4000 | sequential | 4 | 1 | - |
| 0.001 | 4000 | multiprocessing | 0.507 | 7.89 | 0.00174 |
| 0.001 | 4000 | ray_tasks | 0.808 | 4.95 | 0.0771 |
| 0.001 | 4000 | ray_chunked | 0.523 | 7.65 | 0.362 |
| 0.01 | 400 | sequential | 4 | 1 | - |
| 0.01 | 400 | multiprocessing | 0.502 | 7.96 | 0.0059 |
| 0.01 | 400 | ray_tasks | 0.547 | 7.31 | 0.118 |
| 0.01 | 400 | ray_chunked | 0.504 | 7.94 | 0.51 |
| 0.1 | 40 | sequential | 4 | 1 | - |
| 0.1 | 40 | multiprocessing | 0.501 | 7.98 | 0.0251 |
| 0.1 | 40 | ray_tasks | 0.508 | 7.87 | 0.199 |
| 0.1 | 40 | ray_chunked | 0.502 | 7.96 | 0.288 |
| 1 | 8 | sequential | 8 | 1 | - |
| 1 | 8 | multiprocessing | 1 | 7.99 | 0.11 |
| 1 | 8 | ray_tasks | 1 | 7.98 | 0.371 |
| 1 | 8 | ray_chunked | 1 | 7.98 | 0.328 |

Break-even granularity (first speed-up ≥ 1): `{'multiprocessing': 0.0001, 'ray_tasks': 0.0001, 'ray_chunked': 1e-05}`

### Object store (`runs/L07_ray_systems/20261029_020916_object_store_109be30e8d24`)

| experiment | metric | value |
|---|---|---|
| arg passing | 100 MiB by value to 16 tasks (s) | 0.262 |
| arg passing | ray.put once + refs (s, incl. put) | 0.0212 |
| arg passing | speed-up | 12.4 |
| spilling | 600 MiB into a 256 MiB store: spilled (MiB) | 500 |
| spilling | put mean / max (ms) | 96.0 / 723.9 |
| spilling | get mean / max (ms) | 61.0 / 539.6 |
| backpressure | 120 x 10 MiB unbounded: wall (s) / spilled (MiB) | 2.79 / 1200.0 |
| backpressure | bounded window=8: wall (s) / spilled (MiB) | 0.17 / 0.0 |

### Placement groups and fractional GPUs (`runs/L07_ray_systems/20261029_022044_placement_454c4b7ba486`)

Placement groups: PACK ready in 0.96s on nodes ['23e264d8', '23e264d8'], SPREAD ready in 0.85s on nodes ['23e264d8', '23e264d8']

Fractional GPU: 3 actors each requesting `num_gpus=0.25` (logical total 0.75); nvidia-smi used MiB before → after: [37011, 92172] → [37011, 95374].

| actor | ray.get_gpu_ids() | CUDA_VISIBLE_DEVICES | torch allocated (MiB) | torch reserved (MiB) |
|---|---|---|---|---|
| 0 | ['1'] | 1 | 512 | 512 |
| 1 | ['1'] | 1 | 512 | 512 |
| 2 | ['1'] | 1 | 512 | 512 |

### Failure injection (`runs/L07_ray_systems/20261029_020457_failure_injection_c1b7a77692d6`)

| policy | outputs | unique | missing | duplicates | failed refs | crashes | restarts | recovery (s) | exactly-once | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 125 | 125 | 75 | 0 | 77 | 2 | 0 | - | False | 0.577 |
| at_least_once | 202 | 200 | 0 | 2 | 0 | 2 | 2 | 0.26, 0.26 | False | 1.8 |
| exactly_once | 200 | 200 | 0 | 0 | 0 | 2 | 2 | 0.26, 0.27 | True | 1.42 |

### Ray Train (`runs/L07_ray_systems/20261029_022311_ray_train_ebbf8ff5c9fe`)

2 workers, gpu=False: final loss 5.546, 31787 tokens/s/worker, wall 13.9s, checkpoint `checkpoint_2026-10-29_02-23-23.073570`; parameter digest on rank 0 `216450e68317`, identical across workers: True.

### Actor vs task (`runs/L07_ray_systems/20261029_020931_actor_vs_task_6c1576f299d1`, device=cpu)

| mode | inputs | wall (s) | inputs/s | first result (s) | model loads | load (s, mean) | peak GPU MiB |
|---|---|---|---|---|---|---|---|
| task_per_call | 64 | 30.6 | 2.09 | 4.47 | 64 | 1.77 | - |
| actor_pool | 64 | 9.65 | 6.63 | 4.65 | 2 | 3.49 | - |
| actor_batched | 64 | 6.85 | 9.35 | 5.31 | 2 | 3.75 | - |

### Ray Data batch inference (`runs/L07_ray_systems/20261029_021026_ray_data_batch_inference_e4c19e3e8005`, device=cpu)

| batch size | actors | slow preprocess (ms) | rows | wall (s) | rows/s |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 64 | 28.4 | 2.26 |
| 8 | 1 | 0 | 64 | 11.7 | 5.46 |
| 32 | 1 | 0 | 64 | 8.81 | 7.27 |
| 1 | 2 | 0 | 64 | 29.2 | 2.19 |
| 8 | 2 | 0 | 64 | 9.8 | 6.53 |
| 32 | 2 | 0 | 64 | 9.71 | 6.59 |
| 1 | 1 | 50 | 64 | 69.4 | 0.922 |
| 8 | 1 | 50 | 64 | 36.9 | 1.74 |
| 32 | 1 | 50 | 64 | 15.4 | 4.16 |
| 1 | 2 | 50 | 64 | 35.1 | 1.82 |
| 8 | 2 | 50 | 64 | 12.7 | 5.03 |
| 32 | 2 | 50 | 64 | 12.7 | 5.03 |

### Ray Serve bench_fake (`runs/L07_ray_systems/20261029_021512_ray_serve_bench_07733b9c7062`, device=cpu, fake=True)

| replicas | batch wait (ms) | concurrency | req/s | p50 (s) | p99 (s) | mean batch | error rate |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 17.5 | 0.0561 | 0.0814 | 1 | 0 |
| 1 | 0 | 4 | 19.7 | 0.203 | 0.205 | 1 | 0 |
| 1 | 0 | 16 | 19.6 | 0.811 | 0.84 | 1 | 0 |
| 2 | 0 | 1 | 17.5 | 0.0564 | 0.0689 | 1 | 0 |
| 2 | 0 | 4 | 34.5 | 0.107 | 0.204 | 1 | 0 |
| 2 | 0 | 16 | 36.9 | 0.403 | 0.672 | 1 | 0 |
| 1 | 10 | 1 | 14.6 | 0.067 | 0.101 | 1 | 0 |
| 1 | 10 | 4 | 39.8 | 0.101 | 0.174 | 2.78 | 0 |
| 1 | 10 | 16 | 90 | 0.154 | 0.213 | 5.31 | 0 |
| 2 | 10 | 1 | 14.7 | 0.067 | 0.0863 | 1 | 0 |
| 2 | 10 | 4 | 37.2 | 0.1 | 0.203 | 1.5 | 0 |
| 2 | 10 | 16 | 150 | 0.0926 | 0.177 | 6.3 | 0 |

### Ray Serve bench_tiny (`runs/L07_ray_systems/20261029_021644_ray_serve_bench_6899077fd656`, device=cpu, fake=False)

| replicas | batch wait (ms) | concurrency | req/s | p50 (s) | p99 (s) | mean batch | error rate |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 4.06 | 0.241 | 0.303 | 1 | 0 |
| 1 | 0 | 4 | 4.22 | 0.947 | 1.02 | 1 | 0 |
| 1 | 0 | 16 | 4.28 | 3.72 | 3.76 | 1 | 0 |
| 2 | 0 | 1 | 4.15 | 0.24 | 0.288 | 1 | 0 |
| 2 | 0 | 4 | 6.42 | 0.618 | 0.964 | 1 | 0 |
| 2 | 0 | 16 | 6.04 | 2.6 | 3.25 | 1 | 0 |
| 1 | 10 | 1 | 3.85 | 0.25 | 0.334 | 1 | 0 |
| 1 | 10 | 4 | 8.84 | 0.456 | 0.479 | 4 | 0 |
| 1 | 10 | 16 | 13.3 | 1.21 | 1.22 | 8 | 0 |
| 2 | 10 | 1 | 3.84 | 0.251 | 0.329 | 1 | 0 |
| 2 | 10 | 4 | 6.28 | 0.581 | 1.2 | 1.31 | 0 |
| 2 | 10 | 16 | 10.7 | 1.28 | 2.16 | 4.81 | 0 |
