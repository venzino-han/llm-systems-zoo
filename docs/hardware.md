# Hardware tiers and the reference environment

## What each tier can and cannot prove

| Tier | Possible work | What it cannot show |
|---|---|---|
| CPU / laptop | simulators, calculators, analysis, CI, API mocks, minimal implementations with gloo | any GPU performance claim |
| 1 GPU | KV-cache, serving SLO, PEFT, small post-training, profiling, quantisation, speculative decoding | multi-GPU scaling |
| 2+ GPUs (one node) | TP vs replicas, DDP/FSDP, GPU placement, NCCL bandwidth | inter-node network bottlenecks |
| Multi-node | network bottlenecks, distributed recovery, cluster scheduling | (cost and operational complexity rise sharply) |

Experiments run on the `smoke → pilot → measurement` ladder (see `docs/benchmark_protocol.md`).
Before a measurement run: conditions × repeats × expected duration × GPUs = the GPU-hour budget.

## Reference environment (where the published pilots ran)

Collected by `python -m llm_systems_zoo.hardware` on 2026-10-29.

| Item | Value |
|---|---|
| CPU | Intel Core i9-12900KS, 24 threads |
| RAM | 110 GiB |
| GPU 0 | NVIDIA RTX A6000, 48 GB, sm_86 (Ampere) |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, sm_120 (Blackwell) |
| Interconnect | PCIe host bridge (PHB) between the GPUs; **no NVLink** |
| Driver / CUDA | 580.95.05 / CUDA 13.0 |
| Software | Python 3.10.12, torch 2.11.0+cu130, transformers 5.7.0, vLLM 0.23.0, TRL 1.9.2, PEFT 0.14.0, Ray 2.53.0, Triton 3.6.0 |

Two properties of this machine shape how results must be read:

1. **The GPUs are heterogeneous.** Tensor parallelism and DDP across them are bounded by the slower
   card and by PCIe. Multi-GPU results here demonstrate the *method* and PCIe-bound behaviour, not
   NVLink-class scaling. The reports say so explicitly.
2. **Blackwell (sm_120) is new.** Some kernels (bitsandbytes variants, some quantisation backends)
   may not support it yet. When a backend fails, the lab records the exact error instead of silently
   switching cards.

## Shared-GPU caveat for the published pilots

When the pilots were run, the machine was also hosting two resident vLLM servers belonging to
other projects (about 27 GB resident on GPU 0 and 92 GB on GPU 1) plus a small service. The labs
therefore had roughly 16 GB free on the A6000 and 5 GB free on the Blackwell card, shared among
all labs running concurrently. Consequences:

* vLLM-based pilots ran with `gpu_memory_utilization` ≈ 0.10 on the A6000 (small KV cache, low
  concurrency ceiling), 0.5B–1.5B models, `enforce_eager`, `max_model_len` ≤ 2048.
* Throughput and latency numbers are noisy and capacity-limited; they demonstrate the method and
  the direction of effects, not the machine's capability.
* Every such table is graded `pilot, shared GPU`; the exclusive-GPU re-measurements are listed in
  `docs/roadmap.md`.
