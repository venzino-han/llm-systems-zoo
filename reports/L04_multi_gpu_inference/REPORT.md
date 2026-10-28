# L04 report: multi-GPU inference scaling (TP vs replicas)

Status: **interconnect micro-benchmark measured; TP/replica comparison not yet run.**

## Question
On the GPUs available, which layout (single GPU, tensor parallel, replicas) gives the best throughput,
per-GPU throughput and latency for a given model and workload, and why?

## Environment
* Host `kirc`: i9-12900KS (24 threads), 110 GiB RAM.
* GPU 0: NVIDIA RTX A6000, 48 GiB, sm_86. GPU 1: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB, sm_120.
* Link GPU0–GPU1: `PHB` (PCIe through the CPU host bridge), no NVLink (`nvidia-smi topo -m`, parsed by `llm_systems_zoo.hardware.gpu_topology`).
* torch 2.11.0+cu130, vLLM 0.23.0, NCCL bundled with torch, driver 580.95, CUDA 13.0.
* Both GPUs were shared with other processes during the micro-benchmark (see §Limits).

## Hypothesis
1. On a PCIe-host-bridge link, TP=2 adds a per-token communication cost that is small at batch 1
   (latency-bound, tens of µs per all-reduce) but comparable to the decode step at batch ≥ 64
   (bandwidth-bound, single-digit GB/s).
2. Therefore replicas beat TP on throughput for any model that fits on one card, and TP only pays
   off for models that do not fit, or at very low concurrency with NVLink (not available here).
3. Heterogeneous cards make TP run at the A6000's pace while the replica layout exposes two
   different per-GPU throughputs.

## Baseline
Single-GPU vLLM server (layout `single`) on GPU 0, same workload and load generator as the other layouts.

## Change
Layout only: `tp` (one server, `--tensor-parallel-size 2`) and `replica` (two servers behind
`inference/multi_gpu/proxy.py`). Model, workload, scheduler settings and client are held fixed.

## Method
* `comm_bench.py`: NCCL all_reduce/all_gather across 4 KiB–128 MiB, 50 iterations, bf16, two ranks.
* `bench_tp_vs_replica.py`: runs layouts one after another; each layout × load level is a harness run
  with records.jsonl, summary.json and a manifest recording model, dtype, TP size, replicas, GPU names
  and interconnect. Comparison via `serving.slo_analysis` (same metric definitions as L03).
* `minimal_tp.py`: numerical check that column/row-parallel sharding equals the dense block
  (max |Δ| = 1.8e-7 in fp32 on CPU, 2 gloo ranks) and a compute-vs-comm timing per token count.

## Results
### Interconnect (measured)
See `inference/multi_gpu/README.md` §6.1 and `results/comm_bench_nccl_a6000_blackwell_pcie.json`.
Summary: 4 KiB all-reduce ≈ 14.5 µs; all-reduce bus bandwidth plateaus ≈ 2.9 GB/s; TP=2 decode
communication floor ≈ 0.9 ms/token at batch 1 and ≈ 10 ms/token at batch 64 (interpolated) for a 28-layer, 3584-hidden model.

### TP vs replica (not measured)
No data yet. Commands are listed in `inference/multi_gpu/README.md` §5.

## Bottleneck analysis
From the micro-benchmark alone: the link, not the GPUs, is the constraint for TP on this host. The
all-reduce bandwidth ceiling (~3 GB/s) indicates host-memory staging rather than PCIe peer-to-peer,
which makes the per-layer all-reduce cost grow linearly with batch size right where continuous
batching wants to operate. This is the mechanism hypothesis 1 predicts; the serving comparison will
show whether it dominates end-to-end.

## Limits
* Micro-benchmark ran while other jobs used both GPUs; repeat on an idle host before quoting the numbers.
* Heterogeneous pair: TP results will not generalise to homogeneous NVLink nodes; the lab records the
  topology in every manifest so that readers can see this.
* No pipeline-parallel layout yet.

## Reproduction commands
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
  -m inference.multi_gpu.comm_bench --backend nccl --dtype bfloat16 --iters 50 --warmup 10 \
  --out reports/L04_multi_gpu_inference/results/comm_bench_nccl_a6000_blackwell_pcie.json
.venv/bin/python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 --dry-run
# full comparison: README §5
```

## Next experiments
1. Run the 0.5B and 7B comparisons from README §5 on exclusive GPUs (single / tp / replica; closed-loop 1–128, open-loop 2–16 req/s).
2. Add `pp2` as a fourth layout for the 7B model and compare bubbles vs all-reduce cost.
3. Profile one TP decode step (compute vs all-reduce vs idle) and join it with the serving numbers by run id.
4. Repeat the micro-benchmark with `NCCL_P2P_DISABLE=1` and `=0` to confirm the host-staging explanation.
