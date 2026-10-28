# L04. Multi-GPU Inference Scaling

**Question.** For a given model and workload, which layout wins on the GPUs we have: tensor parallelism, pipeline parallelism, or independent replicas?
**Status.** Implemented (topology, NCCL micro-benchmark, minimal TP, TP-vs-replica orchestrator, CPU tests) · Measured: interconnect micro-benchmark only (§6). The TP/replica comparison has **not** been run yet; the exact commands are in §5.
**Hardware needed.** 2+ GPUs for the comparison; CPU for the minimal TP and tests.

## 1. What the technique is

A decoder layer is a few large matmuls plus attention. Multi-GPU inference chooses *what to split*:

### 1.1 Tensor parallelism (TP, Megatron-style)
Each weight matrix is sharded across the TP group. The QKV and MLP-up projections are
*column*-parallel (each rank computes a slice of the output, no communication); the attention-output
and MLP-down projections are *row*-parallel (each rank computes a partial sum that is **all-reduced**).
Result: **two all-reduces per layer** of a `[tokens, hidden]` tensor, whatever the model size.

```
memory per GPU   ≈ (weights + KV cache) / N
compute per token ≈ 1/N of the dense cost
comm per token    = 2 · L · all_reduce(B · H · bytes)    L layers, B sequences in the step, H hidden
```

During decode `B·H·bytes` is a few kilobytes, so the all-reduce is **latency**-bound, not
bandwidth-bound: `2L × all_reduce_latency` is a hard floor on TPOT. `comm_bench.py` measures that
latency on the actual link and `minimal_tp.py` is a runnable ~100-line version of the sharding that
checks numerical equivalence to the dense computation and times compute vs communication.

### 1.2 Pipeline parallelism (PP)
Layers are split into stages on different GPUs; only the activations at stage boundaries move
(point-to-point, small). PP tolerates slow links and lets a model span GPUs or nodes, but a single
request walks through the stages sequentially, so per-token latency does not drop and the pipeline
only fills with enough concurrent micro-batches. Bubbles show up as idle GPU time. vLLM supports PP
(`--pipeline-parallel-size`); this lab treats it as optional and adds it only when a model does not fit
with TP alone.

### 1.3 Replicas (data parallelism)
`N` independent single-GPU servers behind a router. No inter-GPU traffic, so throughput scales
linearly with `N` for any interconnect. Each replica holds a full copy of the weights, so the KV cache
per replica is `(GPU memory − weights)`, not `N×` that; a replica layout reaches a lower maximum batch
per model copy than TP does. Latency at low load equals the single-GPU latency.

### 1.4 Topology
`nvidia-smi topo -m` labels each GPU pair: `NV#` (NVLink, hundreds of GB/s, µs-scale latency),
`PIX`/`PXB` (same PCIe switch), `PHB` (through the CPU host bridge), `NODE`/`SYS` (across NUMA nodes or
sockets). On this host the two GPUs are `PHB`-connected (PCIe through the CPU, no NVLink), and they are
*different cards* (RTX A6000 48 GB sm_86, RTX PRO 6000 Blackwell 96 GB sm_120). `topology.py` prints
the matrix and what it implies.

### 1.5 The heterogeneous-GPU confound
TP shards memory evenly and advances at the pace of the slowest rank, so a mixed pair behaves like two
copies of the weaker card for TP. Replicas run at their own pace, so their aggregate throughput mixes
two different per-GPU numbers. Results on this host must be reported per device and never as
"per GPU" averages; `bench_tp_vs_replica.py` records the device names and the interconnect in every
run manifest and can refuse TP on mixed devices (`--homogeneous-check`).

## 2. When to use it

| Situation | Layout |
|---|---|
| Model fits on one GPU, throughput is the goal | replicas (linear scaling, no comm) |
| Model fits, but latency at low concurrency is the goal and GPUs have NVLink | TP (per-token compute splits N ways) |
| Model does not fit on one GPU, GPUs have NVLink | TP |
| Model does not fit, slow interconnect (PCIe / cross-node) | PP (or TP inside the node + PP across nodes) |
| Many short requests and a large KV footprint | TP inside a node gives a single large KV pool; replicas fragment it |
| Mixed GPU models | replicas; TP wastes the larger card |

## 3. Trade-offs

| Layout | Advantages | Disadvantages / cost |
|---|---|---|
| Tensor parallel | lowest per-token latency when the link is fast; one weight copy, pooled KV cache; serves models that do not fit | 2 all-reduces per layer; needs NVLink-class links; tied to the slowest rank; synchronised scheduling across ranks; harder to scale across nodes |
| Pipeline parallel | tolerates slow links; spans nodes; minimal communication volume | pipeline bubbles; no latency gain; needs concurrency to fill stages; uneven stage balance |
| Replicas / data parallel | linear throughput scaling; isolation (one crash ≠ outage); indifferent to topology; mixed hardware OK | N copies of weights; smaller KV cache per copy; needs a router and load balancing; no latency gain |
| Expert parallel (MoE, not in this lab) | scales parameter count with little per-token compute | all-to-all traffic, load imbalance |

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `topology.py` | prints `nvidia-smi topo -m`, classifies links, prints what the topology implies for TP vs replicas | CPU (needs nvidia-smi for live data) |
| `comm_bench.py` | all_reduce / all_gather latency and bandwidth across message sizes (NCCL on GPUs, gloo on CPU); derives the TP decode communication floor | 2+ GPUs or CPU |
| `minimal_tp.py` | educational column/row-parallel linears; equivalence check vs dense; compute vs all-reduce timing per token count | CPU (gloo) or GPUs |
| `proxy.py` | round-robin OpenAI-compatible proxy for the replica layout | CPU |
| `bench_tp_vs_replica.py` | launches single / TP / replica layouts one after another and drives each with the L03 load generator; `--dry-run` prints the commands | 2+ GPUs |
| `../../configs/l04_server_base.yaml` | base server config for the comparison | — |
| `../../tests/test_l04_multi_gpu.py` | 8 CPU tests: link classification, topology advice, round-robin, layout/plan rendering, dry run, minimal TP equivalence (2 gloo processes), comm floor | CPU |

## 5. How to run

CPU:

```bash
.venv/bin/python -m inference.multi_gpu.topology
.venv/bin/python -m inference.multi_gpu.minimal_tp --spawn 2 --backend gloo --hidden 256 --ffn 1024 --tokens 128
.venv/bin/python -m inference.multi_gpu.comm_bench --backend gloo --spawn 2 --smoke
.venv/bin/python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 --dry-run
.venv/bin/python -m pytest tests/test_l04_multi_gpu.py -q
```

GPU, executed for §6 (NCCL micro-benchmark, bf16, 50 iterations per size):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
    -m inference.multi_gpu.comm_bench --backend nccl --dtype bfloat16 --iters 50 --warmup 10 \
    --layers 28 --hidden 3584 --out reports/L04_multi_gpu_inference/results/comm_bench_nccl_a6000_blackwell_pcie.json
```

GPU, **not yet executed** (needs both GPUs exclusively; ~15 min for a 0.5B model, longer for 7B):

```bash
# 0.5B pilot: shows the mechanics; TP will lose to replicas here (tiny model, PCIe link)
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 \
    --model Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.4 --max-model-len 2048 \
    --layouts single,tp,replica --mode closed --concurrencies 1,8,32,128 \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct --workload benchmarks/workloads/short_qa.yaml \
    --report-dir reports/L04_multi_gpu_inference/results/qwen05b_short_qa

# 7B: the interesting case (decode memory-bound; fits on either card alone, so all three layouts run)
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 \
    --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8 --max-model-len 4096 \
    --layouts single,tp,replica --mode closed --concurrencies 1,8,32,128 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --workload benchmarks/workloads/short_qa.yaml \
    --report-dir reports/L04_multi_gpu_inference/results/qwen7b_short_qa

# same at fixed open-loop rates (queueing view)
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 \
    --model Qwen/Qwen2.5-7B-Instruct --layouts tp,replica --mode open --rates 2,8,16 --duration 30 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --report-dir reports/L04_multi_gpu_inference/results/qwen7b_short_qa_open
```

## 6. Results

### 6.1 Interconnect micro-benchmark (measured)

Run: `reports/L04_multi_gpu_inference/results/comm_bench_nccl_a6000_blackwell_pcie.json`. NCCL, bf16,
50 iterations per size after 10 warm-up, two processes (`torch.distributed.run`), GPU 0 = RTX A6000,
GPU 1 = RTX PRO 6000 Blackwell, link `PHB` (PCIe through the host bridge, no NVLink). Other jobs were
using both GPUs at the time, so treat the numbers as indicative.

| message | all_reduce (µs) | all_reduce algbw / busbw (GB/s) | all_gather (µs) | all_gather algbw (GB/s) |
|---|---|---|---|---|
| 4 KiB | 14.5 | 0.28 / 0.28 | 24.9 | 0.16 |
| 64 KiB | 36.6 | 1.79 / 1.79 | 39.3 | 1.67 |
| 1 MiB | 401.8 | 2.61 / 2.61 | 429.9 | 2.44 |
| 16 MiB | 6442.9 | 2.60 / 2.60 | 7223.9 | 2.32 |
| 128 MiB | 47010.3 | 2.86 / 2.86 | 52067.9 | 2.58 |

Reading:

* **Latency floor.** A 4 KiB all-reduce costs 15.6 µs. A Qwen2.5-7B-class model (28 layers,
  hidden 3584) needs 56 all-reduces per decoded token, so TP=2 adds about **0.87 ms per token at
  batch 1** on this link before any compute. Against a ~20 ms single-GPU decode step for a 15 GB bf16
  model on the A6000 that is a few percent; it is not what kills TP here.
* **Bandwidth ceiling.** The all-reduce bus bandwidth plateaus at 2.9 GB/s, roughly a
  tenth of what a PCIe 4.0 x16 link can move. NCCL is almost certainly staging through host memory
  because the two cards cannot do peer-to-peer across this host bridge. That is the number that
  matters at realistic batch sizes: at 64 sequences the per-layer message is 448 KiB; interpolating
  the measured latencies gives ≈183 µs per all-reduce and **≈10 ms of communication per
  token**, the same order as the decode step itself. TP=2 on this pair would spend about as long
  communicating as computing.
* **Conclusion for this host.** With a PCIe-host-bridge link and two different cards, replicas are the
  expected winner for any model that fits on one GPU; TP only makes sense for a model that does not.
  The §5 comparison commands are written to test exactly that, and the result must be read per device.

### 6.2 TP vs replica vs single (not yet measured)

Not run yet; it needs both GPUs free of other tenants. Expected artefacts once run:
`reports/L04_multi_gpu_inference/results/<model>_<workload>/compare.md|json|png` with throughput,
per-GPU throughput, TTFT/TPOT p50/p99 and goodput per layout and load level.

## 7. Limitations and next experiments

* The two GPUs on this host differ in architecture and memory and are joined by PCIe through the
  CPU. Any TP number measured here says more about the link and the slower card than about TP as a
  technique; the replica layout is the fair baseline on this machine.
* The comparison script runs layouts sequentially with identical workloads but does not yet pin
  CPU threads or isolate the GPUs from other tenants; run it on an idle machine.
* Pipeline parallelism is documented but not benchmarked (nothing here needs it to fit).
* Next: run the §5 commands on exclusive GPUs; add `--pipeline-parallel-size 2` as a fourth layout
  for the 7B model; profile a TP decode step (`torch.profiler` through vLLM's `--profile` or the
  minimal TP block on NCCL) to split compute / all-reduce / idle time per token.

## 8. References

* Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, 2019 (column/row parallel linears, two all-reduces per layer).
* Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC 2021 (TP + PP + DP trade-offs).
* Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023 (partitioning strategies and their latency/throughput costs).
* NVIDIA NCCL tests (`nccl-tests`) for the algbw/busbw definitions used in `comm_bench.py`.
* vLLM documentation: distributed serving (`--tensor-parallel-size`, `--pipeline-parallel-size`, data parallel).
