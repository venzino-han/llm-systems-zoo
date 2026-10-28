# llm-systems-zoo

**Reproducible experiments across LLM training, inference, serving, and distributed systems.**

This repository demonstrates, with runnable code and measured results, how an LLM system behaves
from the application layer down to GPU memory, kernels and inter-GPU communication. Every topic is
framed as a question answered by an experiment (problem → hypothesis → baseline → change → measurement
→ interpretation → applicability), using only open models and open datasets from the Hugging Face Hub.

The exhibit is broad on purpose, and honest about depth: each item in the map below carries a status
label (Planned / Implemented / Measured / Reproduced) and each number carries a grade
(smoke / pilot / measurement). See `docs/decisions/0004-result-status-labels.md`.

## Representative experiments

All numbers below are **pilots on a shared GPU** (the machine was also hosting other projects'
servers; see `docs/hardware.md`). They show the method and the direction of each effect, not the
hardware's ceiling. Each links to the report with the run directory, manifest and raw records.

| Lab | Question | What the pilot showed | Report |
|---|---|---|---|
| **L01 Memory anatomy** | Where does GPU memory go, and why do estimates miss? | Analytic estimate within −4 %…+2 % of measured prefill peak over a 2×2 batch × length grid and 1–4 % for bf16 training once AdamW `foreach` temporaries are counted; LoRA on a 152k-vocab model is under-estimated by 21–27 %, traced to two extra fp32 logits tensors | [REPORT](reports/L01_memory_anatomy/REPORT.md) |
| **L02 KV cache** | When do paging and prefix caching pay off? | Simulator: 71.7 % internal fragmentation (contiguous) vs 1.0 % (paged, block 16), 2.4× shorter makespan. vLLM: warm prefill with shared prefixes 2.0–2.5× faster with prefix caching on, cached tokens = exactly the prefix, engine hit rate 85–95 %; no effect on decode throughput | [REPORT](reports/L02_kv_cache/REPORT.md) |
| **L03 Serving SLO** | How much load meets the SLO, and what happens past that? | Closed loop scales linearly to 128 in flight (75.5 req/s), then flattens; open loop: 80 req/s offered meets the SLO 99 % of the time, 120 req/s collapses to 13 %. At 140 req/s a client-side in-flight cap raises goodput from 7.6 to 34.6 req/s by rejecting 2/3 of requests; `max_num_seqs=8` cuts TPOT but saturates at 8 req/s | [REPORT](reports/L03_serving_slo/REPORT.md) |
| **L04 Multi-GPU** | TP or replicas on this hardware? | NCCL over PCIe host bridge: 4 KiB all-reduce 14.5 µs, bus bandwidth plateaus at 2.9 GB/s, implying a TP=2 decode floor of ≈0.9 ms/token for a 7B-class model. TP-vs-replica comparison is written and dry-run tested, pending exclusive GPUs | [REPORT](reports/L04_multi_gpu_inference/REPORT.md) |
| **L05 Training efficiency** | Memory vs throughput vs quality? | Eight single-GPU conditions on a 0.5B model (fp32 → bf16 AMP 1.8× tokens/s; pure bf16 2.7×; LoRA 2.8× at 60 % memory; QLoRA 2.2× at 50 %); checkpoint-resume is bitwise equal; 2-GPU DDP 1.47× vs FSDP2 0.55× on PCIe (sharding a model that fits is a measured loss) | [REPORT](reports/L05_training_efficiency/REPORT.md) |
| **L06 Post-training** | Rollout or learner: which is the bottleneck? | GRPO step: rollout 81 %, learner forward+backward 17 %, reward < 1 %; a dense format reward was optimised at the expense of correctness (GSM8K 0.30 → 0.07 in 25 steps) and is documented as a reward-design failure with fixes | [REPORT](reports/L06_post_training/REPORT.md) |
| **L07 Ray** | When does distribution pay, and is recovery correct? | Ray tasks break even at ≈100 µs of work (7× slower than a loop at 10 µs); `ray.put` references 12.4× faster than by-value fan-out; bounded `ray.wait` avoids 1.2 GB of spilling; with 2 worker crashes and 3 transient errors, naive loses 75 results, at-least-once duplicates 2, exactly-once with idempotency keys returns 200/200 | [REPORT](reports/L07_ray_systems/REPORT.md) |
| **L08 Inference optimisation** | Under which conditions does each optimisation help? | 0.5B: GPTQ-Int8 keeps perplexity (16.24 vs 16.26) and W4 Marlin decodes 1.65× faster, LLM.int8 is 5× slower; fp8 KV doubles capacity for +0.6 ppl; speculative decoding accepts 73–86 % of draft tokens yet is *slower* at this size (draft step ≈ target step); `torch.compile` 2.2× per decode step after 22–41 s cold compile; fused attention 45–50× faster than math SDPA at 2k tokens | [REPORT](reports/L08_inference_optimization/REPORT.md) |
| **L09 App → infra** | Do system choices move task success and latency? | RAG on 100 questions: EM 0.57 without reranking vs 0.48 with (+16 ms); model compute is 85 % of latency. Agent: verification turned 12 wrong answers into "unknown" and removed no correct ones for 12 ms/task; agent uses 3× the prompt tokens of the fixed pipeline | [REPORT](reports/L09_app_to_infra/REPORT.md) |

### Negative results we kept

Results that did not go the "expected" way are reported with the same care, because they are
where the trade-offs live: FSDP2 slower than DDP on two PCIe GPUs (L05); GRPO reward hacking (L06);
speculative decoding slower despite high acceptance (L08); vLLM fp8 producing garbage perplexity on an
Ampere card but fine on Blackwell (L08); reranking lowering exact match on yes/no questions (L09);
Ray tasks slower than a plain loop below 100 µs (L07); a narrow scheduler collapsing goodput (L03).

## Technique map

Each lab README documents the techniques it covers: what they are, when to use them, advantages,
disadvantages and the trade-offs, followed by the code, how to run it and the results.
`docs/technique_catalog.md` is the one-page index of all of them.

| Track | Techniques (each documented with what / when / pros / cons / trade-offs) | Depth reached | Status | Lab |
|---|---|---|---|---|
| Fundamentals | memory anatomy (weights, grads, optimizer, activations, KV, overhead), FLOPs, arithmetic intensity, roofline, prefill vs decode | Explain · Implement · Benchmark | Measured (pilot) | [L01](fundamentals/README.md) |
| KV cache | paged allocation, block tables, ref-counting, prefix caching, LRU eviction, copy-on-write, capacity model, GQA/MQA, KV dtype | Explain · Implement · Benchmark | Measured (sim) · Measured (pilot) | [L02](inference/kv_cache/README.md) |
| Serving | continuous batching, chunked prefill, scheduler limits, admission control, open vs closed loop, SLO goodput, overload | Explain · Implement · Benchmark · Operate (overload) | Measured (pilot) | [L03](serving/README.md) |
| Multi-GPU inference | tensor / pipeline parallelism, replicas, topology, NCCL bandwidth, minimal TP layers | Explain · Implement · Benchmark (interconnect) | Implemented · TP-vs-replica pending | [L04](inference/multi_gpu/README.md) |
| Training efficiency | AMP, accumulation, activation checkpointing, packing, LoRA, QLoRA, checkpoint resume | Explain · Implement · Benchmark | Measured (pilot) | [L05](training/README.md) |
| Distributed training | DDP, FSDP2/ZeRO, comm overlap, strong vs weak scaling | Explain · Implement · Benchmark | Measured (2-GPU smoke) | [L05](training/distributed/) |
| Post-training | SFT, DPO (reference precompute), GRPO (verifiable rewards), rollout/learner decomposition, weight sync | Explain · Implement · Benchmark | Measured (pilot) · vLLM rollouts pending | [L06](post_training/README.md) |
| Inference optimisation | INT8/INT4 (bnb, AWQ, GPTQ, Marlin), fp8 weights/KV, speculative decoding (minimal + vLLM n-gram/draft), torch.compile, CUDA graphs, SDPA/FlashInfer/FA3, Triton kernels | Explain · Implement · Benchmark | Measured (pilot) · 7B table pending | [L08](inference/README.md) |
| Distributed platform | Ray tasks vs actors, object store / spilling / backpressure, placement groups, fault tolerance & exactly-once, Ray Data, Ray Train, Ray Serve | Explain · Implement · Benchmark · Operate (failure injection) | Measured | [L07](distributed/README.md) |
| Applications | dense retrieval, reranking, cacheable prompt layout, tool-calling agent, claim verification, span tracing | Explain · Implement · Benchmark | Measured (pilot) | [L09](applications/README.md) |
| Observability | run manifests, phase timers, torch.profiler traces, Perfetto export | Implement | Implemented | harness, L01, L09 |
| Infrastructure | container image, CPU CI | Implement | Dockerfile not validated | `docker/`, `.github/` |

## Quickstart

```bash
git clone https://github.com/venzino-han/llm-systems-zoo && cd llm-systems-zoo
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"              # harness + CPU tests, no GPU needed
pytest                                  # CPU tests: simulators, calculators, metrics, minimal implementations

# CPU-only demos (no model download)
python fundamentals/memory_calculator.py --help
python inference/kv_cache/simulate_workloads.py --help
python distributed/ray_core/task_overhead.py --smoke

# GPU labs: install the extra for the track you want and follow the lab README
uv pip install -e ".[inference]"        # vLLM-based labs (L02, L03, L04, L08)
uv pip install -e ".[training]"         # L05, L06
uv pip install -e ".[ray]"              # L07
uv pip install -e ".[apps]"             # L09
```

Hardware needed per lab is stated at the top of each lab README; the reference environment the
published pilots ran on is in `docs/hardware.md` (one RTX A6000 48 GB + one RTX PRO 6000 Blackwell
96 GB, PCIe only).

## Repository layout

```
llm_systems_zoo/        shared harness: run manifests, metrics (TTFT/TPOT/goodput), timers, hardware inventory
fundamentals/           L01  memory & compute anatomy (calculators, measurement, profiler)
inference/
  kv_cache/             L02  paged allocator simulator, capacity model, prefix-caching benchmark
  multi_gpu/            L04  TP vs replicas, NCCL bandwidth, minimal tensor-parallel layers
  quantization/         L08  INT8/INT4/AWQ/GPTQ quality–memory–throughput table
  speculative_decoding/ L08  minimal speculative decoder + vLLM speculative config
  compilation/          L08  torch.compile / CUDA graphs warm-up vs steady state
  kernels/              L08  Triton softmax/RMSNorm, SDPA backend comparison
serving/                L03  workloads, open/closed-loop load generator, SLO goodput, saturation curves
training/               L05  AMP, accumulation, checkpointing, packing, LoRA/QLoRA; DDP/FSDP scaling
post_training/          L06  SFT, DPO, GRPO with rollout / reward / learner time decomposition
distributed/            L07  Ray Core/Data/Train/Serve experiments incl. failure injection
applications/           L09  RAG + agent with verification, traced down to serving metrics
benchmarks/             shared workload definitions and run aggregation
configs/                YAML configs per experiment
reports/                per-lab REPORT.md + curated results (summary.json, figures)
docs/                   architecture, benchmark protocol, decisions (ADRs), competency map, technique catalog
tests/                  CPU tests (CI) + gpu/slow-marked tests
docker/                 reference container (not yet validated)
```

## Competency → evidence

| Competency | Evidence | Lab |
|---|---|---|
| GPU memory analysis | calculator vs measured allocated/reserved, profiler traces | [L01](fundamentals/README.md) |
| KV-cache capacity and paging | paged-allocator simulator, prefix-cache measurements | [L02](inference/kv_cache/README.md) |
| Efficient LLM serving | open/closed-loop load, SLO goodput, saturation and overload curves | [L03](serving/README.md) |
| Multi-GPU inference | TP vs replicas per-GPU throughput, NCCL bandwidth, topology | [L04](inference/multi_gpu/README.md) |
| Training efficiency & distributed training | memory–throughput sweep, DDP/FSDP scaling, checkpoint-resume check | [L05](training/README.md) |
| Post-training systems | SFT/DPO/GRPO with rollout/learner split, GSM8K evaluation | [L06](post_training/README.md) |
| Ray distributed execution | task/actor overhead, object store, placement, exactly-once recovery | [L07](distributed/README.md) |
| Inference optimisation | quantisation, speculative decoding, compilation, kernels with correctness checks | [L08](inference/README.md) |
| Research-to-product implementation | RAG + agent verification with cross-layer traces | [L09](applications/README.md) |

Full table with reader paths and explicit non-claims: `docs/competency_map.md`.

## How numbers are produced

Every run writes a manifest (hardware, software versions, model revision, workload, execution
settings, git commit, config hash) before it writes a metric; metrics are defined once in
`llm_systems_zoo/metrics.py`; failures stay in the denominator; one variable changes at a time.
The full protocol is in `docs/benchmark_protocol.md`.

## Limitations and roadmap

* Pilots ran on shared, heterogeneous, PCIe-connected GPUs with other projects' servers resident; exclusive re-measurements are queued (`docs/roadmap.md`).
* Not yet run: TP-vs-replica (L04), 7B quantisation table and `max-autotune` (L08), DPO reference A/B and vLLM-colocated GRPO rollouts (L06), Ray Serve in front of vLLM (L07).
* No multi-node experiments yet. No production operation is claimed.
* Models are 0.5B–7B; behaviour at 70B+ is not extrapolated.

Phases, release criteria and open decisions: `docs/roadmap.md`. Design decisions: `docs/decisions/`.

## License

MIT. Models and datasets keep their own licences (listed in `llm_systems_zoo/registry.py`).
