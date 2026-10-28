# Competency → evidence map

Each row links a claim a CV might make to the code and the measured evidence that backs it. The
**Status** column uses the labels from `docs/decisions/0004-result-status-labels.md`; a claim is only
as strong as its status.

| Competency claimed | Evidence | Lab | Status |
|---|---|---|---|
| GPU memory analysis | analytical calculator vs measured allocated/reserved, profiler traces | [L01](../fundamentals/README.md) | see lab README |
| KV-cache capacity and paging | paged-allocator simulator, prefix-cache hit-rate and TTFT measurements | [L02](../inference/kv_cache/README.md) | see lab README |
| Efficient LLM serving | open/closed-loop load generator, SLO goodput, saturation curve, overload behaviour | [L03](../serving/README.md) | see lab README |
| Multi-GPU inference | TP vs replica throughput per GPU, NCCL bandwidth, topology-aware interpretation | [L04](../inference/multi_gpu/README.md) | see lab README |
| Training efficiency & distributed training | memory–throughput sweep (AMP, checkpointing, packing, LoRA/QLoRA), DDP/FSDP scaling, checkpoint-resume verification | [L05](../training/README.md) | see lab README |
| Post-training systems | SFT/DPO/GRPO with rollout / reward / learner time decomposition and GSM8K evaluation | [L06](../post_training/README.md) | see lab README |
| Ray distributed execution | task vs actor overhead, object store and spilling, placement, failure injection with exactly-once results | [L07](../distributed/README.md) | see lab README |
| Inference optimisation | quantisation quality/throughput table, speculative decoding acceptance, torch.compile, Triton and attention kernels | [L08](../inference/README.md) | see lab README |
| Research-to-product implementation | RAG + agent with verification, per-stage traces tied to serving metrics | [L09](../applications/README.md) | see lab README |

## Reader paths

| Reader | Start here | Then |
|---|---|---|
| Recruiter | root README (purpose, scope, status map) | one lab REPORT.md |
| ML engineer interviewer | a lab's code, config and tests | the reproduction commands in its REPORT.md |
| ML systems / infra interviewer | L02, L03, L04, L07 bottleneck analyses | profiler traces and failure tests |
| ML researcher interviewer | L05, L06, L09 ablations and evaluation sets | limitations sections |
| Collaborator | `CONTRIBUTING.md`, `docs/benchmark_protocol.md` | the harness in `llm_systems_zoo/` |

## What is *not* claimed

* Production operation of any of these systems.
* Multi-node results (none were run).
* NVLink-class scaling (the reference machine is PCIe-only with heterogeneous GPUs).
