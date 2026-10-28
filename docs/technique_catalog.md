# Technique catalog

A one-page reference for every technique the labs cover: what it is, when it pays off, what it
costs, and where the code and evidence live. Each lab README goes deeper (cost model, equations,
measured results). Status labels follow `docs/decisions/0004-result-status-labels.md`.

---

## Fundamentals (L01 · `fundamentals/`)

### Memory anatomy of a training step and of inference
**What.** Accounting of the six memory consumers: weights, gradients, optimizer states, activations,
KV cache, runtime overhead (CUDA context, allocator reserve, workspaces). For AdamW in mixed precision
a parameter costs about 16 bytes during training (2 weight + 2 grad + 4 master + 8 optimizer) versus 2 bytes at bf16 inference.
**Use when.** Sizing a job before launching it; explaining OOMs; choosing between PEFT and full fine-tuning.
**Pros.** Cheap, predictive, catches impossible configurations before any GPU time is spent.
**Cons.** Activation and overhead terms are approximate; allocator fragmentation and engine workspaces are not analytically known.
**Trade-off.** Precision of the estimate vs effort: the simple model is within tens of percent, the exact number needs a measurement.

### FLOPs, arithmetic intensity and the roofline
**What.** Training ≈ 6·N·D FLOPs; prefill is compute-bound (many tokens per weight read), decode is
memory-bandwidth-bound (one token per weight read). Arithmetic intensity = FLOPs / bytes moved; compare to the GPU's ridge point.
**Use when.** Predicting whether an optimisation can help: bandwidth-bound decode benefits from quantisation and batching, compute-bound prefill from better kernels and lower precision math.
**Pros.** A single ratio explains most inference speed-ups.
**Cons.** Nominal peak numbers are rarely reached; attention and non-matmul ops break the simple model at long context.
**Trade-off.** Modelling fidelity vs simplicity; the lab reports the estimate–measurement gap instead of hiding it.

---

## KV cache (L02 · `inference/kv_cache/`)

### Paged KV cache (block allocation)
**What.** Store K/V in fixed-size blocks with a per-sequence block table instead of a contiguous max-length buffer (vLLM's PagedAttention).
**Use when.** Many concurrent sequences with unknown, varying lengths; memory is the limit on batch size.
**Pros.** Near-zero external fragmentation, internal waste bounded by one block per sequence, enables sharing and swapping.
**Cons.** Indirection in the attention kernel; block-table metadata; a block size to tune.
**Trade-off.** Small blocks waste less memory but cost more metadata and kernel lookups; large blocks the reverse.

### Prefix caching (shared prefixes)
**What.** Hash full blocks by their token content so identical prefixes (system prompts, few-shot examples, shared documents) reuse K/V across requests.
**Use when.** Requests share long common prefixes; TTFT matters; the cache is warm.
**Pros.** Skips prefill compute for the shared part, cuts TTFT and prefill GPU time.
**Cons.** Hashing/lookup overhead; no benefit for unique prompts; cold cache gives none; eviction policy matters under memory pressure.
**Trade-off.** Memory held by cached blocks vs memory for new sequences.

### KV-cache dtype / quantisation, eviction and offload
**What.** Store K/V in fp8/int8 to double capacity; evict or offload to CPU under pressure.
**Use when.** Capacity (max concurrent tokens) binds before compute.
**Pros.** More sequences or longer contexts per GPU.
**Cons.** Possible quality loss; dequantisation in the kernel; offload traffic over PCIe.
**Trade-off.** Capacity vs accuracy and kernel complexity.

### GQA / MQA
**What.** Fewer K/V heads than query heads so the KV cache per token shrinks by n_heads / n_kv_heads.
**Use when.** Designing or choosing a model for long-context serving.
**Pros.** Linear reduction of KV memory and bandwidth.
**Cons.** Baked into the architecture; slight quality cost relative to MHA at the same size.
**Trade-off.** Model quality vs serving cost, decided at training time.

---

## Serving (L03 · `serving/`)

### Continuous (iteration-level) batching
**What.** Add and remove sequences from the running batch at every decode step instead of waiting for the whole batch to finish.
**Use when.** Any multi-user generation workload with variable output lengths.
**Pros.** Keeps the GPU busy, removes head-of-line blocking by long outputs, raises throughput several-fold over static batching.
**Cons.** Scheduler complexity; per-step overhead; interaction with prefill of new requests.
**Trade-off.** Throughput vs per-request latency as the batch grows (TPOT rises with batch size).

### Chunked prefill
**What.** Split long prompts into chunks and interleave them with decode steps of other sequences.
**Use when.** Mixed workloads where long prompts would otherwise stall decoding users (TPOT spikes).
**Pros.** Smoother TPOT, better GPU utilisation by mixing compute-bound and bandwidth-bound work.
**Cons.** Longer TTFT for the chunked request; a chunk size to tune.
**Trade-off.** TTFT of long requests vs TPOT of everyone else.

### Scheduler limits and admission control
**What.** `max_num_seqs`, `max_num_batched_tokens`, queue length limits, client-side in-flight limits, time-outs and cancellation.
**Use when.** Offered load can exceed capacity; SLOs are defined.
**Pros.** Bounded queueing delay; predictable tail latency; graceful degradation instead of collapse.
**Cons.** Rejected requests; needs a policy for who gets rejected.
**Trade-off.** Goodput under an SLO vs raw completed-request count.

### Open-loop vs closed-loop load, SLO goodput
**What.** Closed-loop fixes concurrency; open-loop fixes the arrival rate and exposes queue growth. Goodput counts only requests that met the SLO, over submitted requests.
**Use when.** Any capacity claim.
**Pros.** Open-loop reveals saturation and overload; goodput prevents "fast but useless" results.
**Cons.** Open-loop runs are longer and noisier; p99 needs hundreds of requests per point.
**Trade-off.** Measurement cost vs how much the number means.

### Disaggregated prefill/decode (concept only in v0.1)
**What.** Run prefill and decode on separate GPU pools and ship the KV cache between them.
**Use when.** Large deployments where TTFT and TPOT SLOs conflict on shared hardware.
**Pros.** Independent scaling of the two phases.
**Cons.** KV transfer bandwidth, operational complexity.
**Trade-off.** Hardware efficiency vs system complexity.

---

## Multi-GPU inference (L04 · `inference/multi_gpu/`)

### Tensor parallelism (TP)
**What.** Split each weight matrix across GPUs; all-reduce activations after attention and MLP in every layer.
**Use when.** The model does not fit one GPU, or per-request latency must drop and the interconnect is fast (NVLink).
**Pros.** Lower latency per request; pools memory for weights and KV.
**Cons.** Two all-reduces per layer per step; collapses on PCIe-only or heterogeneous GPUs; per-GPU throughput falls.
**Trade-off.** Latency vs aggregate throughput and communication cost.

### Pipeline parallelism (PP)
**What.** Assign contiguous layers to different GPUs; micro-batches flow through.
**Use when.** Very large models across slower links; TP degree is already maxed.
**Pros.** Point-to-point communication only; works over weaker interconnects.
**Cons.** Pipeline bubbles; higher latency; needs enough concurrent requests to fill the pipeline.
**Trade-off.** Interconnect tolerance vs idle time.

### Replicas (data parallel serving)
**What.** Independent copies of the model, each on its own GPU, behind a router.
**Use when.** The model fits one GPU and throughput, not latency, is the goal.
**Pros.** No inter-GPU traffic; linear throughput scaling; simple failure isolation.
**Cons.** No latency gain; weights duplicated; routing/load-balancing needed; prefix caches are not shared.
**Trade-off.** Throughput per GPU vs latency and memory pooling.

---

## Training efficiency (L05 · `training/`)

### Mixed precision (AMP / bf16)
**What.** Compute in bf16/fp16 with fp32 master weights and accumulation.
**Use when.** Almost always on Ampere+ GPUs.
**Pros.** 2–3× matmul throughput, halves activation memory.
**Cons.** fp16 needs loss scaling; some ops must stay fp32; subtle numeric differences.
**Trade-off.** Speed and memory vs numerical headroom.

### Gradient accumulation
**What.** Sum gradients over K micro-batches before one optimizer step to emulate a larger batch.
**Use when.** The target global batch does not fit in memory.
**Pros.** Decouples batch size from memory.
**Cons.** No speed-up; K times more forward/backward per step; DDP must skip gradient sync for K−1 micro-steps.
**Trade-off.** Memory vs wall-clock per optimizer step.

### Activation (gradient) checkpointing
**What.** Drop intermediate activations in the forward pass and recompute them in backward.
**Use when.** Activations dominate memory (long sequences, large batch).
**Pros.** Activation memory drops from O(layers) to roughly O(√layers) or per-layer boundaries.
**Cons.** Roughly 25–35 % more compute per step.
**Trade-off.** Memory vs throughput; often lets a larger batch recover the loss.

### Sequence packing
**What.** Concatenate several samples into one fixed-length row with per-sample position ids and a block-diagonal (or varlen) attention mask, so no tokens are padding.
**Use when.** Sample lengths vary widely (chat/SFT data).
**Pros.** Effective tokens/s rises sharply; fewer wasted FLOPs.
**Cons.** Needs correct masking (cross-sample attention leakage otherwise); changes batch semantics; needs varlen attention for full benefit.
**Trade-off.** Implementation care vs utilisation.

### LoRA / QLoRA
**What.** Train low-rank adapters on frozen weights; QLoRA additionally stores the frozen base in 4-bit NF4.
**Use when.** Limited memory; many task-specific variants; fast iteration.
**Pros.** Gradients and optimizer states shrink to the adapter size; QLoRA fits 7B+ models on one consumer GPU.
**Cons.** Smaller capacity than full fine-tuning; QLoRA dequantisation slows each step; base-model quantisation error.
**Trade-off.** Memory and cost vs peak quality and step speed.

---

## Distributed training (L05 · `training/distributed/`)

### DDP
**What.** Every GPU holds the full model; gradients are all-reduced (bucketed, overlapped with backward).
**Use when.** The model and optimizer fit on one GPU.
**Pros.** Simple, well-understood, near-linear scaling on good interconnects.
**Cons.** Memory is not shared; communication volume = full gradient per step.
**Trade-off.** Simplicity vs memory ceiling.

### FSDP / ZeRO (stages 1–3)
**What.** Shard optimizer states (1), gradients (2) and parameters (3) across GPUs; all-gather parameters per layer during compute.
**Use when.** The model or its optimizer states do not fit per GPU.
**Pros.** Memory per GPU scales down with the number of GPUs; same math as DDP.
**Cons.** More communication (all-gather + reduce-scatter per layer), sensitive to interconnect, harder checkpointing.
**Trade-off.** Memory vs communication and complexity.

### Communication–computation overlap, strong vs weak scaling
**What.** Bucketing and async collectives hide communication behind backward compute. Strong scaling fixes total work; weak scaling fixes work per GPU.
**Use when.** Reporting any scaling number.
**Pros.** Separates "faster per step" from "more throughput".
**Cons.** Each needs its own experiment.
**Trade-off.** None; they answer different questions.

---

## Post-training (L06 · `post_training/`)

### SFT
**What.** Supervised next-token training on instruction/response pairs with loss masked to the response.
**Use when.** Teaching format, style and basic task behaviour; always the first stage.
**Pros.** Stable, cheap, strong baseline.
**Cons.** Needs demonstrations; cannot optimise a non-differentiable objective.
**Trade-off.** Data cost vs control.

### DPO
**What.** Optimise a preference objective directly from chosen/rejected pairs against a frozen reference model.
**Use when.** Preference data exists and no RL infrastructure is wanted.
**Pros.** No reward model, no rollouts; one extra forward (reference) per step; stable.
**Cons.** Offline: cannot exploit new samples; sensitive to β and data quality; reference-model memory/compute.
**Trade-off.** Simplicity vs the ceiling of online RL.

### GRPO (and online RL in general)
**What.** Sample a group of completions per prompt, score them with a (verifiable) reward, normalise within the group, and update the policy with a clipped objective.
**Use when.** A reward can be computed (math answers, unit tests, format checks) and the goal is reasoning or exact behaviour.
**Pros.** No value network; uses verifiable rewards; improves beyond demonstrations.
**Cons.** Rollouts dominate time; generation length and group size multiply cost; instability.
**Trade-off.** Learner time vs rollout time: this lab measures the split and what moves it (vLLM rollouts, group size, length caps, separate rollout GPUs, weight-sync frequency).

---

## Inference optimisation (L08 · `inference/`)

### Weight-only quantisation (INT8 / INT4: bitsandbytes, AWQ, GPTQ)
**What.** Store weights in 8 or 4 bits (per-channel/per-group scales), dequantise on the fly or use fused kernels.
**Use when.** Memory-bound decode at small batch; fitting a larger model per GPU.
**Pros.** 2–4× smaller weights; decode speed-up when bandwidth-bound; small quality loss at INT8 and with calibrated INT4.
**Cons.** Kernel quality decides whether it is faster at all (naive dequant can be slower); quality loss grows with aggressive groups/bits; calibration data needed for AWQ/GPTQ.
**Trade-off.** Memory and decode latency vs quality and prefill/large-batch throughput.

### FP8 (weights and/or activations)
**What.** 8-bit floating point compute on Hopper/Blackwell tensor cores.
**Use when.** Compute-bound phases (prefill, large batch) on supporting GPUs.
**Pros.** Near-2× matmul throughput, small quality loss.
**Cons.** Hardware-specific; calibration/scaling choices.
**Trade-off.** Portability vs speed.

### Speculative decoding (draft model, n-gram, Medusa/EAGLE-style heads)
**What.** Propose K tokens cheaply, verify them with one target forward; accept the longest matching prefix (exact under greedy; distribution-preserving with rejection sampling).
**Use when.** Small batch, bandwidth-bound decode, a draft that agrees often (same family or repetitive text for n-gram).
**Pros.** Lower per-token latency without changing outputs.
**Cons.** Draft cost and verification waste at low acceptance; benefit vanishes at large batch when decode becomes compute-bound; extra memory.
**Trade-off.** Latency at small batch vs throughput at large batch.

### torch.compile and CUDA graphs
**What.** Trace the model into fused kernels; capture the decode step as a graph to remove launch overhead.
**Use when.** Steady-state serving with static shapes; small models where launch overhead dominates.
**Pros.** Lower per-step overhead, kernel fusion.
**Cons.** Compile/warm-up time; recompilation on shape changes; memory for graphs.
**Trade-off.** Start-up cost vs steady-state speed.

### Attention kernels and custom Triton kernels
**What.** FlashAttention-style fused, IO-aware attention; FlashInfer for paged decode; hand-written Triton kernels for softmax/RMSNorm/fusion.
**Use when.** Long sequences (attention memory and time), or profiler shows a non-matmul kernel dominating.
**Pros.** O(N) memory for attention; fewer memory round-trips; large speed-ups for the fused op.
**Cons.** Hardware/dtype constraints; correctness risk; a kernel win may be invisible end-to-end.
**Trade-off.** Engineering effort vs end-to-end gain (always validate against the profile).

---

## Distributed execution with Ray (L07 · `distributed/`)

### Tasks vs actors
**What.** Stateless remote functions vs stateful remote objects (e.g., a model loaded once).
**Use when.** Tasks for independent, coarse work; actors when start-up (model load) is expensive or state must persist.
**Pros.** Simple parallelism; actors amortise load time and allow batching.
**Cons.** Per-task scheduling overhead (sub-millisecond to milliseconds) dominates fine-grained work; actors need lifecycle management.
**Trade-off.** Task granularity vs overhead; statefulness vs elasticity.

### Object store, zero-copy, spilling, backpressure
**What.** Shared-memory store for large objects; references instead of copies; spill to disk when full; bound in-flight work with `ray.wait`.
**Use when.** Large arrays/tensors flow between stages.
**Pros.** Avoids serialisation copies; decouples producers and consumers.
**Cons.** Spilling is slow; unbounded submission exhausts memory.
**Trade-off.** Memory vs throughput; must be measured.

### Placement groups and GPU resource requests
**What.** Reserve bundles of resources, PACK or SPREAD them; fractional `num_gpus`.
**Use when.** Co-locating or isolating workers.
**Pros.** Deterministic placement.
**Cons.** Resources are logical; they do not limit real GPU memory.
**Trade-off.** Scheduling control vs a false sense of isolation (the lab shows the gap).

### Fault tolerance
**What.** Task retries, actor restarts, at-least-once execution.
**Use when.** Long jobs on preemptible or failing workers.
**Pros.** Automatic recovery.
**Cons.** Duplicates without idempotency keys; recovery time; lost actor state.
**Trade-off.** Correctness (exactly-once results) requires application-level idempotency.

### Ray Data / Ray Train / Ray Serve
**What.** Streaming batch pipelines; training orchestration over torch DDP/FSDP; replica management, routing and autoscaling above an engine.
**Use when.** Multi-stage inference pipelines; multi-node training launch; multi-replica serving.
**Pros.** Uniform API, autoscaling, stage statistics.
**Cons.** Another layer with its own overhead; Serve routing is not engine batching.
**Trade-off.** Operational uniformity vs the overhead and the risk of double-batching layers.

---

## Applications (L09 · `applications/`)

### RAG: dense retrieval + reranking
**What.** Embed passages, retrieve top-k by vector similarity, optionally rerank with a cross-encoder, generate with the passages in context.
**Use when.** Answers must be grounded in a corpus; the model is small or the knowledge changes.
**Pros.** Grounding, freshness, citations; reranking raises precision at k.
**Cons.** Retrieval latency and index cost; reranking adds a model call; context length grows.
**Trade-off.** Answer quality vs latency and token cost; prefix ordering decides cache hits.

### Agent tool loops and claim verification
**What.** Model decides tool calls; the loop executes and feeds results back; a verification step checks claims against evidence before answering.
**Use when.** Tasks need external data or computation; errors are costly.
**Pros.** Higher task success and fewer hallucinated claims.
**Cons.** More calls, more latency, tool wait time; verification can remove correct claims.
**Trade-off.** Success rate vs calls, tokens and latency (measured, never assumed).

### Tracing (spans) and the link to serving metrics
**What.** Per-stage spans (retrieval, rerank, model, tool, verify) with one run id, exportable to Perfetto/OpenTelemetry.
**Use when.** Any multi-stage system.
**Pros.** Separates tool wait from model compute; connects application quality to engine metrics.
**Cons.** Instrumentation effort.
**Trade-off.** None worth mentioning; the lab shows how little code it takes.
