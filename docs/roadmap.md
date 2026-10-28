# Roadmap and release criteria

Durations are work estimates for focused weeks, not a committed schedule. Public releases accumulate;
the repository never waits for the whole scope.

| Phase | Effort | Core work | Done when |
|---|---|---|---|
| P0 Design & base | 0.5–1 wk | hardware inventory, environment pinning, workload/metric protocol | CPU smoke + run manifests exist |
| P1 Memory & KV | 1–2 wk | L01/L02 simulators, engine baseline | first reproducible report |
| P2 Serving | 1–2 wk | L03 load generator, SLO analysis | saturation curve, raw results, repro commands |
| P3 Multi-GPU & Ray | 2–3 wk | L04/L07 placement, communication, recovery | scaling report and failure tests |
| P4 Training & post-training | 2–4 wk | L05/L06 checkpoints, rollout analysis | performance comparison including quality |
| P5 Advanced optimisation & apps | 2–4 wk | L08/L09 demo, documentation integration | cross-layer trace and hiring navigation paths |

## Releases

* **v0.1** memory, KV and serving flagship experiments; shared harness; reproducible reports.
* **v0.2** multi-GPU and Ray experiments; failure-recovery evidence.
* **v0.3** training/post-training connected to the application demo.
* Later: advanced kernels, disaggregated prefill/decode, context parallelism, MoE/expert parallelism,
  chosen by available hardware and the target roles.

## Pending exclusive-GPU measurements

Pilots in `reports/` ran on shared GPUs. The following re-runs are queued for exclusive hardware:

- [ ] L02 prefix-caching sweep (cold/warm, prefix length × concurrency), 3 repeats
- [ ] L03 open-loop saturation curve, ≥ 200 requests per rate, 3 repeats, plus overload point
- [ ] L04 TP=2 vs 2 replicas vs 1 GPU on the heterogeneous pair (PCIe-bound), then on a homogeneous node
- [ ] L05 strong/weak scaling DDP vs FSDP on 2 GPUs
- [ ] L06 GRPO rollout/learner split with vLLM colocated rollouts vs HF generate
- [ ] L06 DPO inline-reference vs `precompute_ref_log_probs` A/B (OOM on the shared GPU)
- [ ] L07 Ray Serve replicas in front of a live vLLM engine (`--proxy` mode)
- [ ] L08 7B bf16 vs AWQ vs GPTQ quality/throughput table; `torch.compile(mode="max-autotune")`
- [ ] L09 prefix-cache hit rate of the RAG system prompt measured against the L03 server

## Open decisions

* Multi-node access and total GPU budget.
* Priority tracks per target job posting.
* Whether to add SGLang / TensorRT-LLM adapters after the vLLM baselines are reproduced.
* Korean companion commentary alongside the English reports.
