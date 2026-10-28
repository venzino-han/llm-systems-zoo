# L02 Report — KV Cache Lab

Status: simulator **Measured (CPU, deterministic)**; engine **pilot, shared GPU**. Raw results in `results/`.

## Question
When context length and concurrency grow, what limits the KV cache, and under which workloads do paged allocation and prefix caching pay off — and by how much?

## Environment
* Simulator: CPU only (`inference/kv_cache/simulate_workloads.py`, numpy seed 0), ~2 min for the four experiments.
* Engine: vLLM 0.23.0 (V1 engine), Qwen/Qwen2.5-0.5B-Instruct bf16, GPU 0 = RTX A6000 48 GB shared with resident vLLM servers, `gpu_memory_utilization=0.10` → 252 896 KV tokens (block 16), `max_model_len=2048`, `enforce_eager=True`, greedy. torch 2.11.0+cu130, driver 580.95.05, CUDA 13.0. Manifest: `results/prefix_caching_pilot_manifest.json`; run dir `runs/L02_kv_cache/20261029_021124_prefix_caching_584984e4d606`.
* Data: system prompts and user prompts cut from `HuggingFaceH4/ultrachat_200k` (`test_sft`).

## Hypothesis
1. Contiguous (max-length) reservation wastes most of the cache under realistic length spreads; paged allocation recovers it and raises concurrency 2–5× for the same memory.
2. Prefix caching's benefit is proportional to the shared-prefix fraction of prompt tokens and vanishes when the set of distinct prefixes exceeds the cache; it reduces prefill time, not decode throughput.

## Baseline
Simulator: `ContiguousAllocator` reserving `max_seq_len` per sequence. Engine: vLLM with `enable_prefix_caching=False`.

## Change
Simulator: `PagedAllocator` (block tables, refcounts, chained-hash prefix cache, LRU eviction, COW) across block sizes, sharing ratios, memory budgets and prefix-variant counts. Engine: `enable_prefix_caching=True`, same prompts.

## Method
* Simulator: 300 requests, lognormal prompt lengths (mean 512), exponential generation lengths (mean 128), 50 % share a 256-token system prompt, FCFS admission with 5 % block reserve, one decode token per active sequence per step; metrics: internal fragmentation, mean concurrency, makespan (steps to drain), hit-token fraction, evictions.
* Engine: batches of 32 prompts with a shared P-token prefix (P ∈ {128, 512, 1024}); phases cold → warm-same → warm-new (prefill only, `max_tokens=1`) → generation (64 tokens); `num_cached_tokens` per request and `LLM.get_metrics()` prefix-cache counters; an unrelated warm-up batch precedes each condition.

## Results
Full tables in `inference/kv_cache/README.md` §6.
* Waste: 71.7 % (contiguous) vs 1.0 % (paged, block 16); makespan 1668 → 689 steps; waste grows to 14.6 % at block 256.
* Concurrency for the same memory: paged 5.2× (8k tokens) … 2.0× (128k tokens) the contiguous allocator.
* Hit fraction follows `share × prefix/(prefix+body)`: up to 0.65 for a 1024-token prefix; collapses from 0.65 to 0.02 as distinct prefixes go from 4 to 32 under a 16k-token budget (LRU thrash); resident prefixes also cost capacity (+29 % makespan from 1 to 4 prompts with hits unchanged).
* Engine: warm prefill of a batch with new suffixes 2.0–2.5× faster with caching on (0.163→0.080 s, 0.279→0.112 s, 0.357→0.141 s); cached tokens per request = exactly the prefix; engine hit rate 85–95 %; generation throughput unchanged (1.8–2.1k tok/s both ways).

## Bottleneck analysis
* Memory: with GQA the 0.5B model needs 12 KiB/token, so even 10 % of an A6000 holds 250k tokens — capacity was never the limit in the engine pilot; it is for 7B+/MHA models (`kv_capacity.py`: 392 KiB/token for an MHA 7B).
* Compute: prefix caching converts prefill from O(prefix + suffix) to O(suffix + 1) per request; decode is unchanged, so the end-to-end gain depends on the prefill share of the workload (large for short answers to long prompts, small for long generations).
* Policy: without an admission reserve the simulator preempted 296 times at the tightest budget and recomputed nearly as many tokens as it produced; the 5 % reserve removed all preemptions at a negligible concurrency cost.

## Limits
* Simulator: no kernel time, no chunked prefill, no swap preemption, FCFS only.
* Engine: shared GPU, eager mode, 10 % memory; single batch per condition, so no spread/percentiles; absolute latencies are not representative of an exclusive GPU with CUDA graphs.
* Only one model size; fp8 KV, offload and multi-GPU TP effects on KV not measured.

## Reproduction commands
See `inference/kv_cache/README.md` §5. CPU tests: `.venv/bin/python -m pytest tests/test_l02_kv_cache.py -q` (15 passed).

## Next experiments
1. Open-loop load (L03 generator) against a vLLM server with/without prefix caching on multi-turn traces: TTFT p50/p99 under arrival rates.
2. Engine `block_size` sweep and `--no-enforce-eager` on an exclusive GPU; repeat each condition ≥ 5× for spread.
3. fp8 KV cache and capacity validation against `kv_capacity.py` on the 7B model (Blackwell GPU).
4. Eviction under pressure in the engine: many distinct prefixes vs `gpu_memory_utilization`.
