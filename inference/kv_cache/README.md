# L02. KV Cache Lab

**Question.** As context length and concurrency grow, where does the KV cache hit its limits, and under which workloads do paged allocation and prefix caching actually pay off?
**Status.** Explain ✔ · Implement ✔ (simulator, capacity model) · Benchmark: simulator *Measured (CPU)*, vLLM *Measured (pilot, shared GPU)* · Operate: n/a
**Hardware needed.** Simulator and capacity model: CPU only. Engine benchmark: 1 GPU with ≥ 5 GB free (pilot: RTX A6000 at 10 % memory utilisation, shared).

## 1. What the technique is

### The cost

Every decoded token must attend to the keys and values of every previous token in every layer, so a serving engine stores them:

```
kv_bytes_per_token = 2 (K,V) × L (layers) × a_kv (KV heads) × d (head dim) × p (bytes)
```

Qwen2.5-0.5B (24 layers, 2 KV heads of 64, bf16): **12 KiB/token**. Qwen2.5-7B: 56 KiB/token. Llama-3.1-8B: 128 KiB/token. Grouped-query attention (GQA) is what makes these numbers small: with one KV head per query head (MHA) the 7B model would need 392 KiB/token (`kv_capacity.py`).

Capacity follows directly: `max_tokens = (gpu_memory × utilisation − weights − overhead) / kv_bytes_per_token`, and concurrency is that divided by the tokens per request. On a 48 GB GPU at 90 % utilisation, Qwen2.5-7B can hold ~525k KV tokens (≈ 410 concurrent 1024+256-token requests); its MHA twin would hold 72k (56 requests).

### Contiguous vs paged allocation

The naive design (HF `generate`, early FasterTransformer) reserves `max_seq_len` slots per sequence up front because the cache must be contiguous. Nobody uses the full length, so the reserved-but-unused tail is **internal fragmentation**, and freeing sequences of different lengths leaves holes (**external fragmentation**).

PagedAttention (vLLM) splits the cache into fixed-size *blocks* and gives each sequence a *block table* (logical block index → physical block id), exactly like virtual memory pages. Waste shrinks to the unfilled tail of the last block (< `block_size` tokens), any free block can serve any sequence, and because the mapping is indirect, blocks can be **shared** between sequences.

### Prefix caching

Full blocks are immutable, so they can be identified by a hash chained from the previous block's hash and the block's tokens. A new request whose first blocks hash to blocks already resident reuses them (reference count +1) and skips their prefill compute. Blocks whose reference count drops to zero are kept in an LRU list and only evicted when the free list is empty; a cache hit on an evictable block revives it. Parallel sampling / beam search shares a partial block between parent and child, and the first append triggers **copy-on-write**.

### Minimal implementation vs production engine

`paged_allocator.py` implements the bookkeeping (block tables, refcounts, chained hashes, LRU eviction, COW, rollback on OOM) in ~300 lines of Python and never touches a tensor. vLLM additionally has to: run the PagedAttention kernels that gather scattered blocks, schedule chunked prefill and preemption, account for multi-GPU (TP splits KV heads across devices), keep the hash → block map consistent under concurrency, and handle sliding-window / MLA layouts. The simulator answers *allocation-policy* questions; `bench_prefix_caching.py` answers *latency* questions on the real engine.

## 2. When to use it

* Paged allocation: always, once more than a handful of sequences are served concurrently with variable lengths. The only case it loses is a single fixed-length sequence, where a contiguous buffer is simpler and has no indirection.
* Prefix caching helps when requests share long prefixes: system prompts, few-shot templates, multi-turn chat (the previous turns are the prefix), RAG with repeated documents, agent loops that re-send the trajectory. It helps less when prefixes are shorter than a block, when the working set of distinct prefixes exceeds the cache, or when the workload is decode-dominated (the saving is prefill compute, not decode time).
* Larger block sizes reduce metadata and improve kernel efficiency but increase tail waste and reduce sharing granularity (a prefix only hits at full-block boundaries).

## 3. Trade-offs

| Technique | Advantages | Disadvantages |
|---|---|---|
| Paged (block) allocation | near-zero internal fragmentation (≤ 1 % at block 16 in the simulation vs 72 % contiguous); no external fragmentation; 2–5× more concurrent sequences for the same memory; enables sharing and preemption | indirection in the attention kernel (gather by block table); block-table metadata and bookkeeping; tail waste grows with block size |
| Prefix caching | skips prefill for shared prefixes (TTFT and prefill throughput); zero extra memory when the prefix is already resident | hashing cost per block; only full blocks hit; LRU state and cache pressure; a cold or diverse workload gets no benefit; correctness depends on hashing exactly what affects the KV (tokens, LoRA id, multimodal inputs) |
| Larger block size | fewer blocks to track; better kernel locality | more tail waste (14.6 % at block 256 in the simulation); sharing only at coarser boundaries |
| Admission reserve (watermark) | avoids preemption thrash when the cache is nearly full | slightly lower peak concurrency |
| KV dtype (fp8) / eviction / offload | 2× tokens per GB; longer effective context | quality and kernel-support constraints; CPU offload adds PCIe latency (not measured here) |

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `paged_allocator.py` | `ContiguousAllocator` and `PagedAllocator` simulators (block tables, refcounts, chained-hash prefix cache, LRU eviction, COW, rollback) | CPU |
| `simulate_workloads.py` | discrete-time admission/decode simulation; four experiments with tables and figures | CPU |
| `kv_capacity.py` | capacity model (`kv_bytes_per_token`, `max_kv_tokens`, `max_concurrency`, GQA vs MHA) | CPU |
| `bench_prefix_caching.py` | vLLM offline benchmark: prefix caching off/on, cold / warm-same / warm-new prefill, generation throughput, engine hit counters | GPU |
| `../../tests/test_l02_kv_cache.py` | 15 tests: allocation/free/refcount, prefix hits, chained hashes, LRU eviction, revival, COW, fragmentation, capacity, simulator | CPU |
| `../../reports/L02_kv_cache/` | `REPORT.md`, `results/sim/` (JSON + 6 figures), `results/prefix_caching_*` (engine runs) | — |

## 5. How to run

```bash
# CPU
.venv/bin/python inference/kv_cache/kv_capacity.py --gpu-gib 48 --prompt-len 1024 --gen-len 256
.venv/bin/python inference/kv_cache/simulate_workloads.py --smoke                       # ~2 s
.venv/bin/python inference/kv_cache/simulate_workloads.py --out reports/L02_kv_cache/results/sim   # ~2 min, the tables below
.venv/bin/python -m pytest tests/test_l02_kv_cache.py -q

# GPU (vLLM 0.23; shared GPU 0 → 10 % memory utilisation, eager mode)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
.venv/bin/python inference/kv_cache/bench_prefix_caching.py --smoke          # P=512, 8 prompts, ~3 min incl. 2 engine start-ups
.venv/bin/python inference/kv_cache/bench_prefix_caching.py                  # P ∈ {128, 512, 1024}, 32 prompts, 64 output tokens
```

## 6. Results

### 6.1 Simulator (CPU, deterministic; `results/sim/simulation_results.json`)

Workload: 300 requests, prompt length lognormal (mean 512, σ 0.5, cap 4096), generation length exponential (mean 128, cap 512), 50 % of requests begin with a 256-token system prompt, FCFS admission with a 5 % block reserve for decode growth, one token per sequence per step. "Makespan" is the number of decode steps to drain the queue at a fixed memory budget; it is a throughput proxy, not a latency.

**Waste vs block size** (capacity 65 536 tokens) — `waste_vs_block_size.png`, `makespan_vs_block_size.png`

| allocator | block size | internal fragmentation | mean concurrent seqs | makespan (steps) |
|---|---|---|---|---|
| contiguous (reserve `max_seq_len`) | — | **71.7 %** | 20.6 | 1668 |
| paged | 1 | 0.0 % | 50.4 | 683 |
| paged | 8 | 0.5 % | 50.0 | 688 |
| paged | **16** | **1.0 %** | 50.0 | 689 |
| paged | 64 | 4.1 % | 48.8 | 705 |
| paged | 128 | 7.9 % | 47.7 | 722 |
| paged | 256 | 14.6 % | 45.2 | 762 |

Reserving the maximum length wastes 72 % of the cache for this length distribution; block 16 (vLLM's default) wastes 1 % and drains the queue 2.4× faster with 2.4× the concurrency. Waste grows linearly with block size (≈ `block_size / 2` tokens per sequence), and the makespan penalty follows.

**Prefix-cache hit fraction vs sharing** (block 16, capacity 65 536) — `prefix_hit_vs_shared_ratio.png`

| system prompt length | 0 % share | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|---|
| 128 tokens | 0.000 | 0.059 | 0.112 | 0.155 | 0.190 |
| 512 tokens | 0.000 | 0.199 | 0.336 | 0.424 | 0.484 |
| 1024 tokens | 0.000 | 0.331 | 0.502 | 0.594 | 0.652 |

The fraction of prompt tokens served from cache is bounded by `prefix / (prefix + body)` (65 % for a 1024-token prefix on ~570-token bodies) and reaches that bound minus block-granularity loss. In the simulator a hit saves memory but not steps (one step per decode token regardless), so the makespan hardly moves (688 → 697); the compute saving only appears in the engine measurement below.

**Concurrency vs memory budget** (block 16) — `concurrency_vs_capacity.png`, `throughput_vs_capacity.png`

| capacity (tokens) | contiguous: mean active / makespan | paged: mean active / makespan | paged advantage |
|---|---|---|---|
| 8 192 | 2.0 / 17 379 | 10.2 / 3 364 | 5.2× |
| 16 384 | 4.9 / 7 043 | 20.6 / 1 671 | 4.2× |
| 32 768 | 10.4 / 3 306 | 33.3 / 1 035 | 3.2× |
| 65 536 | 20.6 / 1 668 | 50.0 / 689 | 2.4× |
| 131 072 | 33.0 / 1 043 | 65.2 / 528 | 2.0× |

The advantage is largest when memory is tight; once the budget exceeds the whole queue's footprint both converge to the workload's natural parallelism. No preemptions occurred with the 5 % reserve (without it the paged allocator thrashed: 296 preemptions at 8 192 tokens in an earlier run).

**Cache pressure: distinct system prompts** (capacity 16 384 tokens, 1024-token prompts, every request uses one of N prompts round-robin) — `hit_vs_prefix_variants.png`

| distinct prompts | hit fraction | evictions | makespan |
|---|---|---|---|
| 1 | 0.652 | 11 236 | 1 833 |
| 4 | 0.645 | 11 427 | 2 360 |
| 8 | 0.504 | 15 588 | 3 218 |
| 16 | 0.155 | 25 828 | 3 839 |
| 32 | 0.020 | 29 795 | 4 051 |

Hit rate collapses once the prefix working set (N × 1024 tokens) approaches the cache (16 384 tokens): LRU evicts prefixes before they are reused. Even while hits stay high (1 → 4 prompts) the makespan rises 29 %, because resident shared prefixes consume capacity that would otherwise admit more sequences — sharing is not free memory.

### 6.2 Engine: vLLM prefix caching off vs on (pilot, shared GPU)

Environment: vLLM 0.23.0, Qwen/Qwen2.5-0.5B-Instruct bf16, RTX A6000 shared with other jobs, `gpu_memory_utilization=0.10` (engine reported 3.16 GiB of KV cache = 275 696 tokens, block 16), `max_model_len=2048`, `enforce_eager=True`, greedy sampling. Prompts: a system prompt of P tokens built from UltraChat text followed by `\n\nUser: ` and a distinct UltraChat prompt truncated to 48 tokens; two disjoint sets of 32 user prompts (A and B). Prefill-only latencies use `max_tokens=1` for the whole batch (a TTFT proxy without network or queueing). The engine process is warmed up with an unrelated prompt before each condition.

Run `runs/L02_kv_cache/20261029_021124_prefix_caching_584984e4d606` (summary, manifest and figures copied to `results/prefix_caching_pilot_*`). 32 prompts per batch, 64 output tokens for the generation phase.

| prefix caching | P (tokens) | prompt tokens / batch | cold (s) | warm, same prompts (s) | warm, new suffixes (s) | tokens served from cache (warm-new) | engine hit rate (cumulative) | generation, cache warm (output tok/s) |
|---|---|---|---|---|---|---|---|---|
| off | 128 | 5 499 | 0.135 | 0.120 | 0.163 | 0 / 5 499 | — | 1 891 |
| off | 512 | 17 787 | 0.282 | 0.278 | 0.279 | 0 / 17 787 | — | 1 815 |
| off | 1024 | 34 171 | 0.351 | 0.353 | 0.357 | 0 / 34 171 | — | 2 141 |
| **on** | 128 | 5 499 | 0.075 | 0.074 | **0.080** | 4 112 / 5 499 (75 %) | 18 880 / 22 174 = 85 % | 2 012 |
| **on** | 512 | 17 787 | 0.142 | 0.118 | **0.112** | 16 400 / 17 787 (92 %) | 82 432 / 88 996 = 93 % | 2 095 |
| **on** | 1024 | 34 171 | 0.164 | 0.187 | **0.141** | 32 784 / 34 171 (96 %) | 214 032 / 224 747 = 95 % | 2 053 |

Interpretation:

* **Prefill of a batch that shares a system prompt is 2.0–2.5× faster with prefix caching** (warm-new: 0.163 → 0.080 s at P = 128, 0.279 → 0.112 s at P = 512, 0.357 → 0.141 s at P = 1024). The saving tracks the cached fraction: per request, exactly the prefix (128.5 / 512.5 / 1024.5 tokens, i.e. all full 16-token blocks) is reported as `num_cached_tokens`; the unique suffix and the final token are always recomputed.
* **"Cold" is not cold inside a batch.** With caching on, the first batch is already fast (0.164 vs 0.351 s at P = 1024) because vLLM caches blocks as the first request is prefilled and the other 31 requests in the same scheduling step hit them. The cold → warm gain *within* the caching-on rows is therefore only 1.16–1.26×; the off → on comparison is the honest measure of the mechanism.
* **Decode is untouched.** Generation throughput (1.8–2.1k output tok/s, decode-dominated) is the same within the noise of a shared GPU in both configurations: prefix caching removes prefill compute, not per-token decode cost. Workloads that are decode-bound gain little.
* **The engine's own counters agree with the simulator's model:** hit rate = full prefix blocks / all blocks queried, rising with P (85 → 95 %); the simulator predicted the same shape (§6.1, prefix experiment).
* Identical prompts ("warm, same") are not faster than new suffixes: all but the last block are cached in both cases. The P = 1024 "same" value (0.187 s) is above "new" (0.141 s), which is timing noise on the shared GPU — single batches of 32 requests, not repeated, so no spread is reported.
* Caching off vs on costs no measurable memory here: both engines reported 252 896 KV tokens available at start-up; the hash table lives in host memory.

Smoke run (`results/prefix_caching_smoke_*`, P = 512, 8 prompts, no engine warm-up) showed the same 2.5× warm-new speed-up (0.052 → 0.021 s) and is kept only as a reproducibility check.


## 7. Limitations and next experiments

* The simulator charges one step per decode token for every active sequence; it models memory, admission and sharing, not kernel time. It has FCFS admission and recompute preemption only (no swap, no chunked prefill, no priority).
* Engine numbers are from a shared GPU at 10 % memory utilisation in eager mode; absolute times will differ on an exclusive GPU with CUDA graphs (`--no-enforce-eager`). Hit rates and relative speed-ups are the quantities to read. p99 is not reported: 32 requests per condition is too few.
* Not measured yet: KV dtype fp8 (`--kv-cache-dtype fp8`), CPU offload / eviction policies under memory pressure, multi-turn chat traces, prefix caching under concurrent arrivals (needs the L03 load generator against a server).
* Next: run the L03 open-loop load generator with and without prefix caching on a long shared prefix; sweep `block_size` in the engine; measure fp8 KV on the Blackwell GPU.

## 8. References

* Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
* vLLM docs: *Automatic Prefix Caching* (design: chained block hashes, LRU eviction) and V1 KV-cache manager.
* Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
* Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, 2023 (RadixAttention: prefix sharing as a radix tree).
* Pope et al., *Efficiently Scaling Transformer Inference*, 2022.
