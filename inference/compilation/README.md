# L08d. Compilation and CUDA graphs

**Question.** How much of a decode step is launch overhead rather than GPU work, and what do
`torch.compile` and CUDA graphs recover, at what one-off cost?
**Status.** Implemented, Measured (pilot, shared GPU).
**Hardware needed.** 1 CUDA GPU, ~2 GB for the 0.5B model.

## 1. What the technique is

A decode step of a small model launches hundreds of tiny kernels (per layer: norms, projections,
rotary, attention, MLP, residual adds). Each launch costs ~5-10 µs of CPU and driver time, and
in eager PyTorch the Python interpreter sits between every pair. When the GPU work per kernel is
also a few microseconds, the step is **launch-bound**: the GPU idles waiting for the CPU.

Two remedies, which compose:

* **`torch.compile`** (TorchDynamo + Inductor): traces the Python forward into a graph, fuses
  element-wise ops into generated Triton kernels (fewer launches, fewer memory passes) and, in
  `mode="reduce-overhead"`, replays the whole step as a **CUDA graph** (one launch for the entire
  step). `mode="max-autotune"` additionally benchmarks GEMM/conv kernel variants at compile time.
  The catch for decoding: shapes change every step (the KV cache grows), and Dynamo
  **recompiles** on a new static shape. transformers' `cache_implementation="static"`
  pre-allocates the cache at `max_length` so the decode step has fixed shapes; the prefill and
  the first decode step are still distinct graphs.
* **CUDA graphs in the engine** (vLLM, default on): the engine captures the decode step for a
  set of batch sizes at start-up and replays the captured graph, with no Python in the loop at
  all. `enforce_eager=True` disables this and is the comparison point.

### Cost model

```
step_time ~= max(launch_overhead, gpu_time)         (eager, if launches overlap GPU work)
          ~= launch_overhead + gpu_time             (eager, when the CPU is the bottleneck)
compiled: gpu_time (fused)  +  ~one launch per graph
```

Gains are largest when `gpu_time` is small: small models, batch 1, short contexts. At large
batch or for a 70B model the GPU time dominates and compilation buys a few percent.
One-off costs: compile time (tens of seconds to minutes per distinct graph), graph-capture
memory, and *recompilations* whenever a guard fails (new shape, new dtype, a Python value that
changed).

## 2. When to use it

* Small-to-medium models at low batch where profiler traces show gaps between kernels.
* Serving engines: keep CUDA graphs on unless memory for the captures is the constraint or the
  batch sizes vary beyond the captured set.
* Avoid for one-off runs (compile cost never amortises), for highly dynamic control flow, and
  when debugging numerics (compiled kernels fuse and reorder reductions).

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| Removes launch overhead: 1.5-3x per-token speed-up at batch 1 for small models | Compile / capture time of seconds to minutes before the first token; cold-start latency |
| Fused kernels also cut memory traffic for element-wise ops | Recompiles on shape changes; static shapes require pre-allocated (over-sized) caches |
| Engine CUDA graphs need no model code changes | Captured graphs hold memory per batch size; batch sizes outside the set run eager |
| `max-autotune` can pick better GEMM kernels per shape | Longer compile, more disk cache; gains are shape-specific |

What you pay: start-up time, memory for static caches / captures, debugging opacity.
What you gain: the launch-bound fraction of the step, which can be most of it for small models.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `bench_torch_compile.py` | Part A: transformers decode eager vs `torch.compile` modes with a static cache, first-call / second-call / steady-state timing, recompile counts and reasons. Part B: vLLM `enforce_eager` vs CUDA graphs at batch 1 and 8 | GPU |
| `../../tests/test_l08_compile.py` | the recompile counter helper on a toy function (`backend="eager"`, CPU) | CPU |

## 5. How to run

```bash
.venv/bin/python inference/compilation/bench_torch_compile.py --smoke               # eager vs default, batch 1, 32 tokens
.venv/bin/python inference/compilation/bench_torch_compile.py --part hf --modes eager default reduce-overhead --batch-sizes 1 8
.venv/bin/python inference/compilation/bench_torch_compile.py --part vllm --batch-sizes 1 8
.venv/bin/python -m pytest tests/test_l08_compile.py
```

## 6. Results

**Environment.** Part A on the RTX PRO 6000 Blackwell (sm_120), Part B on the RTX A6000 (sm_86);
both GPUs shared with other jobs (*pilot, shared GPU*). torch 2.11.0+cu130, transformers 5.7.0,
vllm 0.23.0. Model Qwen2.5-0.5B-Instruct bf16, 64-token prompts, 128 greedy tokens, steady state
= min of 3 generate() calls after two warm-up calls.

### 6.1 transformers decode: eager vs `torch.compile` (static KV cache)

| mode | batch | first call (s) | second call (s) | steady ms/step | steady tok/s | graphs after 1st call | recompiles on 2nd call | peak mem (GiB) |
|---|---|---|---|---|---|---|---|---|
| eager | 1 | 1.4 | 1.1 | 8.15 | 123 | 0 | 0 | 0.94 |
| eager | 8 | 1.3 | 1.3 | 10.65 | 751 | 0 | 0 | 0.98 |
| default | 1 | 22.2 | 16.2 | **3.78** | 264 | 2 | 1 | 1.07 |
| default | 8 | 41.0 | 25.0 | **4.07** | 1965 | 5 | 1 | 1.10 |
| reduce-overhead | 1 | **segfault (exit 139)** during the first generate, after Inductor logged `skipping cudagraphs due to mutated inputs (72 instances)` from `StaticCache.update -> cumulative_length.add_` | | | | | | |

Run `20261029_022747_torch_compile_edb4674c3b8c` (rows in `metrics.jsonl` / `partial_rows.json`;
the process was killed by the reduce-overhead crash before writing `summary.json`).
A second, clean run minutes later (`20261029_023604_torch_compile_8855b15801f2`) reproduced the
steady state (eager 8.49 / 9.82 ms, default 3.86 / 4.05 ms at batch 1 / 8) but with a **warm
Inductor cache** the first call fell from 22.2 s to 5.1 s (batch 1) and from 41.0 s to 8.9 s
(batch 8), and the second call from 16.2 / 25.0 s to 3.7 / 5.6 s. The compile cost is paid in
full only once per machine and code version; the on-disk cache (`~/.cache` / `TORCHINDUCTOR_CACHE_DIR`)
is what makes compiled serving restarts tolerable.

Reading:

* At batch 1 the eager step costs 8.2 ms for a model whose GPU work is ~1 ms: it is launch-bound.
  Compilation fuses the element-wise ops and removes most of the Python overhead, giving
  **2.2x** per token; at batch 8 the eager step grows only 30% (the GPU work is still hidden
  behind launch overhead) and compilation gives 2.6x.
* The one-off price: 22 s (batch 1) to 41 s (batch 8) for the first call, and a second
  expensive call (16-25 s). The captured recompile reasons explain both: the prefill graph
  (`input_ids` width 64) and the decode graph (width 1) are two different graphs, and on the
  second `generate()` the cache's `is_initialized` guard flips, which triggers one more
  recompile. Three to five graphs for one model; a serving loop with variable batch sizes would
  add one graph per batch size.
* `reduce-overhead` (CUDA graphs through Inductor) does not work with this transformers / torch
  pair: the static cache mutates its inputs in place, Inductor refuses to graph it, and the
  process then segfaults. This is the kind of version-coupling that makes engines capture their
  own CUDA graphs (Part B) instead of relying on `torch.compile` for it.

### 6.2 vLLM: `enforce_eager` vs CUDA graphs

Qwen2.5-0.5B-Instruct on the RTX A6000 (shared), 2 GiB KV budget, 128 greedy tokens, min of 3
(run `20261029_024314_torch_compile_68f51ef15115`, one engine per subprocess).

| engine | batch | steady ms/step | tok/s | engine init (s) | graph capture |
|---|---|---|---|---|---|
| `enforce_eager=True` | 1 | 5.35 | 187 | 8.8 | - |
| `enforce_eager=True` | 8 | 6.98 | 1147 | 8.8 | - |
| CUDA graphs (default) | 1 | **4.48** | 223 | 23.1 | 2.0 s, 0.16 GiB |
| CUDA graphs (default) | 8 | 6.59 | 1213 | 23.1 | 2.0 s, 0.16 GiB |

Reading: on vLLM the step is already fused and scheduled outside Python, so CUDA graphs buy a
further 1.19x at batch 1 and 1.06x at batch 8 (launch overhead is a smaller share than in eager
transformers, and shrinks with batch). The price is 14 s of extra start-up (compile + capture for
every batch size in the capture list) and 0.16 GiB of captured graphs. The same run measured
minutes earlier in the back-to-back quantisation check gave 6.3 ms/step for the same engine, so
the absolute gap here is within co-tenant noise; the direction and the start-up cost are the
reliable parts.


## 7. Limitations and next experiments

* `max-autotune` was skipped in the pilot (minutes of compile on a shared GPU); it belongs in
  the exclusive-GPU run.
* Steady-state ms/step divides a 128-token generate by 128, so it includes one prefill of a
  64-token prompt (<1% of the time).
* Next: a profiler trace (`torch.profiler`) of one eager vs one compiled step to show the gaps
  directly, and the same comparison on a 7B model where launch overhead is a smaller fraction.

## 8. References

* PyTorch docs: `torch.compile`, TorchDynamo guards and recompilation, CUDA graphs.
* transformers docs: *Optimizing inference* / static KV cache + `torch.compile`.
* vLLM docs: CUDA graph capture, `enforce_eager`.
