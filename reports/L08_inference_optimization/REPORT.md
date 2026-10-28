# L08 Inference Optimization: report

## Question

Quantisation, attention / execution optimisation (kernels, compilation, CUDA graphs) and
speculative decoding each promise lower latency or memory. On which conditions (model size,
batch size, sequence length, GPU generation) does each one actually deliver, what does it cost
in quality or start-up time, and which of them even have working kernels on the hardware at
hand?

## Environment

| | |
|---|---|
| GPUs | GPU 0: NVIDIA RTX A6000 (sm_86, 48 GB, Ampere). GPU 1: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120, 96 GB). PCIe (PHB), no NVLink. |
| Sharing | **Both GPUs were shared** during every run: a 90 GB vLLM server (Gemma) on GPU 1, a 27 GB one on GPU 0, plus the other labs' jobs. Free memory fluctuated between ~0 and ~16 GB. All numbers are labelled *pilot, shared GPU*; orderings were stable, absolute values are noisy. |
| Software | Python 3.10.12, torch 2.11.0+cu130, transformers 5.7.0, vllm 0.23.0, triton 3.6.0, bitsandbytes 0.48.2, flashinfer 0.6.12, flash-attn-3 3.0.0b1 (Hopper build), CUDA driver 13.0 / 580.95.05 |
| Models | Qwen/Qwen2.5-0.5B-Instruct (bf16, `-AWQ`, `-GPTQ-Int4`, `-GPTQ-Int8`), Qwen/Qwen2.5-1.5B-Instruct (speculative target). 7B models were **not** run (memory budget on the shared GPUs). |
| Data | wikitext-2-raw-v1 test (perplexity), HuggingFaceH4/ultrachat_200k test_sft first user turns (decode prompts), 6 fixed prompts (minimal speculative decoder) |
| Provenance | every run directory under `runs/L08_inference_optimization/` carries `manifest.json` with hardware, software, config hash and git state; curated copies in `reports/L08_inference_optimization/results/` |

## Hypotheses

1. On a 0.5B model the decode step is launch-bound, so weight quantisation will save memory but
   not time, while compilation / CUDA graphs will give the largest latency win.
2. bitsandbytes INT8 (LLM.int8) will be slower than bf16; NF4 smaller but also slower; AWQ/GPTQ
   through vLLM's Marlin kernels will be the only quantised modes at or above bf16 speed.
3. FP8 KV cache will roughly double KV capacity at a small perplexity cost.
4. Speculative decoding with a 0.5B draft and a 1.5B target will show a high acceptance rate but
   little or no wall-clock gain in the eager minimal implementation (draft cost `c` too high);
   n-gram speculation on chat prompts will have a near-zero acceptance rate.
5. Fused attention backends will be 10-50x faster than materialised attention at L >= 2k and
   use linear instead of quadratic memory; the Hopper-only FA3 wheel will not run on sm_120.

## Baseline

bf16 weights, eager transformers decode (HF path) or vLLM with default CUDA graphs and
`kv_cache_dtype=auto` (vLLM path); no speculation; `sdpa_math` for attention; eager PyTorch for
the element-wise ops.

## Change (one variable at a time)

Loading mode (bf16 / fp16 / int8_bnb / nf4_bnb / awq / gptq / gptq_int8 / fp8 / nf4 in vLLM),
KV dtype (auto / fp8), speculation (none / ngram / draft model, K = 4; K sweep in the minimal
implementation), compilation mode (eager / default / reduce-overhead; vLLM enforce_eager vs CUDA
graphs), attention backend, kernel implementation.

## Method

* Perplexity: token-level on 65,536 wikitext-2 tokens, 1024-token non-overlapping windows
  (identical windows for HF logits and vLLM prompt logprobs). Smoke runs use 8,192 tokens.
* Decode throughput: greedy, `min_tokens = max_tokens = 128`, batch 1 and 8 (same prompts for
  every mode), min of repeats after a warm-up call; HF timings are `generate()` wall-clock.
* vLLM runs use a **fixed 2 GiB KV budget** (`kv_cache_memory_bytes`) so capacity comparisons
  are not a function of whatever memory happened to be free; `max_model_len = 2048`.
* Kernel timings: `triton.testing.do_bench` (10 warm-up / 50 reps) or CUDA events (5 / 20);
  correctness is checked against an fp32 reference before any timing.
* Each quantisation mode runs in its own subprocess so loader failures are recorded as data.

## Results

All numbers: *pilot, shared GPU*. Full tables and figures are in the four lab READMEs; this
section keeps the headline per experiment. Model Qwen2.5-0.5B-Instruct unless stated.

### R1. Weight quantisation (details: `inference/quantization/README.md` 6.1-6.2)

| scheme (engine) | weights GiB | ppl (65k tok) | decode ms/step, batch 1 | note |
|---|---|---|---|---|
| bf16 (HF eager, Blackwell) | 0.92 | 16.26 | 8.36 | baseline |
| LLM.int8 (HF+bnb) | 0.59 | 16.27 | 43.6 | 5x slower than bf16 |
| NF4 (HF+bnb) | 0.42 | 17.92 | 16.4 | 2x slower than bf16 |
| bf16 (vLLM, A6000) | 0.93 | 16.26 | 4.79* | CUDA graphs on |
| GPTQ-Int8 Marlin (vLLM) | 0.60 | 16.24 | 2.01 | free in ppl |
| GPTQ-Int4 Marlin (vLLM) | 0.43 | 19.05 | 1.77 | +2.8 ppl |
| AWQ Marlin (vLLM) | 0.43 | 19.39 | 1.83 | +3.1 ppl |
| FP8 dynamic (vLLM, A6000 sm_86) | 0.59 | **2.9e6** | 4.34 | broken: Marlin W8A16 fallback produces garbage |
| FP8 dynamic (vLLM, Blackwell sm_120) | 0.59 | 16.45 | n/a (other GPU) | native FP8 path is fine |
| NF4 in-flight (vLLM) | 0.45 | 17.93 | 4.65* | matches bnb ppl |
| AWQ / GPTQ (HF loader) | fail | | | `gptqmodel` / `optimum` not installed |

\* measured in a later window under heavier co-tenant load than the Marlin rows; see the
back-to-back re-check:

| mode (vLLM, A6000, same window) | ms/step b=1 | ms/step b=8 | vs bf16 |
|---|---|---|---|
| bf16 | 6.30 | 6.83 | 1.00x / 1.00x |
| GPTQ-Int8 Marlin | 3.95 | 6.24 | 1.60x / 1.09x |
| GPTQ-Int4 Marlin | 3.82 | 3.98 | 1.65x / 1.71x |
| AWQ Marlin | 3.77 | 4.00 | 1.67x / 1.71x |
| NF4 in-flight | 6.36 | 10.45 | 1.00x / 0.65x |

### R2. KV-cache dtype (fixed 2 GiB budget, batch 64)

FP8 KV: **2.00x** tokens (174,752 -> 349,520), ppl 16.26 -> 16.89, step 5.02 -> 3.44 ms.

### R3. Speculative decoding

* Minimal implementation, 0.5B draft -> 1.5B target, batch 1, eager HF: acceptance 0.73-0.86 per
  position, 3.0 tokens per target forward at K = 4 and 3.7 at K = 8, yet **0.71x / 0.47x** wall-clock
  because both models' steps cost ~10 ms (launch-bound): draft cost `c ~ 1`.
* vLLM (1.5B target, K = 4): draft-model speculation reaches 2.9-3.1 tokens per target step
  (acceptance ~0.5 per position) but wall-clock is **1.00x** at batch 1 and **0.74-0.77x** at
  batch 4-16; n-gram speculation accepts 4% at batch 1 (open-ended chat prompts) and ~30% at
  batch 4-16, for 0.81-0.95x. Greedy outputs identical to plain decode at batch 1, divergent on
  some prompts at larger batch (finite-precision, not a correctness bug).

### R4. Compilation and CUDA graphs

* transformers, static cache: eager 8.2-8.5 ms/step -> `torch.compile` default **3.8-3.9 ms** at
  batch 1 (2.2x) and 10.7 -> 4.1 ms at batch 8 (2.6x). First-call cost 22 s / 41 s with a cold
  Inductor cache, 5 s / 9 s warm; 3-5 graphs (prefill vs decode shapes, cache-init guard).
  `reduce-overhead` **segfaults** (exit 139) with transformers 5.7 / torch 2.11 (static cache
  mutates inputs; Inductor skips CUDA graphs, then crashes).
* vLLM: `enforce_eager` 5.35 / 6.98 ms per step (batch 1 / 8) vs CUDA graphs **4.48 / 6.59 ms**
  (1.19x / 1.06x); CUDA-graph engine init 23.1 s vs 8.8 s, 2.0 s capture, 0.16 GiB of graphs.

### R5. Kernels (Blackwell)

* Fused attention vs materialised scores, 8 heads, d = 64: at L = 2048, 0.034 ms (FA2) /
  0.037 ms (cuDNN) vs 1.68 ms (`sdpa_math`), i.e. 45-50x, with 20 MiB vs 322 MiB workspace; the
  materialised variants OOM at L >= 4096 on the ~2 GB the shared GPU had free. All fused backends
  agree with the fp32 reference to bf16 rounding (max abs err <= 1.6e-2).
* Triton RMSNorm one-pass kernel: 3.4-12x faster than the HF eager composition; `torch.compile`
  matches or beats it at 2-4k hidden (untuned Triton launch config). Fused add+RMSNorm costs ~1.5x
  a plain norm while replacing two ops.
* Triton softmax: beats `torch.softmax` by 1.3-3x from 1k to 8k columns, loses at 128 (launch-bound)
  and at 16k (one block per row is too little parallelism).
* flash-attn-3 (Hopper build) aborts the process on sm_120: `no kernel image is available`.


## Bottleneck analysis

* **Launch overhead dominates the 0.5B model.** Eager HF spends ~8 ms per step for ~1 ms of GPU
  work; vLLM's fused kernels + CUDA graphs get to ~2 ms and `torch.compile` to ~3.9 ms. Anything
  that only reduces bytes moved (INT8 / INT4 weights) cannot show up in latency here, and
  anything that adds work per step (LLM.int8's two-pass matmul, NF4 dequantisation, a draft model
  that is itself launch-bound) shows up as a slowdown. The memory-bound regime that quantisation
  and speculation are designed for starts at models whose weight read exceeds the per-step fixed
  cost, i.e. several GB of weights at batch 1.
* **Quality cost of 4 bits is model-size dependent.** +2.8-3.1 perplexity for a 0.5B model is
  far above the ~0.1-0.3 typically reported for 7B+ models; small models have less redundancy.
  INT8 weights and FP8 activations (on hardware that supports them) are the "free" tier.
* **Kernel availability is a first-class variable.** Three failures were found in one afternoon:
  FP8 is numerically wrong on sm_86 in vLLM 0.23 (works on sm_120), FA3 is Hopper-only, and
  `reduce-overhead` segfaults with the current transformers static cache. Each would have
  silently produced a wrong benchmark or a crashed deployment; the perplexity check caught the
  first, compute-capability gating the second, per-process isolation the third.
* **Shared-GPU measurement artefacts.** vLLM's free-memory profiling asserts when a co-tenant
  moves memory (fixed KV budget avoids it); KV capacity depends on free memory at start-up
  (fixed budget again); in-process engines leak across restarts (one engine per subprocess);
  speed rows measured minutes apart are not comparable (back-to-back re-check). These are the
  harness changes this lab made, and the reason the numbers are labelled pilot.


## Limits

* Shared GPUs: no exclusive measurement exists yet; variance across repeats was not quantified
  beyond "min of 2-3" per point. Nothing here should be quoted as a hardware number.
* Only 0.5B / 1.5B models: the memory-bound regime where INT4 weights and speculation pay off
  (7B+ at batch 1-4) is not covered. The 7B bf16 vs AWQ table is **pending an exclusive GPU**.
* HF-side AWQ / GPTQ loaders need `gptqmodel` / `optimum` (not installed): recorded as failures,
  the vLLM path covers those checkpoints.
* flash-attn-3 is a Hopper-only build and aborts the process on sm_120 (gated by compute
  capability in the benchmark).
* Perplexity is the only quality metric; task accuracy was not measured in this lab.

## Reproduction

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>
.venv/bin/python -m pytest tests/test_l08_*.py                       # CPU tests
.venv/bin/python inference/quantization/bench_quant.py --hf-modes bf16 fp16 int8_bnb nf4_bnb awq gptq --vllm-modes --max-tokens 65536
.venv/bin/python inference/quantization/bench_quant.py --hf-modes --vllm-modes bf16 awq gptq gptq_int8 fp8 nf4_bnb --max-tokens 65536
.venv/bin/python inference/quantization/kv_cache_dtype.py --dtypes auto fp8 --batch 64
.venv/bin/python inference/speculative_decoding/spec_decode_minimal.py --k 1 2 4 8 --max-new-tokens 128
.venv/bin/python inference/speculative_decoding/bench_spec_decode.py --methods none ngram draft --batch-sizes 1 4 16
.venv/bin/python inference/compilation/bench_torch_compile.py --part hf --modes eager default reduce-overhead --batch-sizes 1 8
.venv/bin/python inference/compilation/bench_torch_compile.py --part vllm --batch-sizes 1 8
.venv/bin/python inference/kernels/triton_softmax.py --rows 2048 --cols 128 512 1024 2048 4096 8192 16384
.venv/bin/python inference/kernels/triton_rmsnorm.py --rows 4096
.venv/bin/python inference/kernels/attention_bench.py --heads 8 --seq 256 512 1024 2048 4096 8192 --head-dim 64 128
```

Add `--min-free-gib N` to any script to wait for free memory on a shared GPU.

## Next experiments

1. **Exclusive-GPU measurement run** of everything above with 3 repeats and error bars; that is
   the gate for quoting any number outside this repository.
2. **7B regime**: Qwen2.5-7B-Instruct bf16 vs `-AWQ` vs `-GPTQ-Int4` in vLLM at batch 1 / 4 / 16
   (memory-bound decode, where INT4 weights should give 1.5-2.5x), plus the 0.5B draft -> 7B
   target speculative run (where draft cost `c` is small enough to pay off).
3. **Quality beyond perplexity**: GSM8K accuracy (`post_training/eval_gsm8k.py`) for each
   quantised checkpoint, since 4-bit damage shows up on reasoning first.
4. **FP8 on Blackwell**: W8A8 FP8 with native tensor cores at batch 16-64 (compute-bound
   regime), FP8 KV with calibrated scales vs the unscaled e4m3 default.
5. **Compilation**: `max-autotune`, `reduce-overhead` once transformers' static cache stops
   mutating inputs in place (or with a hand-written static-cache decode loop), and a profiler
   trace that attributes the eager step to launch gaps vs kernels.
6. **Kernels**: `triton.autotune` over block size / warps, multi-block rows for the 16k-column
   case, a fused RMSNorm + FP8-quantise kernel, and paged-attention decode kernels tied to the
   KV-cache lab.
7. **Speculation**: EAGLE drafter in vLLM, acceptance vs temperature for the sampling variant,
   n-gram drafting on the RAG workload of L09 where answers quote the prompt.

