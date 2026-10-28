# L08a. Quantization: weights, activations and the KV cache

**Question.** How much memory and latency does each quantisation scheme save on this model and
GPU, what does it cost in perplexity, and which schemes actually have a working kernel here?
**Status.** Implemented, Measured (pilot on a shared GPU, 0.5B model). 7B bf16 vs AWQ table:
*pending an exclusive GPU*.
**Hardware needed.** 1 CUDA GPU with ~5 GB free for the 0.5B pilots; ~20 GB for the 7B table.

## 1. What the technique is

Quantisation stores (and sometimes computes) tensors in fewer bits than bf16. For LLM
inference it attacks three different memory consumers, and the right scheme depends on which one
is the bottleneck:

| Target | Scheme family | What it saves | What it costs |
|---|---|---|---|
| Weights | weight-only INT8 / INT4 (W8A16, W4A16) | model size, and HBM traffic per decode step (decode is weight-bandwidth-bound at small batch) | dequantise inside the GEMM; accuracy loss grows as bits drop |
| Weights + activations | W8A8 (INT8 / FP8) | uses INT8/FP8 tensor cores: faster *compute-bound* prefill and large-batch decode | activation outliers; needs calibration or per-token scaling; GPU must have the tensor-core path |
| KV cache | FP8 / INT8 KV | halves per-token KV memory: 2x tokens in the same budget, longer contexts, more concurrency | dequantise in the attention kernel; small perplexity change |

Schemes exercised in this lab:

* **bitsandbytes LLM.int8** (`int8_bnb`, Dettmers et al. 2022): INT8 weights with vector-wise
  scaling; activation outlier columns are split out and kept in fp16. Zero-shot (no calibration),
  works on any HF model, but its mixed-precision decomposition is slow: it is a memory-saving
  tool, not a speed-up.
* **bitsandbytes NF4** (`nf4_bnb`, QLoRA): 4-bit "normal float" weight-only with block-wise
  scales and double quantisation. Smallest memory, zero-shot, but every matmul dequantises to
  bf16 first, so decode is slower than bf16 at batch 1 on most GPUs.
* **AWQ** (`awq`, Lin et al. 2023): 4-bit weight-only, post-training. Observes activations on a
  small calibration set and scales the ~1% of weight channels that see large activations so they
  lose less precision. Pre-quantised checkpoints are common on the Hub; vLLM runs them through
  Marlin-style W4A16 kernels, which are *faster* than bf16 at small batch because they move 4x
  fewer bytes per step.
* **GPTQ** (`gptq`, Frantar et al. 2022): 4-bit (or 8-bit) weight-only, post-training; quantises
  layer by layer with second-order (Hessian) error correction. Similar deployment profile to AWQ.
* **FP8** (`fp8`): W8A8 with 8-bit floating point (e4m3 for weights/activations, e5m2 alternative).
  vLLM can quantise a bf16 checkpoint *dynamically* at load time; the fast path needs FP8 tensor
  cores (sm_89+: Ada, Hopper, Blackwell). On older GPUs vLLM falls back to a W8A16 Marlin kernel
  (memory saving only).
* **FP8 KV cache** (`kv_cache_dtype=fp8`): stores K/V in e4m3 (no scales by default, or with
  calibrated scales). Orthogonal to weight quantisation.

### Cost model

Per decode step at batch size `B`, a layer's linear ops move `W_bytes = params x bits/8` once and
compute `2 x params x B` FLOPs. The step is memory-bound while
`B < (peak FLOPs) / (HBM bandwidth) x 8/bits` (hundreds of tokens on modern GPUs), so reducing
`bits` reduces time almost proportionally *at small batch*, and not at all once compute-bound.
This is why weight-only INT4 helps single-stream latency and FP8 W8A8 helps throughput.

KV memory per token is `2 x layers x kv_heads x head_dim x bytes`; for Qwen2.5-0.5B (24 layers,
2 KV heads, d=64) that is 12 KiB/token in bf16 and 6 KiB in FP8.

### Educational vs production

`eval_perplexity.py` uses stock transformers + bitsandbytes for the zero-shot schemes and vLLM
for the kernel-backed ones. Nothing here re-implements a quantiser; the value is the controlled
comparison (same model, same windows, same prompts) and the explicit record of which kernels
exist on which GPU.

## 2. When to use it

* **Memory does not fit** (model or KV): weight-only INT4 (AWQ/GPTQ) for weights, FP8 KV for
  context. NF4 if you need zero-shot and can afford slower decode.
* **Single-stream latency** (batch 1-8, chat): W4A16 kernels (AWQ/GPTQ via Marlin) give the
  largest decode speed-up because the step is weight-bandwidth-bound.
* **Throughput at large batch** (offline batch inference, high-concurrency serving): W8A8 FP8 on
  Ada/Hopper/Blackwell; INT4 weight-only stops helping because the dequantise cost sits inside a
  now compute-bound GEMM.
* **Do not** quantise when the model is small enough that the step is launch-bound (as with the
  0.5B model below): there is nothing to save, and dequantisation adds work.

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| 2-4x smaller weights; 1.3-3x faster decode at small batch with a good W4A16 kernel | Perplexity / task accuracy loss, larger at 4 bits, on small models, and on reasoning / code tasks; must be measured per model |
| FP8 KV doubles context or concurrency for the same memory | Dequantise step in attention; e4m3 without scales clips outliers (visible as a ppl bump below) |
| Pre-quantised checkpoints make deployment a config flag | Kernel availability is per GPU generation (Marlin needs sm_80+, FP8 tensor cores sm_89+, FA3 sm_90 only); bitsandbytes needs its own build per CUDA |
| Post-training (AWQ/GPTQ) needs minutes of calibration, no training | Calibration set bias; AWQ/GPTQ checkpoints are usually only released for the base precision's tokenizer/config, so you depend on a third party's quantisation choices |

What you pay: quality (must be re-evaluated), kernel dependence, extra engineering per scheme.
What you gain: memory and, in the memory-bound regime, latency.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `eval_perplexity.py` | One (engine, mode): wikitext-2 perplexity, weights + peak memory, prefill tok/s, decode tok/s at batch 1 and N | GPU |
| `bench_quant.py` | Runs `eval_perplexity.py` per mode in a subprocess, collates the table, draws quality-vs-memory / quality-vs-speed | GPU |
| `kv_cache_dtype.py` | vLLM `kv_cache_dtype` auto vs fp8 under a fixed KV budget: capacity, concurrency, throughput, perplexity | GPU |
| `common.py` | prompt loading (ultrachat), wikitext tokenisation, vLLM log capture, shared-GPU helpers (`wait_for_free_memory`, `build_llm`) | CPU |
| `../../tests/test_l08_quantization.py` | window bookkeeping, perplexity math on a tiny model, checkpoint resolution, log parsing, table building | CPU |

## 5. How to run

```bash
# smoke (bf16, 8k tokens, under 2 minutes)
.venv/bin/python inference/quantization/eval_perplexity.py --engine hf --load bf16 --smoke
.venv/bin/python inference/quantization/eval_perplexity.py --engine vllm --load awq --smoke
.venv/bin/python inference/quantization/kv_cache_dtype.py --smoke

# the pilots quoted below (HF modes and vLLM modes were run on different GPUs, see 6)
.venv/bin/python inference/quantization/bench_quant.py --hf-modes bf16 fp16 int8_bnb nf4_bnb awq gptq --vllm-modes --max-tokens 65536
.venv/bin/python inference/quantization/bench_quant.py --hf-modes --vllm-modes bf16 awq gptq gptq_int8 fp8 nf4_bnb --max-tokens 65536
.venv/bin/python inference/quantization/kv_cache_dtype.py --dtypes auto fp8 --batch 64
```

vLLM scripts default to `--kv-cache-bytes 2GiB`, a fixed KV budget that also skips vLLM's
free-memory profiling (which asserts when another process touches the GPU during start-up).

## 6. Results

**Environment.** HF path on the RTX PRO 6000 Blackwell (sm_120, GPU 1); vLLM path on the RTX A6000
(sm_86, GPU 0, no FP8 tensor cores). Both shared with other jobs (*pilot, shared GPU*).
Model Qwen2.5-0.5B-Instruct; perplexity on 65,536 wikitext-2 tokens (65,472 scored), 1024-token
windows; decode = 128 greedy tokens on ultrachat prompts, batch 1 and 8. vLLM: 2 GiB fixed KV
budget, `max_model_len` 2048, CUDA graphs on. Raw rows: `reports/L08_inference_optimization/
results/quant_table_Qwen2.5-0.5B-Instruct.json` and the run directories named below.

### 6.1 transformers + bitsandbytes (run `20261029_022610_bench_quant_4674aae4e65d`)

| mode | weights (GiB) | peak mem, ppl eval (GiB) | ppl | prefill tok/s | decode tok/s b=1 | ms/step b=1 | decode tok/s b=8 |
|---|---|---|---|---|---|---|---|
| bf16 | 0.92 | 1.52 | 16.26 | 68.9k | 119.7 | 8.36 | 828 |
| fp16 | 0.92 | 1.52 | 16.24 | 69.4k | 118.8 | 8.42 | 770 |
| int8_bnb (LLM.int8) | 0.59 | 1.19 | 16.27 | 17.8k | **22.9** | 43.6 | 160 |
| nf4_bnb (NF4 + double quant) | 0.42 | 1.03 | 17.92 | 36.8k | 61.1 | 16.4 | 342 |
| awq (HF loader) | fail | | | | | | `ImportError: Loading an AWQ quantized model requires gptqmodel` |
| gptq (HF loader) | fail | | | | | | `ImportError: Loading a GPTQ quantized model requires optimum` |

Reading: on this model the embedding / `lm_head` matrix (151,936 x 896, tied) is 27% of the
parameters and is never quantised, so INT8 saves 36% rather than 50% and NF4 saves 54%. LLM.int8
costs essentially nothing in perplexity but is **5x slower** than bf16 at batch 1 (its
outlier decomposition runs the matmul twice, in INT8 and fp16); NF4 costs +1.7 ppl and 2x
decode latency. On a launch-bound 0.5B model neither saves time; they are memory tools.

### 6.2 vLLM kernels (run `20261029_022606_bench_quant_6994ea75ab3d`, A6000)

| mode | checkpoint | weights (GiB) | ppl | prefill tok/s | decode tok/s b=1 | ms/step b=1 | decode tok/s b=8 |
|---|---|---|---|---|---|---|---|
| bf16 | base | 0.93 | 16.26 | 24.9k | 209 | 4.79 | 1580 |
| gptq_int8 (Marlin W8A16) | `-GPTQ-Int8` | 0.60 | 16.24 | 34.1k | 496 | 2.01 | 3627 |
| gptq (Marlin W4A16) | `-GPTQ-Int4` | 0.43 | 19.05 | 35.1k | **565** | 1.77 | 4117 |
| awq (Marlin W4A16) | `-AWQ` | 0.43 | 19.39 | 34.3k | 548 | 1.83 | 4052 |
| fp8 (dynamic, sm_86 fallback) | base | 0.59 | **2.9e6 (broken)** | 18.9k | 231 | 4.34 | 1245 |
| nf4_bnb (vLLM in-flight) | base | 0.45 | 17.93 | 21.1k | 215 | 4.65 | 972 |

**FP8 on a GPU with FP8 tensor cores.** The same `--load fp8` run on the RTX PRO 6000 Blackwell
(sm_120, GPU 1, 1 GiB KV budget, `max_num_seqs` 32; run `20261029_023711_ppl_vllm_fp8_Qwen2.5-0.5B-Instruct_bf2fe6a6e2d4`):
weights 0.59 GiB, **ppl 16.45** (+0.19 over bf16), 764 tok/s at batch 1 and 5,494 tok/s at batch 8.
So dynamic FP8 W8A8 is numerically fine where the native kernel exists, and the garbage result above
is specific to vLLM 0.23's sm_86 fallback path. (The Blackwell numbers are not comparable in speed to
the A6000 rows: different GPU, different co-tenants.)

**Caveat on the speed columns of 6.2.** The bf16 and nf4 rows were measured ~10 minutes after the
AWQ / GPTQ / FP8 rows (their first attempt OOM-ed when a co-tenant grabbed the GPU), under
visibly heavier co-tenant load: the identical-shape bf16 model ran at 4.8 ms/step there versus
1.8-2.0 ms/step for the W4/W8 rows, a gap no 0.5B weight-only kernel can produce. The
back-to-back re-check in 6.2b is the number to read for speed; 6.2 is the number to read for
perplexity and memory (which are load-independent).

Reading: the vLLM path is 2-5x faster than eager transformers at batch 1 before any quantisation
(CUDA graphs + fused kernels, see `../compilation/`). Among the kernel-backed schemes, INT8
weights are free in perplexity (16.24 vs 16.26) and INT4 costs +2.8 (GPTQ) / +3.1 (AWQ) ppl on
this small model, which is a lot: 4-bit hurts small models far more than large ones. vLLM's
in-flight NF4 reproduces the bitsandbytes perplexity (17.93 vs 17.92). The W4A16 Marlin kernels
decode ~14% faster than W8A16 at batch 1 within the same measurement window.
FP8 on the A6000 (no FP8 tensor cores, vLLM selects a Marlin W8A16 fallback) produced garbage
perplexity: the kernel path is present but numerically wrong for this model on sm_86 in vLLM
0.23.0, a bug worth reporting and a reminder to validate every quantised deployment with a
perplexity check before trusting its speed.

### 6.2b Back-to-back speed re-check (same window, 8k-token perplexity)

Five vLLM engines started one after the other on the A6000 (run `20261029_023853_bench_quant_fa9c7d49cbc4`,
perplexity on 8,192 tokens only, so the ppl column is a sanity check against a *different* token
subset than 6.2 and is not comparable to it). Same 2 GiB KV budget, same prompts, same window of
co-tenant load (still shared, so treat the ratios as indicative).

| mode | weights (GiB) | decode tok/s b=1 | ms/step b=1 | decode tok/s b=8 | ms/step b=8 | vs bf16 (b=1 / b=8) |
|---|---|---|---|---|---|---|
| bf16 | 0.93 | 159 | 6.30 | 1171 | 6.83 | 1.00x / 1.00x |
| gptq_int8 (W8A16 Marlin) | 0.60 | 253 | 3.95 | 1281 | 6.24 | 1.60x / 1.09x |
| gptq (W4A16 Marlin) | 0.43 | 262 | 3.82 | 2008 | 3.98 | 1.65x / 1.71x |
| awq (W4A16 Marlin) | 0.43 | 265 | 3.77 | 2002 | 4.00 | 1.67x / 1.71x |
| nf4_bnb (in-flight, dequantise GEMM) | 0.45 | 157 | 6.36 | 766 | 10.45 | 1.00x / 0.65x |

Reading: with the engines measured back to back, the W4A16 Marlin kernels are ~1.65x faster
than bf16 at batch 1 *and* at batch 8, and the W8A16 kernel helps at batch 1 but not at batch 8.
Even on a 0.5B model the vLLM step is partly weight-bandwidth-bound (CUDA graphs removed the
launch floor), so reading 4x fewer weight bytes shows. The bitsandbytes-style in-flight NF4 gets
the memory saving with none of the speed and a 35% loss at batch 8: its GEMM dequantises to bf16
first. This is the practical distinction: *packed-weight kernels* (Marlin, Machete, ExLlama)
trade memory for speed; *dequantise-then-matmul* schemes trade memory for time.

### 6.3 KV-cache dtype (vLLM, fixed KV budget)

Qwen2.5-0.5B-Instruct on the A6000, **2 GiB fixed KV budget**, `max_model_len` 2048, 64 concurrent
prompts x 128 greedy tokens, perplexity on 65,472 wikitext-2 tokens
(run `20261029_023735_kv_cache_dtype_f503734241d1`).

| `kv_cache_dtype` | KV capacity (tokens) | vLLM max concurrency at 2048 tok/req | ppl | output tok/s, batch 64 | ms/step |
|---|---|---|---|---|---|
| auto (bf16) | 174,752 | 85.3x | 16.26 | 12,739 | 5.02 |
| fp8 (e4m3, no scales) | 349,520 | 170.7x | 16.89 | 18,582 | 3.44 |

Reading: with the budget held fixed, FP8 KV holds exactly **2.0x** the tokens (the KV block count
is the only thing that changed), at +0.63 perplexity for the calibration-free e4m3 format. At 64
concurrent sequences the decode step was 30% faster with FP8 KV, consistent with the attention
kernel reading half the bytes per token, although part of that gap may be co-tenant noise. The
earlier smoke run (8 k tokens, batch 8, free-memory sizing instead of a fixed budget) gave 18.97
vs 19.76 ppl and only a 1.37x capacity gain, which is why capacity comparisons must fix the
budget: free memory on a shared GPU differed between the two engine start-ups.


## 7. Limitations and next experiments

* The 7B bf16 vs AWQ comparison, where INT4 weight-only should show a real decode speed-up, is
  pending an exclusive GPU (the pilots ran next to a 90 GB vLLM server; see REPORT.md).
* Perplexity on 64k tokens of wikitext-2 is a coarse quality signal; task accuracy (GSM8K via
  `post_training/eval_gsm8k.py`) should accompany any quantisation claim.
* HF-side AWQ/GPTQ need `autoawq` / `gptqmodel`, which are not installed; the vLLM path is the
  one with kernels here.
* Next: calibrated FP8 KV scales, INT8 KV, AWQ for the 7B model with and without Marlin, and the
  batch sweep that locates the memory-bound -> compute-bound crossover per scheme.

## 8. References

* Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, 2022.
* Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023 (NF4, double quantisation).
* Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023.
* Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
* Frantar et al., *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*, 2024.
* Micikevicius et al., *FP8 Formats for Deep Learning*, 2022.
* vLLM documentation: Quantization, `kv_cache_dtype`, `kv_cache_memory_bytes`.
