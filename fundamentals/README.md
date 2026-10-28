# L01. LLM Memory & Compute Anatomy

**Question.** Where does an LLM's GPU memory go during training, prefill and decode, and why do analytic estimates differ from what the allocator and `nvidia-smi` report?
**Status.** Explain ✔ · Implement ✔ · Benchmark: *Measured (pilot, shared GPU)* · Operate: n/a
**Hardware needed.** Calculators: CPU only. Measurements: 1 GPU with ≥ 5 GB free (pilot ran on an RTX A6000 shared with other jobs).

## 1. What the technique is

Before touching a GPU you can predict, to within ~10 %, how much memory a model needs and whether a phase will be bound by compute or by memory bandwidth. This lab builds those two cost models and then checks them against measurements, because the *gap* between estimate and measurement is where the engineering lives (allocator caching, optimizer temporaries, loss-path copies, attention backends).

### Memory categories

For a decoder-only transformer with `N` parameters, `L` layers, hidden size `h`, `a` query heads, `a_kv` KV heads of dimension `d`, MLP width `f`, vocabulary `V`, batch `b`, sequence length `s`, and `p` bytes per element:

| Category | Bytes | Notes |
|---|---|---|
| Parameters | `N · p` | counted exactly from the architecture (`ModelConfig.param_breakdown`) |
| Gradients | `N_trainable · p` | zero for frozen base weights under LoRA |
| Optimizer states | AdamW `2 · N_trainable · p_state` | torch keeps states in the parameter dtype; fp32 master weights add `4 · N` |
| Activations (per layer) | `≈ b·s·p·(8h + 4f + 2·a_kv·d)` | + `2·a·s²·b·p` if attention scores are materialised (no fused kernel) |
| Activations with checkpointing | `L · b·s·h·p` + one layer | only layer inputs are stored; one layer is recomputed at a time |
| Logits | `b·s·V·(p + 4)` | the loss up-casts to fp32; with `V = 152k` this dominates small models |
| KV cache | `2 · L · a_kv · d · p` per token | K and V for every layer; GQA divides this by `a / a_kv` |
| Runtime overhead | measured, not estimated | CUDA context, cuBLAS/attention workspaces, allocator fragmentation |

### Compute and the roofline

Per token, a forward pass costs `2N + 4·L·h·s_ctx` FLOPs (linear layers plus attention at context `s_ctx`); training costs three times that (`≈ 6N` per token, the Kaplan/Chinchilla rule of thumb). Prefill multiplies the weight reads over `s` tokens, so its arithmetic intensity is `~s` FLOPs per byte of weights and it is compute bound once `s` reaches a few hundred. Decode reads every weight (and the whole KV cache) to produce one token per sequence, so its intensity is `≈ b` FLOPs/byte: far below the ridge point of any GPU (A6000 ≈ 200 FLOP/B, H100 ≈ 300 FLOP/B). Decode is memory-bandwidth bound, and batching is the lever that raises its intensity.

`roofline_time = max(FLOPs / peak_FLOPS, bytes / bandwidth)`; the fraction of roofline a measurement achieves tells you whether the kernels, or the framework around them, are the bottleneck.

### Three observers, three numbers

`torch.cuda.memory_allocated` counts live tensors (what the estimate models). `memory_reserved` is what the caching allocator holds from the driver, including cached free segments. `nvidia-smi` per-process usage adds the CUDA context and library workspaces. The lab records all three.

Minimal implementation vs production: the calculators are a few hundred lines of pure Python; the measurement uses stock Hugging Face eager execution, deliberately *not* an optimised engine, so the "fraction of roofline" numbers show how much a serving engine (vLLM, L02/L03) and CUDA graphs / compilation (L08) have to recover.

## 2. When to use it

* Sizing: will this model + batch + context fit, and how many KV tokens remain for serving (feeds L02's capacity model)?
* Choosing a training recipe: full fine-tuning vs LoRA vs QLoRA, mixed precision, gradient checkpointing, sequence length, before burning GPU hours.
* Diagnosing an OOM: compare the measured peak against the estimate and the reserved/allocated gap to tell fragmentation from genuine demand.
* Deciding where optimisation effort goes: a decode loop at 16 % of the bandwidth roofline is a launch-overhead problem, not a kernel problem.

Not useful when the architecture is exotic (MoE routing, sliding-window or linear attention, multimodal encoders): the estimator's per-layer formula assumes a Llama/Qwen-style dense block and must be extended.

## 3. Trade-offs

| Technique quantified here | Advantages | Disadvantages |
|---|---|---|
| Mixed precision (bf16 autocast) | ~2× less activation memory; ~2× faster matmuls on tensor cores; fused attention kernels become available | fp32 weights/grads/states remain (16 B/param); bf16 weight copies cached by autocast; numerics differ from fp32 |
| Gradient checkpointing | activations drop from `O(L · layer)` to `O(L · layer_input)`; enables longer sequences / larger batches | one extra forward per layer: measured step time +50–100 % at small shapes; does nothing for weights/optimizer/logits |
| LoRA (r = 16) | gradients and optimizer states shrink to the adapters (1.8 % of parameters here); base weights can be bf16 or 4-bit | activations are *not* reduced (gradients still flow through every layer); extra adapter matmuls; quality ceiling vs full FT |
| Batching in decode | tokens/s scales almost linearly with batch while per-step time barely moves | per-request latency grows slightly; KV cache grows linearly with batch |
| Analytic estimate vs profiling | instant, no GPU, works for hypothetical hardware | ±10–20 % error; blind to allocator behaviour, optimizer temporaries, loss-path copies; must be validated per framework version |

What you pay: estimates are only as good as the formula's assumptions; measurements are only as good as the control of the environment (a shared GPU perturbs timing but not per-process memory).

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `memory_calculator.py` | `ModelConfig`, exact parameter counts, `estimate_training` / `estimate_inference`, CLI | CPU |
| `flops_calculator.py` | FLOPs per phase, bytes moved, arithmetic intensity, roofline helper with nominal GPU specs | CPU |
| `measure_memory.py` | measures allocated / reserved / `nvidia-smi` memory for prefill, decode and a training step; estimate-vs-measured figure | GPU |
| `profile_prefill_decode.py` | CUDA-synchronised prefill/decode timing, roofline fraction, torch.profiler traces, top kernels | GPU |
| `../tests/test_l01_calculators.py` | 15 tests: parameter counts vs published sizes, KV formula, checkpointing, LoRA, FLOPs, roofline | CPU |
| `../reports/L01_memory_anatomy/` | `REPORT.md` and `results/` (run summaries, manifests, figures) | — |

## 5. How to run

```bash
# CPU: calculators and tests
.venv/bin/python fundamentals/memory_calculator.py --model Qwen/Qwen2.5-0.5B-Instruct --batch 2 --seq-len 512 --amp --grad-ckpt
.venv/bin/python fundamentals/memory_calculator.py --model Qwen/Qwen2.5-7B-Instruct --lora-rank 16 --base-dtype nf4 --dtype bf16 --amp --batch 1 --seq-len 1024
.venv/bin/python fundamentals/flops_calculator.py --model Qwen/Qwen2.5-7B-Instruct --gpu "H100 SXM" --batch 1 32 256
.venv/bin/python -m pytest tests/test_l01_calculators.py -q

# GPU (commands that produced the results below; GPU 0 = RTX A6000, shared)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
.venv/bin/python fundamentals/measure_memory.py --smoke                      # ~1 min sanity run
.venv/bin/python fundamentals/measure_memory.py --name pilot_full_grid       # 2 prompt lengths × 2 batches; 3 training modes × 4 shapes
.venv/bin/python fundamentals/measure_memory.py --name pilot_foreach_false --skip-inference --optim-foreach false --train-modes fp32 amp --train-grid 1x256 2x512
.venv/bin/python fundamentals/measure_memory.py --name pilot_lora_r16 --skip-inference --train-model Qwen/Qwen2.5-0.5B-Instruct --lora-rank 16 --train-modes amp amp+ckpt --train-grid 1x512 2x512
.venv/bin/python fundamentals/profile_prefill_decode.py --batch 1 --prompt-len 1024 --decode-steps 64 --repeats 5
.venv/bin/python fundamentals/profile_prefill_decode.py --batch 8 --prompt-len 1024 --decode-steps 64 --repeats 5
```

Each GPU run writes `runs/L01_memory_anatomy/<timestamp>_<name>_<cfghash>/` with `manifest.json`, `summary.json`, `metrics.jsonl` and the figure.

## 6. Results

All numbers: **pilot, shared GPU** (RTX A6000 48 GB, driver 580.95, CUDA 13.0, torch 2.11.0+cu130, transformers 5.7.0; two unrelated vLLM servers were resident on the same GPU, so timings are indicative and per-process memory is exact). Manifests: `reports/L01_memory_anatomy/results/*_manifest.json`.

### 6.1 Inference: estimate vs measured (Qwen2.5-0.5B-Instruct, bf16, SDPA) — run `pilot_full_grid`

| batch | prompt | prefill est (GiB) | prefill peak alloc (GiB) | decode est (GiB) | decode peak alloc (GiB) | KV est (MiB) | Δalloc after 64 decode steps (MiB) | ms / decode step |
|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0.936 | 0.951 | 0.937 | 0.941 | 3.8 | 13.2 | 8.2 |
| 1 | 1024 | 0.982 | 0.984 | 0.983 | 0.950 | 12.8 | 22.2 | 7.3 |
| 4 | 256 | 0.983 | 0.984 | 0.986 | 0.955 | 15.0 | 25.8 | 7.7 |
| 4 | 1024 | 1.167 | 1.123 | 1.170 | 1.000 | 51.0 | 65.8 | 7.2 |

* Prefill peaks are within **−4 % … +2 %** of the estimate. Weights (0.92 GiB) dominate a 0.5B model; the KV cache for 4 × 1088 tokens is only 51 MiB (12 KiB/token with GQA 14:2).
* The decode estimate is *above* the measured peak because the estimate carries the prefill transient (one layer's working set at prompt length); the decode peak itself is weights + KV + a few MiB.
* The Δalloc column grows with the estimated KV slope (3.8 → 12.8 MiB estimate ≈ 13.2 → 22.2 MiB measured, +9 MiB each) plus a ~9–15 MiB constant: logits, cache metadata and the last step's hidden states.
* Decode step time is **flat at ~7–8 ms from batch 1 to 4** — the first hint that eager decode is not bandwidth bound.

### 6.2 Training step: estimate vs measured (SmolLM2-135M-Instruct, full fine-tuning, AdamW)

| mode | b × s | est (GiB) | est, scores materialised | peak alloc (GiB) | peak reserved (GiB) | ms / step |
|---|---|---|---|---|---|---|
| fp32 | 1×256 | 2.401 | 2.532 | 2.594 | 2.898 | 56 |
| fp32 | 2×256 | 2.797 | 3.060 | 2.689 | 3.049 | 93 |
| fp32 | 1×512 | 2.797 | 3.324 | 2.822 | 3.248 | 76 |
| fp32 | 2×512 | 3.589 | 4.644 | 4.020 | 4.248 | 117 |
| amp (bf16) | 1×256 | 2.477 | 2.543 | 2.573 | 2.820 | 52 |
| amp | 2×256 | 2.698 | 2.830 | 2.576 | 2.957 | 51 |
| amp | 1×512 | 2.698 | 2.962 | 2.585 | 2.957 | 51 |
| amp | 2×512 | 3.142 | 3.669 | 3.253 | 3.279 | 59 |
| amp + ckpt | 1×256 | 2.339 | 2.341 | 2.575 | 2.621 | 88 |
| amp + ckpt | 2×256 | 2.422 | 2.427 | 2.586 | 2.801 | 90 |
| amp + ckpt | 1×512 | 2.422 | 2.431 | 2.584 | 2.799 | 107 |
| amp + ckpt | 2×512 | 2.589 | 2.607 | 2.607 | 3.211 | 90 |

Optimizer implementation check (run `pilot_foreach_false`, `AdamW(foreach=False)`):

| mode | b × s | peak alloc, foreach (default) | peak alloc, foreach=False |
|---|---|---|---|
| fp32 | 1×256 | 2.594 | **2.285** |
| fp32 | 2×512 | 4.020 | 4.023 |
| amp | 1×256 | 2.573 | **2.274** |
| amp | 2×512 | 3.253 | 3.257 |

Interpretation:

* **The static part is exact.** 135M × 16 B = 2.01 GiB of weights + grads + Adam states; every small-shape measurement sits just above it.
* **At small shapes the peak happens inside `optimizer.step()`, not in backward.** The default `foreach` AdamW allocates temporaries for a chunk of parameters; switching it off lowers the peak by 0.30 GiB at 1×256 and changes nothing at 2×512, where activations + logits set the peak during backward. The estimator does not model optimizer temporaries, which is why it is 8 % low at 1×256 and 5 % high at 2×512 with `foreach=False`.
* **The attention backend matters for fp32.** At fp32 2×512 the measurement (4.02 GiB) is 12 % above the fused-attention estimate (3.59) and 13 % below the "scores materialised" estimate (4.64): SDPA's fp32 math path keeps one `a·s²` probability tensor per layer rather than two. With bf16 autocast the flash kernel is used and the estimate is within 1–4 %.
* **Checkpointing flattens memory and costs time.** 2×512 drops from 3.25 → 2.61 GiB (−20 %) for +53 % step time (59 → 90 ms); at 1×256 it saves nothing because activations were never the peak. Reserved memory can still be 0.6 GiB above allocated (2×512) — allocator caching, invisible to the estimator.
* **Mixed precision is the cheapest win:** 2×512 fp32 → amp is −19 % memory and −50 % step time, with no recomputation.

### 6.3 LoRA step (Qwen2.5-0.5B-Instruct, bf16 base, r = 16 on all linear layers, 8.80 M trainable = 1.8 %) — run `pilot_lora_r16`

| mode | b × s | est (GiB) | peak alloc (GiB) | peak reserved (GiB) | ms / step |
|---|---|---|---|---|---|
| amp | 1×512 | 2.081 | 2.652 | 2.867 | 62 |
| amp | 2×512 | 3.111 | 4.225 | 4.613 | 133 |
| amp + ckpt | 1×512 | 1.531 | 1.936 | 2.197 | 103 |
| amp + ckpt | 2×512 | 2.011 | 2.830 | 3.266 | 119 |

The estimator is 21–27 % low here, and the shortfall scales with `b·s` (0.57 GiB per 512 tokens ≈ 1.1 MiB/token). With `V = 151 936` a single fp32 logits tensor is 0.58 MiB/token, so the loss path holds about two more logits-sized tensors than the formula's `p + 4` (shifted labels/logits and the fp32 softmax gradient). For small models with large vocabularies the logits, not the layers, are the activation budget; checkpointing cannot touch them, which is why `amp+ckpt` still needs 2.83 GiB at 2×512.

### 6.4 Where the time goes (Qwen2.5-0.5B, bf16, prompt 1024, 64 decode steps, 5 repeats) — runs `profile_b1_s1024`, `profile_b8_s1024`

| batch | prefill (ms) | prefill tok/s | prefill % of compute roofline | decode ms/step | decode tok/s | decode % of bandwidth roofline | effective GB/s |
|---|---|---|---|---|---|---|---|
| 1 | 15.2 | 67 400 | 45 % | 7.96 | 126 | 16 % | 126 |
| 8 | 133.0 | 61 600 | 41 % | 8.32 | 961 | 17 % | 131 |

* Prefill reaches 41–45 % of the nominal 155 TFLOPS: GEMM kernels (`cutlass … s16816gemm`, `ampere_bf16_s16816gemm_256x128`) dominate the trace, as expected for a compute-bound phase.
* Decode moves only ~130 GB/s of the 768 GB/s the A6000 offers. Per step the model launches ~170 GEMV/GEMM kernels plus elementwise ops; at ~8 ms/step with batch-independent cost, the loop is **CPU launch-bound**, not bandwidth-bound. Batch 8 is therefore 7.6× the throughput for +4.5 % step time. Removing this overhead (CUDA graphs, `torch.compile`, a serving engine) is the subject of L02/L03/L08.
* Chrome traces (`trace_prefill.json`, `trace_decode.json`) are in the run directories under `runs/` (git-ignored).

## 7. Limitations and next experiments

* Shared GPU: timings carry noise from co-resident jobs; memory numbers are per-process and unaffected. A re-run on an exclusive GPU is queued (`exclusive-GPU measurement pending`).
* The full fine-tuning step was measured on SmolLM2-135M to stay under the 5 GB budget; the Qwen2.5-0.5B full-FT step (≈ 8 GiB static + activations) is pending an exclusive GPU.
* Synthetic random token inputs: realistic for memory, but attention patterns do not affect memory anyway.
* The estimator ignores optimizer temporaries, loss-path copies and allocator caching by design; the measured gaps above are the calibration data for a v2 with per-framework correction terms.
* Next: fp32-vs-bf16 attention backend ablation via `attn_implementation`, `nvidia-smi` vs `reserved` gap across model sizes, and the same grid on the Blackwell GPU once it is free.

## 8. References

* Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, 2022 (activation memory formula).
* Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 (`6ND`).
* Williams, Waterman, Patterson, *Roofline: an insightful visual performance model*, CACM 2009.
* Pope et al., *Efficiently Scaling Transformer Inference*, 2022 (prefill/decode intensity, KV cache).
* PyTorch docs: *CUDA semantics — memory management* (allocated vs reserved), `torch.optim` `foreach` implementation notes.
* Qwen2.5 technical report (architecture: GQA 14:2, 24 layers, tied embeddings).
