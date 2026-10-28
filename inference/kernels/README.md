# L08c. Kernels: attention backends and fused Triton ops

**Question.** Where does a decode or prefill step actually spend its time, and what does a fused
kernel (attention, softmax, RMSNorm) buy over the composed PyTorch ops at each shape?
**Status.** Implemented, Measured (pilot on a shared GPU).
**Hardware needed.** 1 CUDA GPU; the Triton kernels need `triton>=3`, the attention benchmark
uses whatever backends PyTorch / FlashInfer / flash-attn expose on that GPU.

## 1. What the technique is

A transformer layer is a handful of large matmuls plus many small element-wise / reduction ops
(softmax, normalisation, residual adds, activation functions). The matmuls run close to the
hardware roofline; the small ops do not, because each one is a separate kernel that reads the
whole tensor from HBM and writes it back. **Kernel fusion** rewrites several of those passes as
one kernel so the tensor crosses the memory bus once. Two families matter for LLM inference:

### Fused attention (FlashAttention and friends)

Naive attention materialises the `L x L` score matrix:

```
S = Q K^T / sqrt(d)      # L x L, written to HBM
P = softmax(S)           # read + write again
O = P V                  # read again
```

Memory is `O(L^2)` per head and the three passes are each memory-bound. FlashAttention
(Dao et al. 2022) tiles Q/K/V into SRAM-sized blocks and computes the softmax *online*
(running max and running sum per row), so the score matrix never exists in HBM: memory becomes
`O(L)` and the arithmetic intensity rises enough to use the tensor cores. Variants differ in
tiling, hardware targeting and features:

| Backend | What it is | Notes |
|---|---|---|
| `sdpa_math` | PyTorch reference: materialised scores, fp32 softmax | correctness oracle, `O(L^2)` memory |
| `sdpa_efficient` | xFormers-derived memory-efficient attention | `O(L)` memory, older tiling |
| `sdpa_flash` | FlashAttention-2 kernel shipped inside PyTorch | fp16/bf16 only, head dim <= 256 |
| `sdpa_cudnn` | cuDNN fused attention (PyTorch >= 2.5) | often the fastest on Hopper/Blackwell |
| `flashinfer_prefill` | FlashInfer's prefill kernel (used by vLLM/SGLang) | built for paged KV; also handles the plain case |
| `flash_attn_3` | FlashAttention-3 (Hopper-specific: warp specialisation, FP8) | **sm_90 only** |

All of them compute the same function; the benchmark checks each against `sdpa_math`.

### Fused element-wise kernels in Triton

`triton_softmax.py` and `triton_rmsnorm.py` are one-program-per-row kernels in the style of the
Triton tutorials: load a row once into registers, reduce, write once. RMSNorm is the
normalisation used by Llama/Qwen/Mistral (`x / sqrt(mean(x^2) + eps) * w`); the **fused
residual-add + RMSNorm** variant also folds the `h = h + residual` that precedes every norm,
saving another read and write of the hidden state, and is exactly the op vLLM ships as
`fused_add_rms_norm`.

Cost model for a row-wise op over an `R x C` tensor of element size `b` bytes:

```
bytes moved (fused)   = 2 * R * C * b            (one read + one write)
bytes moved (naive)   = ~5 * 2 * R * C * b       (max, sub, exp, sum, div each read+write)
time                  >= bytes moved / HBM bandwidth
```

At batch 1 decode these ops are pure bandwidth, so the fused version is bounded by `2RCb / BW`
and the naive one by roughly five times that, plus a kernel launch (~5 µs) per op. The
benchmarks report *effective GB/s* (`2RCb / time`) so a kernel can be read against the GPU's
HBM bandwidth directly.

### Educational vs production

The Triton kernels here use one block per row and a power-of-two block size. Production kernels
(vLLM's CUDA ops, TensorRT-LLM plugins, Liger-Kernel) tile wide rows across several blocks, use
vectorised loads, tune `num_warps` per shape, and are fused further (e.g. RMSNorm + quantise to
FP8). The attention benchmark calls the production kernels directly; the Triton ops are there to
show the mechanism and to be compared against `torch.compile`, which generates a similar fused
kernel automatically.

## 2. When to use it

* Fused attention: always, for any sequence longer than a few hundred tokens. The memory saving
  alone decides whether long contexts fit. Which backend to pick depends on the GPU generation
  (cuDNN and FA3 on Hopper/Blackwell, FA2 elsewhere), head dim, dtype, and whether you need
  paged KV (FlashInfer / vLLM's kernels).
* Hand-written Triton kernels: when `torch.compile` cannot fuse the pattern (dynamic control
  flow, custom quantisation, paged memory layouts), when you need a specific numerical recipe, or
  when the framework's op is measurably off the bandwidth roofline at your shape. Otherwise
  `torch.compile` gets most of the fusion win for free, as the RMSNorm numbers below show.

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| Fused attention: `O(L)` memory instead of `O(L^2)`; 5-50x faster at long L; enables long context at all | Backend availability depends on GPU generation, dtype, head dim, masks; numerics differ slightly from the fp32 reference (bf16 accumulation order) |
| Fused element-wise ops: approach HBM bandwidth; remove launch overhead (matters most at batch 1) | A hand-written kernel is code you must test across shapes and dtypes and re-tune per GPU |
| Triton gives Python-level control with near-CUDA performance for memory-bound ops | Triton kernels still lag hand-tuned CUDA/CUTLASS on compute-bound GEMMs and on the newest architectures until the compiler catches up |
| `torch.compile` produces comparable fused kernels with no kernel code | Compile time, recompilation on shape changes, and no control over the generated kernel |

What you pay: engineering and validation effort, portability. What you gain: the difference
between launch-bound and bandwidth-bound execution for the small ops, and between quadratic and
linear memory for attention.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `triton_softmax.py` | Row softmax kernel, correctness vs fp32 reference, bandwidth benchmark vs `torch.softmax`, naive eager and `torch.compile` | GPU |
| `triton_rmsnorm.py` | RMSNorm and fused add+RMSNorm kernels, correctness vs the HF implementation, latency benchmark vs eager and compiled | GPU |
| `attention_bench.py` | SDPA backends (math / efficient / flash / cuDNN), FlashInfer, FA3 and naive attention across L and head dim, with peak memory and error vs math | GPU |
| `../../tests/test_l08_kernels.py` | CPU check of the naive attention reference; Triton correctness suites under `--gpu` | CPU / GPU |

## 5. How to run

```bash
# correctness suites + quick benchmark (under 2 minutes)
.venv/bin/python inference/kernels/triton_softmax.py --smoke
.venv/bin/python inference/kernels/triton_rmsnorm.py --smoke
.venv/bin/python inference/kernels/attention_bench.py --smoke

# the pilots quoted below
.venv/bin/python inference/kernels/triton_softmax.py --rows 2048 --cols 128 512 1024 2048 4096 8192 16384
.venv/bin/python inference/kernels/triton_rmsnorm.py --rows 4096
.venv/bin/python inference/kernels/attention_bench.py --heads 8 --seq 256 512 1024 2048 4096 8192 --head-dim 64 128

# GPU unit tests
.venv/bin/python -m pytest tests/test_l08_kernels.py --gpu
```

## 6. Results

**Environment.** NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120, 96 GB), torch 2.11.0+cu130,
triton 3.6.0, flashinfer 0.6.12, flash-attn-3 3.0.0b1 (Hopper build). The GPU was **shared with
other jobs during these pilots** (a 90 GB vLLM server and other builders), so absolute numbers are
noisy and labelled *pilot, shared GPU*; relative orderings were stable across repeats.
Timer: `triton.testing.do_bench` (10 warm-up, 50 reps) for the Triton scripts, CUDA events (5
warm-up, 20 reps) for attention. Raw data: `reports/L08_inference_optimization/results/
{triton_softmax,triton_rmsnorm,attention_bench}.json`.

### 6.1 Backend availability on this GPU (sm_120)

| Backend | Status |
|---|---|
| SDPA math / efficient / flash / cuDNN | ok |
| FlashInfer prefill + decode | ok |
| Triton 3.6 | ok |
| flash-attn-3 (`flash_attn_interface`) | **fails: `no kernel image is available for execution on the device`** (Hopper-only wheel). The failure is a hard CUDA error that aborts the process, so `attention_bench.py` gates it by compute capability. |

### 6.2 Causal attention, batch 1, 8 heads, bf16 (run `20261029_022442_attention_bench_bc7f8e92cd20`)

Latency in ms, peak extra memory in MiB (allocator delta while the kernel runs).

| L | d | naive fp32 | sdpa_math | sdpa_efficient | sdpa_flash | sdpa_cudnn | flashinfer |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | 0.245 / 70 | 0.277 / 85 | 0.039 / 2 | 0.016 / 6 | 0.039 / 2 | 0.027 / 37 |
| 2048 | 64 | 1.564 / 270 | 1.683 / 322 | 0.085 / 4 | 0.034 / 20 | 0.037 / 4 | 0.051 / 42 |
| 4096 | 64 | 6.617 / 1060 | OOM (2 GiB) | 0.336 / 8 | 0.217 / 8 | 0.090 / 8 | 0.186 / 52 |
| 8192 | 64 | OOM | OOM | 1.013 / 16 | 0.429 / 16 | 0.470 / 16 | 0.547 / 72 |
| 2048 | 128 | 1.761 / 280 | 1.921 / 340 | 0.156 / 8 | 0.057 / 40 | 0.061 / 8 | 0.085 / 52 |
| 8192 | 128 | OOM | OOM | 1.904 / 32 | 0.804 / 32 | 0.728 / 32 | 0.988 / 112 |

Reading: the materialised-score implementations grow quadratically in both time and memory (4x
per doubling of L) and ran out of the ~2 GB this shared GPU had free at L = 4096-8192; the fused
kernels grow linearly in memory and roughly 2-2.5x per doubling in time (the FLOPs still grow
quadratically, but they now run on tensor cores). At L = 2048 fused attention is 40-50x faster
than the math backend. cuDNN and FA2 trade places depending on L and head dim; FlashInfer's
constant ~30 MiB is its workspace, not a per-token cost. Max abs error vs the fp32 math backend
is 0.8-1.6e-2 for every fused backend (bf16 output rounding), i.e. they are numerically
equivalent to each other. The OOM rows are a consequence of the shared GPU, not of the kernels.

### 6.3 RMSNorm, 4096 tokens, bf16 (run `20261029_022433_triton_rmsnorm_5db1d23b1bcc`), ms

| hidden | Triton | HF eager | `torch.compile` | Triton fused add+norm | eager add+norm | compiled add+norm |
|---|---|---|---|---|---|---|
| 896 (Qwen2.5-0.5B) | 0.023 | 0.079 | 0.026 | 0.039 | 0.083 | 0.057 |
| 1536 (Qwen2.5-1.5B) | 0.023 | 0.160 | 0.023 | 0.048 | 0.110 | 0.069 |
| 2048 | 0.064 | 0.192 | 0.030 | 0.110 | 0.213 | 0.076 |
| 3584 (Qwen2.5-7B) | 0.054 | 0.302 | 0.049 | 0.165 | 0.375 | 0.153 |
| 4096 | 0.087 | 0.371 | 0.066 | 0.144 | 0.426 | 0.203 |
| 8192 | 0.111 | 1.325 | 0.185 | 0.365 | 1.614 | 0.379 |

Reading: the one-pass Triton kernel is 3.4-12x faster than the eager composition (six kernel
launches), and `torch.compile` reaches the same ballpark automatically; at 2048-4096 hidden the
Inductor kernel beats the untuned Triton one (block size / warp count were not tuned per shape).
The fused add+norm variant is only ~1.5x slower than the plain norm while replacing two ops.
Correctness: all 15 shape/dtype combinations within tolerance of the HF reference
(`correctness.json` in the run directory).

### 6.4 Softmax, 2048 rows, fp16, effective GB/s (= 2 x bytes / time; run `20261029_022536_triton_softmax_828d06acd384`)

| columns | Triton | `torch.softmax` | naive eager | naive `torch.compile` |
|---|---|---|---|---|
| 128 | 74 | 195 | 38 | 99 |
| 512 | 250 | 296 | 107 | 490 |
| 1024 | 1083 | 367 | 99 | 893 |
| 2048 | 678 | 423 | 117 | 484 |
| 4096 | 614 | 386 | 129 | 709 |
| 8192 | 706 | 534 | 182 | 461 |
| 16384 | 543 | 747 | 94 | 371 |

Reading: below ~512 columns every variant is launch-bound (tens of GB/s regardless of kernel);
from 1024 to 8192 the one-block-per-row Triton kernel moves 1.3-3x more bytes per second than
`torch.softmax`; at 16384 columns a single block per row no longer has enough parallelism and
PyTorch's multi-block kernel wins. The naive eager composition sits at 100-180 GB/s throughout,
which is the cost of five passes. (The GPU's HBM peak is ~1.8 TB/s; nothing here reaches it
because rows are short and the GPU was shared.)

## 7. Limitations and next experiments

* Shared-GPU pilots: absolute latencies carry noise from co-running jobs; re-run on an exclusive
  GPU before quoting numbers externally, and add repeats with error bars.
* The Triton kernels are not tuned (block size, `num_warps`, multi-block rows). Tuning per shape
  with `triton.autotune` is the obvious next step and would also make the 16384-column case fair.
* FA3 could not be exercised on sm_120 (Hopper-only build); Blackwell-specific kernels
  (cuDNN 9 / FlashInfer's Blackwell paths) are what matters here and are covered implicitly.
* Next: paged-attention decode kernels (the `inference/kv_cache` lab's territory), a fused
  RMSNorm + FP8 quantise kernel, and a profiler trace of a full decode step to attribute time to
  attention vs GEMM vs element-wise ops.

## 8. References

* Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022.
* Dao, *FlashAttention-2*, 2023; Shah et al., *FlashAttention-3*, 2024.
* Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, 2025.
* Tillet et al., *Triton: an intermediate language and compiler for tiled neural network computations*, 2019; Triton tutorials (fused softmax, layer norm).
* Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.
* PyTorch `torch.nn.attention.sdpa_kernel` documentation.
