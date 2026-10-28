# L01 Report — LLM Memory & Compute Anatomy

Status: **pilot, shared GPU** (first measurement pass; exclusive-GPU re-run pending). Figures and raw summaries: `results/`.

## Question
Where does GPU memory go in training, prefill and decode, and why do analytic estimates differ from what the allocator and `nvidia-smi` report? Secondary: is HF eager decode bandwidth bound on this GPU?

## Environment
* GPU 0: NVIDIA RTX A6000 48 GB (sm_86, PCIe, no NVLink), shared with two resident vLLM servers (~32 GB used by others during the pilot).
* Driver 580.95.05, CUDA 13.0, torch 2.11.0+cu130, transformers 5.7.0, peft 0.14.0, Python 3.10.12.
* Models: Qwen/Qwen2.5-0.5B-Instruct (inference, LoRA), HuggingFaceTB/SmolLM2-135M-Instruct (full fine-tuning step).
* Full manifests: `results/pilot_full_grid_manifest.json`, `results/pilot_foreach_false_manifest.json`, `results/pilot_lora_r16_manifest.json`; run dirs `runs/L01_memory_anatomy/20261029_0203*_pilot_full_grid_3a0dfc30a61b`, `…_pilot_foreach_false_8c0c7297e709`, `…_pilot_lora_r16_6dcc0b2054e0`, `…_profile_b1_s1024_b7ab8a3f911b`, `…_profile_b8_s1024_6db6380c47b5`.

## Hypothesis
1. The analytic estimate matches `max_memory_allocated` within ~10 % for inference and bf16 training; the remaining gap is explained by (a) allocator reserve vs allocated, (b) optimizer temporaries, (c) loss-path logits copies, (d) attention backend.
2. HF eager decode on a 0.5B model runs far below the bandwidth roofline and its step time is nearly independent of batch size (launch bound).

## Baseline
`memory_calculator.estimate_training / estimate_inference` (pure Python) and `flops_calculator` roofline bounds with nominal A6000 specs (155 TFLOPS bf16 dense, 768 GB/s).

## Change
Measured the same conditions with `measure_memory.py` (allocated / reserved / nvidia-smi) and `profile_prefill_decode.py` (CUDA-synchronised timers, torch.profiler); one ablation on the AdamW `foreach` implementation.

## Method
* Inference grid: batch {1, 4} × prompt {256, 1024}, 64 decode steps, bf16, SDPA, synthetic random tokens; peaks reset per condition.
* Training grid: {fp32, bf16 autocast, bf16 autocast + gradient checkpointing} × {1×256, 2×256, 1×512, 2×512}; two optimizer steps, the second one measured (steady state). LoRA r=16 on all linear layers of the 0.5B model (bf16 base).
* Profiling: 1 warm-up, 5 timed repeats; profiler traces captured separately for prefill and decode.
* Fair-comparison rules from `docs/benchmark_protocol.md`: one variable per comparison; estimate and measurement on identical shapes; all conditions reported including the ones where the estimate is wrong.

## Results
See `fundamentals/README.md` §6 for the full tables. Headlines:
* Inference prefill peak: estimate within −4 % … +2 % of measured over the 2×2 grid; KV cache slope matches (12 KiB/token).
* Training (bf16 autocast): estimate within 1–4 % of measured once the `foreach` temporaries (0.30 GiB at small shapes) are accounted for; checkpointing −20 % memory / +53 % time at 2×512; mixed precision −19 % memory / −50 % time vs fp32.
* fp32: SDPA's math path adds one `a·s²` tensor per layer; the measurement lands between the fused and the fully-materialised estimates.
* LoRA on a 152k-vocabulary model: estimator 21–27 % low; the shortfall is ~1.1 MiB per token ≈ two extra fp32 logits tensors in the loss path.
* Decode: 7.96 ms/step at batch 1, 8.32 ms at batch 8 (126 → 961 tok/s); 16–17 % of the bandwidth roofline; prefill 41–45 % of the compute roofline.

## Bottleneck analysis
* Memory: for 0.1–0.5B models the static term (16 B/param fp32 training) and the `b·s·V` logits dominate; per-layer activations only matter at `b·s ≥ 1k`. Gradient checkpointing cannot reduce the logits term.
* Time: eager decode issues ~170 kernels per step; at 8 ms/step and 1 GB of weights the loop is launch bound. The remedy is fewer launches (CUDA graphs, compilation) or more work per launch (batching, continuous batching in an engine).

## Limits
* Shared GPU: timing noise (±10 % plausible), memory exact.
* Full-FT measured on a 135M model; Qwen2.5-0.5B full-FT pending exclusive GPU (needs ~10 GiB).
* Estimator assumes Llama/Qwen dense blocks; no MoE / sliding-window support.
* Nominal GPU peaks from datasheets, not micro-benchmarked.

## Reproduction commands
See `fundamentals/README.md` §5; CPU tests: `.venv/bin/python -m pytest tests/test_l01_calculators.py -q` (15 passed).

## Next experiments
1. Exclusive-GPU re-run of both grids and the profiles; add the Blackwell GPU.
2. Add correction terms (optimizer temporaries, loss-path logits) to the estimator and re-validate.
3. Attention-backend ablation (`eager` / `sdpa` / flash) for fp32 and bf16.
4. Feed the measured KV slope and overhead into L02's capacity model.
