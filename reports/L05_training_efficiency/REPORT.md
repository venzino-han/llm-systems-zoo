# L05 report — Training efficiency and distributed training

## Question
What does each single-GPU efficiency switch (precision, checkpointing, accumulation, packing,
LoRA/QLoRA) cost and buy in memory, throughput and (provisionally) loss, and how do DDP and FSDP2
behave when a second GPU is added over PCIe?

## Environment
| Item | Value |
|---|---|
| Host | 12th Gen Intel i9-12900KS, 24 threads, 110 GiB RAM, single node |
| GPU 0 | NVIDIA RTX A6000, 48 GB, sm 8.6 (all single-GPU runs) |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell Workstation, 96 GB, sm 12.0 (2-GPU runs only) |
| Interconnect | PCIe host bridge (`nvidia-smi topo -m`: PHB), no NVLink |
| Software | Python 3.10.12, torch 2.11.0+cu130, transformers 5.7.0, peft 0.14.0, bitsandbytes 0.48.2, driver 580.95, CUDA 13.0 |
| Sharing | **Both GPUs were shared with other resident jobs during every run** (27–37 GB in use on GPU 0, ~92 GB on GPU 1). Numbers are pilot-grade. |
| Models | Qwen/Qwen2.5-0.5B-Instruct (sweep), HuggingFaceTB/SmolLM2-135M-Instruct (distributed smoke), random 2-layer Llama (CPU tests) |
| Data | HuggingFaceTB/smoltalk `everyday-conversations` train, shuffled with seed 0, chat template applied, truncated to seq_len |

Each run directory under `runs/L05_training_efficiency/` holds `manifest.json` (hardware,
software, git commit, command), `config.json`, `metrics.jsonl` (per-step loss, step time,
tokens, memory) and `summary.json`. Run directories created for this report:

```
20261029_015640_fp32_effdf6af0b85 20261029_015654_bf16_autocast_62229afce13e 20261029_015703_bf16_weights_4782486353f8 20261029_015710_bf16+ckpt_e05448cafa40 20261029_015720_bf16+accum4_090bb407c9d8 20261029_015732_bf16+packing_a89d602e48b1 20261029_015744_bf16+lora16_88b80c1291d3 20261029_015752_qlora16_50dccccc73c6 20261029_020205_ddp_w1_strong_6e4eed9b4449 20261029_020221_ddp_w2_strong_5f25f1e23cd7 20261029_020238_fsdp2_w1_strong_6e4eed9b4449 20261029_020257_fsdp2_w2_strong_5f25f1e23cd7 
```

## Hypothesis
1. AMP doubles throughput without changing peak memory at this model size (static state dominates).
2. Checkpointing will save little at 0.5 B / seq 512 because activations are not the peak.
3. Accumulation with batch 1 micro-steps costs throughput; packing raises tokens/s.
4. LoRA removes the optimizer cost; QLoRA trades speed for memory.
5. DDP scales sub-linearly on a heterogeneous PCIe pair; FSDP2 will not beat DDP for a model that fits.

## Baseline
`bf16_autocast`: fp32 weights + AdamW, bf16 autocast, batch 4 × seq 512, no checkpointing, dynamic
padding, lr 1e-4, 30 steps with the first 5 excluded from timing.

## Change
One switch per condition relative to the baseline (table in `training/README.md` §6.1).
Distributed: same loop under DDP or FSDP2, strong scaling with global batch 8, 1 vs 2 ranks.

## Method
* Metrics: effective (non-padding) tokens/s over the timed steps; `torch.cuda.max_memory_allocated`
  and `max_memory_reserved`; forward/backward/optimizer share from `PhaseTimer` with CUDA sync;
  mean of the last 5 losses.
* Static memory estimate: weights + trainable grads + 2 AdamW moments at their dtypes.
* DDP communication: exposed = fwd+bwd with gradient sync − fwd+bwd under `no_sync()` (lr 0 so
  the probe does not train); total = sum of NCCL kernel device time from `torch.profiler` over 3 steps.
* Resume check: continuous 8-step run vs 4 steps + checkpoint + rebuild + 4 steps on the same
  seeded batch stream; CPU, fp32, bitwise comparison.

## Results
See `training/README.md` §6 for the full tables and figures
(`results/sweep_table.md`, `results/memory_vs_throughput.png`, `results/scaling_table.md`,
`results/scaling_efficiency.png`). Headline numbers (pilot, shared GPU):

| Switch (vs bf16 AMP baseline 5 503 tok/s, 10.13 GiB) | tokens/s | peak alloc GiB |
|---|---|---|
| fp32 | 3 048 (0.55×) | 10.28 |
| pure bf16 weights | 8 265 (1.50×) | 6.21 |
| + activation checkpointing | 4 779 (0.87×) | 9.25 |
| + accumulation 1 × 4 | 3 497 (0.64×) | 9.26 (reserved 10.45) |
| + packing | 8 323 (1.51×) | 13.26 |
| + LoRA r16 | 8 563 (1.56×) | 6.40 |
| QLoRA r16 | 6 648 (1.21×) | 5.21 |

| 1 → 2 GPUs, strong scaling, tiny model + LoRA | speedup | efficiency | NCCL ms/step (exposed) |
|---|---|---|---|
| DDP | 1.47× | 0.73 | 10.0 (4.5) |
| FSDP2 | 0.55× | 0.27 | 108.7 (n/a) |

Resume check: max |Δloss| = 0 (padded and packed), checkpoint 16 KB for the tiny model.

## Bottleneck analysis
* Baseline step: 21 % forward, 37 % backward, **41 % optimizer**. AdamW over 494 M fp32
  parameters streams ~16 bytes/param × several passes; once matmuls run in bf16 the optimizer
  is the largest single phase. Pure bf16 (−32 % optimizer bytes) and LoRA (−98 % optimizer
  parameters) attack exactly that phase, which is why they are the fastest conditions.
* Peak memory composition at the baseline: 7.4 GiB static + ~1.2 GiB fp32 logits + activations
  ≈ 10.1 GiB allocated; reserved 15.6 GiB (caching-allocator headroom of 5.5 GiB).
* Checkpointing targets activations, which are < 2 GiB here, so its effect is bounded by that.
* Packing's throughput gain is mostly "more tokens per kernel" rather than "less padding"; the
  O(L²) mask and larger token count raise the peak to 13.3 GiB.
* FSDP2 on 2 PCIe GPUs: 70 % of device kernel time is all-gather/reduce-scatter; the whole bf16
  weight set is gathered twice and gradients scattered once per step while only adapters train.

## Limits
* Single repeat per condition; shared GPUs; 30-step loss is not a quality measure; same lr for
  all parameterisations (LoRA vs full FT quality is not comparable from this table).
* The 2-GPU pair is heterogeneous (sm 8.6 + sm 12.0): the slower rank bounds every step, so
  efficiency numbers understate what a homogeneous pair would give. PCIe-only; no NVLink data.
* fp16 autocast + GradScaler is implemented but not measured. Weak scaling not measured.
* Full-FT 0.5B on 2 GPUs, and FSDP2 `--no-reshard`, are pending an exclusive GPU window.

## Reproduction commands
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python training/train_efficiency.py --sweep --model small --dataset smoltalk --batch-size 4 --seq-len 512 --steps 30 --warmup-steps 5
python training/train_efficiency.py --model random-tiny --device cpu --precision fp32 --steps 8 --resume-test
for strat in ddp fsdp; do for w in 1 2; do
  CUDA_VISIBLE_DEVICES=$([ $w = 1 ] && echo 0 || echo 0,1) python -m torch.distributed.run --nproc_per_node $w \
    training/distributed/${strat}_train.py --model tiny --dataset smoltalk --precision bf16_weights --lora 16 \
    --scaling strong --global-batch 8 --seq-len 256 --steps 20 --warmup-steps 5 --probe-comm
done; done
python training/distributed/scaling_analysis.py
```

## Next experiments
1. Exclusive-GPU rerun of the sweep with 3 repeats and lr tuned per parameterisation; add a
   held-out eval loss so the quality column means something.
2. Equal-tokens-per-step packed vs padded comparison; `cu_seqlens` path with a varlen kernel.
3. Sequence-length sweep (512 → 4 k) to find where checkpointing and packing's O(L²) mask flip sign.
4. Full-FT 0.5B DDP vs FSDP2 (with and without reshard-after-forward), `bucket_cap_mb` sweep,
   weak scaling, and NCCL bandwidth micro-benchmark on this PCIe pair.
