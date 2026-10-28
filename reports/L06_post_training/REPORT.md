# L06 report — Post-training pipeline: where does the step time go?

## Question
In SFT → DPO → GRPO, which phase dominates an optimizer step (rollout, reward, reference forward,
learner forward/backward, optimizer), and does the resulting policy actually get better on the
task the reward was meant to encode?

## Environment
| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 48 GB (sm 8.6), `CUDA_VISIBLE_DEVICES=0`, **shared with other resident jobs (27–46 GB in use)** |
| Software | torch 2.11.0+cu130, transformers 5.7.0, trl 1.9.2, peft 0.14.0, vllm 0.23.0, driver 580.95 / CUDA 13.0 |
| Models | Qwen/Qwen2.5-0.5B-Instruct (pilot), HuggingFaceTB/SmolLM2-135M-Instruct (smokes) |
| Data | openai/gsm8k `main` (train prompts for GRPO, test for eval, seed 0 shuffles); HuggingFaceTB/smoltalk everyday-conversations (SFT); trl-lib/ultrafeedback_binarized (DPO) |
| Rewards | `correctness` = 1.0 if the parsed answer equals the GSM8K gold; `format` = 0.2 if a `#### n` line is present |

Run directories (`runs/L06_post_training/`, each with manifest, config, metrics.jsonl, summary.json;
GRPO runs also hold the LoRA adapter and `phases.png`; evals hold per-question `records.jsonl`):

```
20261029_020352_sft_tiny_lora16_7f647860f004 20261029_020420_dpo_tiny_lora16_c3839fa7b282 20261029_020452_grpo_tiny_G2_L48_hf_134355ee11e8 20261029_020458_eval_tiny_base_eac540b18c96 20261029_020700_dpo_tiny_lora16_e073b9a206e0 20261029_020714_grpo_tiny_G2_L48_hf_195575e74298 20261029_020812_eval_small_base_3d03f73fa948 20261029_020844_grpo_small_G4_L128_hf_96305c48bb88 20261029_020919_eval_small_grpo_adapter_c6c3acd19c75 20261029_020950_grpo_small_G4_L128_vllm_374815379d8a 20261029_021101_dpo_small_lora16_inline_ref_b319e2667e56 20261029_021116_dpo_small_lora16_precomp_ref_11583026f39e 20261029_021148_grpo_small_G4_L128_vllm_9ce8ee4c69bd 
```

## Hypothesis
1. With HF `generate` the rollout is > 70 % of a GRPO step at G = 4, L = 128; the verifiable reward is negligible.
2. The DPO reference forward is a large fraction of the policy-forward phase and can be removed by precomputation.
3. 25 GRPO steps with a 0.2 format reward will raise the format rate; whether accuracy rises is the question.

## Baseline
Base Qwen2.5-0.5B-Instruct, greedy, GSM8K test n = 100: accuracy 0.30 (95 % CI 0.21–0.39), format rate 0.11.
Run `20261029_020812_eval_small_base_3d03f73fa948`.

## Change
25 GRPO steps, LoRA r = 16, 2 prompts × G = 4, `max_completion_length` 128, lr 5e-5, β = 0, HF rollouts
(run `20261029_020844_grpo_small_G4_L128_hf_96305c48bb88`), then the same eval with the adapter merged
(run `eval_small_grpo_adapter`). DPO: 12 steps with the reference computed inline vs precomputed.

## Method
* Phase timing by wrapping the trainer instance's methods (`post_training/timing_callback.py`), CUDA-synchronised,
  nested phases subtracted to exclusive shares; optimizer window from the `TrainerCallback` hooks; the first step is dropped.
* Accuracy with a 2 000-sample bootstrap CI over per-question correctness; format rate and mean completion length
  reported next to it so a format change cannot masquerade as a reasoning change.
* Reward curve and `completions/mean_length` from TRL's log history.

## Results
GRPO step (0.94 s): rollout 81.3 %, reward 0.1 %, scoring 0.7 %, learner forward 7.3 %, backward 9.6 %,
optimizer 0.1 %; peak allocated 8.27 GiB; ≈ 188 generated tokens/s for the batch.

Quality: accuracy **0.30 → 0.07** (CI 0.03–0.12), format rate 0.11 → 0.98, mean completion 743 → 27 chars;
training reward 0.21 → 0.28 (first/last 5-step means), mean completion length 18 tokens.

DPO reference A/B: not measured — both runs OOM'd on the shared GPU (0.6 GB free at launch; this process at 5.98 GB), retry skipped at 1.8 GB free; pending exclusive GPU.

vLLM colocate: not measured — util 0.08 left no KV-cache blocks beside the resident learner; util 0.12 failed at startup with 0.18 GiB free on the shared GPU; pending exclusive GPU (expected floor ~0.15–0.2 on 48 GB).

Smokes (tiny model) for SFT, DPO, GRPO and eval all complete; see `results/smoke.log`.

## Bottleneck analysis
* **Rollout dominates (81 %)** because HF `generate` decodes 8 sequences for up to 128 steps with one
  kernel launch per token per layer and no CUDA graphs; the learner then needs 0.16 s for the same tokens.
  Doubling the prompts per step would roughly double learner time but leave the decode loop length unchanged,
  so the share would move only slowly; cutting L or switching the engine moves it immediately.
* **Reward is free** (1 ms): string parsing. A reward model would add a forward pass comparable to the learner forward.
* **The optimiser found the cheap reward.** Format (dense, 0.2, achievable with 3 tokens) was learned within
  a few steps; correctness (sparse, 1.0) requires long reasoning that the policy stopped producing. Shorter
  completions also shorten the rollout, so the system got faster while getting worse: 23 accuracy points lost.
* **Memory** at this scale is logits-bound (8 × ~380 tokens × 152 k vocab in fp32), not weight-bound.

## Limits
* Single run per condition on a shared GPU; n = 100 eval (±9 points).
* Reward design was intentionally naive to expose the failure mode; it is not a recommended recipe.
* vLLM colocate could not be given enough memory beside other jobs; server mode (separate GPU) not attempted.
* SFT/DPO were only smoked on the tiny model beyond the DPO A/B.

## Reproduction commands
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python post_training/eval_gsm8k.py --model small --n 100 --label eval_small_base
python post_training/grpo.py --model small --steps 25 --batch-size 8 --num-generations 4 --max-completion-length 128 --n-samples 400 --name grpo_small_G4_L128_hf
python post_training/eval_gsm8k.py --model small --n 100 --adapter runs/L06_post_training/<grpo run>/adapter --label eval_small_grpo_adapter
python post_training/dpo.py --model small --steps 12 --batch-size 2 --max-length 384 --n-samples 64 --name dpo_small_lora16_inline_ref
python post_training/dpo.py --model small --steps 12 --batch-size 2 --max-length 384 --n-samples 64 --precompute-ref --name dpo_small_lora16_precomp_ref
python post_training/grpo.py --model small --steps 25 --batch-size 8 --num-generations 4 --max-completion-length 128 --use-vllm --vllm-gpu-util 0.12
```

## Next experiments
1. Correctness-gated format reward + length floor; 100+ steps; eval n ≥ 500 before/after.
2. Rollout engine A/B on an exclusive GPU: HF vs vLLM colocate vs vLLM server on GPU 1 (weight-sync time, overlap).
3. G ∈ {4, 8, 16} at fixed B·L and L ∈ {128, 256, 512}: rollout share as a function of decode length.
4. `num_iterations` 1 vs 2 vs 4: rollout amortisation vs clip-ratio drift.
5. DPO with `precompute_ref_log_probs` over multiple epochs where the up-front pass pays back.
