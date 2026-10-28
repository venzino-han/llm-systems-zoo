"""L06 step 2: Direct Preference Optimization (DPO) with TRL on ultrafeedback_binarized, LoRA.

    python post_training/dpo.py --smoke
    python post_training/dpo.py --model small --steps 30 [--precompute-ref]

DPO needs reference log-probs for both the chosen and rejected completion. With LoRA the
reference is the same model with adapters disabled (no second copy in memory), but it still
costs an extra forward pass per pair unless `--precompute-ref` pays that cost once up front.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from llm_systems_zoo import PhaseTimer, Run
from post_training.common import (
    LAB,
    apply_smoke,
    common_args,
    cuda_peak_gib,
    dpo_dataset,
    lora_config,
    reset_cuda_peak,
    resolve_model_id,
)
from post_training.timing_callback import PhaseTimingCallback, instrument_trainer


def main(argv=None) -> int:
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), 30)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--precompute-ref", action="store_true", help="precompute reference log-probs once (trades a pass up front for a cheaper step)")
    a = p.parse_args(argv)
    apply_smoke(a)
    from trl import DPOConfig, DPOTrainer

    model_id = resolve_model_id(a.model)
    ds = dpo_dataset(a.n_samples, a.seed)
    cfg = {k: v for k, v in vars(a).items() if k not in ("runs_root", "out", "name")}
    name = a.name or f"dpo_{a.model}_{'lora' + str(a.lora) if a.lora else 'full'}{'_precomp' if a.precompute_ref else ''}"
    with Run(LAB, name, cfg, root=a.runs_root, model={"id": model_id, "lora": a.lora, "beta": a.beta},
             workload={"dataset": "trl-lib/ultrafeedback_binarized", "n_samples": len(ds), "max_length": a.max_length,
                       "batch_size": a.batch_size, "grad_accum": a.grad_accum},
             execution={"steps": a.steps, "precompute_ref_log_probs": a.precompute_ref}) as run:
        args = DPOConfig(
            output_dir=str(run.path("trainer")), max_steps=a.steps, per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr if a.lora == 0 else 5e-5, beta=a.beta,
            max_length=a.max_length, precompute_ref_log_probs=a.precompute_ref,
            bf16=a.bf16 and torch.cuda.is_available(), gradient_checkpointing=a.grad_checkpointing, logging_steps=1,
            save_strategy="no", report_to="none", seed=a.seed, dataloader_num_workers=0, warmup_steps=max(1, a.steps // 20),
        )
        timer = PhaseTimer(sync_cuda=torch.cuda.is_available())
        cb = PhaseTimingCallback(timer, "DPOTrainer", run.log_metric)
        trainer = DPOTrainer(model=model_id, ref_model=None, args=args, train_dataset=ds, callbacks=[cb],
                             peft_config=lora_config(a.lora) if a.lora else None)
        wrapped = instrument_trainer(trainer, timer)
        run.note(f"instrumented phases: {wrapped}")
        reset_cuda_peak()
        t0 = time.perf_counter()
        result = trainer.train()
        wall = time.perf_counter() - t0
        hist = [h for h in trainer.state.log_history if "loss" in h]
        last = hist[-1] if hist else {}
        summary = {
            "name": name, "model": model_id, "steps": a.steps, "wall_s": wall, "train_runtime_s": result.metrics.get("train_runtime"),
            "first_loss": hist[0]["loss"] if hist else None, "final_loss": last.get("loss"),
            "reward_margin_last": last.get("rewards/margins"), "reward_accuracy_last": last.get("rewards/accuracies"),
            "reward_margin_mean_last5": sum(h.get("rewards/margins", 0.0) for h in hist[-5:]) / max(1, len(hist[-5:])),
            "reward_accuracy_mean_last5": sum(h.get("rewards/accuracies", 0.0) for h in hist[-5:]) / max(1, len(hist[-5:])),
            "phases": cb.summary(), "ref_precompute_total_s": timer.totals.get("ref_precompute", 0.0),
            **cuda_peak_gib(), "run_dir": str(run.dir),
        }
        run.save_summary(summary)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        print(f"[dpo] {name}: loss {summary['first_loss']} → {summary['final_loss']}, margin {summary['reward_margin_mean_last5']:.3f}, "
              f"acc {summary['reward_accuracy_mean_last5']:.2f}, wall {wall:.1f}s, peak alloc {summary['peak_allocated_gib']:.2f} GiB, "
              f"phases {summary['phases'].get('phase_share')}")
        print("run dir:", run.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
