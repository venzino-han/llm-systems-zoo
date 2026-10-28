"""L06 step 1: supervised fine-tuning (SFT) with TRL on smoltalk, LoRA by default.

    python post_training/sft.py --smoke                       # tiny model, 3 steps
    python post_training/sft.py --model small --steps 30      # 0.5B pilot

SFT is the baseline every later stage (DPO, GRPO) is compared against: it needs no reference
model and no rollouts, so its step is a single forward/backward on fixed targets.
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
    lora_config,
    reset_cuda_peak,
    resolve_model_id,
    sft_dataset,
)
from post_training.timing_callback import PhaseTimingCallback, instrument_trainer


def main(argv=None) -> int:
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), 30)
    p.add_argument("--packing", action="store_true", help="TRL packing (bfd) with padding-free batches")
    a = p.parse_args(argv)
    apply_smoke(a)
    from trl import SFTConfig, SFTTrainer

    model_id = resolve_model_id(a.model)
    ds = sft_dataset(a.n_samples, a.seed)
    cfg = {k: v for k, v in vars(a).items() if k not in ("runs_root", "out", "name")}
    name = a.name or f"sft_{a.model}_{'lora' + str(a.lora) if a.lora else 'full'}"
    with Run(LAB, name, cfg, root=a.runs_root, model={"id": model_id, "lora": a.lora},
             workload={"dataset": "HuggingFaceTB/smoltalk/everyday-conversations", "n_samples": len(ds), "max_length": a.max_length,
                       "batch_size": a.batch_size, "grad_accum": a.grad_accum, "packing": a.packing},
             execution={"steps": a.steps}) as run:
        out_dir = run.path("adapter")
        args = SFTConfig(
            output_dir=str(run.path("trainer")), max_steps=a.steps, per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr if a.lora == 0 else 1e-4, max_length=a.max_length,
            bf16=a.bf16 and torch.cuda.is_available(), gradient_checkpointing=a.grad_checkpointing, packing=a.packing,
            padding_free=a.packing, logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
            dataloader_num_workers=0, warmup_steps=max(1, a.steps // 20),
        )
        timer = PhaseTimer(sync_cuda=torch.cuda.is_available())
        cb = PhaseTimingCallback(timer, "SFTTrainer", run.log_metric)
        trainer = SFTTrainer(model=model_id, args=args, train_dataset=ds, callbacks=[cb],
                             peft_config=lora_config(a.lora) if a.lora else None)
        wrapped = instrument_trainer(trainer, timer)
        run.note(f"instrumented phases: {wrapped}")
        n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in trainer.model.parameters())
        reset_cuda_peak()
        t0 = time.perf_counter()
        result = trainer.train()
        wall = time.perf_counter() - t0
        if a.lora:
            trainer.model.save_pretrained(str(out_dir))
        hist = [h for h in trainer.state.log_history if "loss" in h]
        summary = {
            "name": name, "model": model_id, "steps": a.steps, "wall_s": wall, "train_runtime_s": result.metrics.get("train_runtime"),
            "train_tokens_per_s": result.metrics.get("train_tokens_per_second"), "train_samples_per_s": result.metrics.get("train_samples_per_second"),
            "first_loss": hist[0]["loss"] if hist else None, "final_loss": hist[-1]["loss"] if hist else None,
            "mean_token_accuracy_last": hist[-1].get("mean_token_accuracy") if hist else None,
            "n_trainable": n_train, "n_params": n_total, "trainable_frac": n_train / n_total,
            "phases": cb.summary(), **cuda_peak_gib(), "adapter_dir": str(out_dir) if a.lora else None, "run_dir": str(run.dir),
        }
        run.save_summary(summary)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        print(f"[sft] {name}: loss {summary['first_loss']} → {summary['final_loss']}, wall {wall:.1f}s, "
              f"peak alloc {summary['peak_allocated_gib']:.2f} GiB, phases {summary['phases'].get('phase_share')}")
        print("run dir:", run.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
