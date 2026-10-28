"""L06 step 3: GRPO with a verifiable GSM8K reward, with rollout / reward / learner decomposition.

    python post_training/grpo.py --smoke
    python post_training/grpo.py --model small --steps 25 --num-generations 4 --max-completion-length 128
    python post_training/grpo.py --model small --use-vllm --vllm-gpu-util 0.08     # colocated vLLM rollouts

Each GRPO step: sample G completions per prompt (rollout), score them (reward), normalise the
rewards within the group (advantages), then run the policy forward/backward on the sampled
tokens (learner). With HF `generate` the rollout is a sequential decode loop of up to
`max_completion_length` steps and usually dominates the step; that is the bottleneck this lab
measures, and vLLM rollouts + weight sync is the standard fix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset

from llm_systems_zoo import PhaseTimer, Run
from llm_systems_zoo.plotting import PALETTE, plt, save_fig, setup_style
from post_training.common import (
    LAB,
    apply_smoke,
    common_args,
    cuda_peak_gib,
    gsm8k_rows,
    lora_config,
    reset_cuda_peak,
    resolve_model_id,
)
from post_training.rewards import REWARD_FUNCS
from post_training.timing_callback import PhaseTimingCallback, instrument_trainer


def plot_phases(cb_summary: dict, reward_curve: list[tuple[int, float]], path: Path, title: str) -> None:
    if not cb_summary:
        return
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    phases = cb_summary["mean_phase_s"]
    names = [k for k, v in sorted(phases.items(), key=lambda kv: -kv[1]) if v > 0]
    ax1.barh(names, [phases[k] for k in names], color=PALETTE[: len(names)])
    ax1.set_xlabel("mean seconds per optimizer step")
    ax1.set_title(f"{title}: step decomposition")
    ax1.invert_yaxis()
    if reward_curve:
        xs, ys = zip(*reward_curve)
        ax2.plot(xs, ys, marker="o", color=PALETTE[0])
        ax2.set_xlabel("step")
        ax2.set_ylabel("mean reward")
        ax2.set_title("reward curve")
    save_fig(fig, path)


def main(argv=None) -> int:
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), 25)
    p.add_argument("--num-generations", type=int, default=4, help="group size G")
    p.add_argument("--max-completion-length", type=int, default=128)
    p.add_argument("--steps-per-generation", type=int, default=None, help="reuse one generation batch for this many steps")
    p.add_argument("--num-iterations", type=int, default=1, help="policy updates per generation batch (μ)")
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty to the reference; 0 skips the reference forward")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rewards", default="correctness,format", help="comma list of reward functions")
    p.add_argument("--use-vllm", action="store_true", help="vLLM colocate rollouts (vllm_mode='colocate')")
    p.add_argument("--vllm-gpu-util", type=float, default=0.08)
    a = p.parse_args(argv)
    apply_smoke(a)
    if a.smoke:
        a.num_generations, a.max_completion_length = 2, 48
    from trl import GRPOConfig, GRPOTrainer

    model_id = resolve_model_id(a.model)
    rows = gsm8k_rows("train", a.n_samples, a.seed)
    ds = Dataset.from_list(rows)
    reward_funcs = [REWARD_FUNCS[r.strip()] for r in a.rewards.split(",") if r.strip()]
    cfg = {k: v for k, v in vars(a).items() if k not in ("runs_root", "out", "name")}
    name = a.name or f"grpo_{a.model}_G{a.num_generations}_L{a.max_completion_length}_{'vllm' if a.use_vllm else 'hf'}"
    with Run(LAB, name, cfg, root=a.runs_root, model={"id": model_id, "lora": a.lora, "beta": a.beta},
             workload={"dataset": "openai/gsm8k/main/train", "n_prompts": len(ds), "num_generations": a.num_generations,
                       "max_completion_length": a.max_completion_length,
                       "batch_size": a.batch_size, "grad_accum": a.grad_accum, "rewards": a.rewards},
             execution={"steps": a.steps, "generation_backend": "vllm_colocate" if a.use_vllm else "hf_generate",
                        "num_iterations": a.num_iterations, "steps_per_generation": a.steps_per_generation}) as run:
        args = GRPOConfig(
            output_dir=str(run.path("trainer")), max_steps=a.steps, per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr if a.lora == 0 else 5e-5,
            num_generations=a.num_generations, max_completion_length=a.max_completion_length,
            steps_per_generation=a.steps_per_generation, num_iterations=a.num_iterations, beta=a.beta, temperature=a.temperature,
            bf16=a.bf16 and torch.cuda.is_available(), gradient_checkpointing=a.grad_checkpointing, logging_steps=1,
            save_strategy="no", report_to="none", seed=a.seed, dataloader_num_workers=0, warmup_steps=0,
            use_vllm=a.use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gpu_util,
            log_completions=False,
        )
        timer = PhaseTimer(sync_cuda=torch.cuda.is_available())
        cb = PhaseTimingCallback(timer, "GRPOTrainer", run.log_metric)
        trainer = GRPOTrainer(model=model_id, reward_funcs=reward_funcs, args=args, train_dataset=ds, callbacks=[cb],
                              peft_config=lora_config(a.lora) if a.lora else None)
        wrapped = instrument_trainer(trainer, timer)
        run.note(f"instrumented phases: {wrapped}")
        reset_cuda_peak()
        t0 = time.perf_counter()
        trainer.train()
        wall = time.perf_counter() - t0
        adapter_dir = None
        if a.lora:
            adapter_dir = run.path("adapter")
            trainer.model.save_pretrained(str(adapter_dir))
        hist = [h for h in trainer.state.log_history if "reward" in h]
        reward_curve = [(h["step"], h["reward"]) for h in hist]
        comp_len = [h.get("completions/mean_length") for h in hist if h.get("completions/mean_length") is not None]
        ph = cb.summary()
        gen_tokens_per_step = (sum(comp_len) / len(comp_len) if comp_len else 0.0) * a.batch_size * a.grad_accum
        summary = {
            "name": name, "model": model_id, "steps": a.steps, "wall_s": wall, "generation_backend": "vllm_colocate" if a.use_vllm else "hf_generate",
            "num_generations": a.num_generations, "max_completion_length": a.max_completion_length,
            "prompts_per_step": a.batch_size * a.grad_accum // a.num_generations, "completions_per_step": a.batch_size * a.grad_accum,
            "first_reward": hist[0]["reward"] if hist else None, "final_reward": hist[-1]["reward"] if hist else None,
            "mean_reward_last5": sum(h["reward"] for h in hist[-5:]) / max(1, len(hist[-5:])) if hist else None,
            "mean_reward_first5": sum(h["reward"] for h in hist[:5]) / max(1, len(hist[:5])) if hist else None,
            "mean_completion_length": sum(comp_len) / len(comp_len) if comp_len else None,
            "rollout_tokens_per_s": (gen_tokens_per_step / ph["mean_phase_s"]["rollout"]) if ph and ph["mean_phase_s"].get("rollout") else None,
            "phases": ph, "reward_curve": reward_curve, **cuda_peak_gib(), "run_dir": str(run.dir),
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
        }
        run.save_summary(summary)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        plot_phases(ph, reward_curve, run.path("phases.png"), name)
        if not a.smoke:
            plot_phases(ph, reward_curve, out / f"{name}_phases.png", name)
        print(f"[grpo] {name}: reward {summary['mean_reward_first5']} → {summary['mean_reward_last5']} (first/last 5), wall {wall:.1f}s, "
              f"peak alloc {summary['peak_allocated_gib']:.2f} GiB")
        if ph:
            print("phase share:", {k: round(v, 3) for k, v in ph["phase_share"].items()}, "unaccounted", round(ph["unaccounted_share"], 3))
        print("run dir:", run.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
