"""L05: single-GPU training efficiency sweep and checkpoint-resume verification.

Examples (from the repo root):

    # CPU smoke, no download
    python training/train_efficiency.py --model random-tiny --device cpu --smoke

    # one configuration on a GPU
    python training/train_efficiency.py --model small --dataset smoltalk --precision bf16_autocast \
        --grad-checkpointing --steps 40

    # the predefined pilot grid + memory/throughput figure
    python training/train_efficiency.py --sweep --model small --dataset smoltalk

    # checkpoint resume verification
    python training/train_efficiency.py --resume-test --model random-tiny --device cpu
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from llm_systems_zoo import Run
from llm_systems_zoo.io import dump_json
from llm_systems_zoo.plotting import PALETTE, plt, save_fig, setup_style
from training.common import (
    TINY_RANDOM,
    TrainConfig,
    build_model,
    build_tokenizer,
    iter_batches,
    load_samples,
    make_optimizer,
    resolve_model_id,
    train_loop,
)

LAB = "L05_training_efficiency"

# The pilot grid: one switch at a time relative to the bf16-autocast baseline.
SWEEP_GRID: list[dict[str, Any]] = [
    {"name": "fp32", "precision": "fp32"},
    {"name": "bf16_autocast", "precision": "bf16_autocast"},
    {"name": "bf16_weights", "precision": "bf16_weights"},
    {"name": "bf16+ckpt", "precision": "bf16_autocast", "grad_checkpointing": True},
    {"name": "bf16+accum4", "precision": "bf16_autocast", "grad_accum": 4, "batch_size": 1},
    {"name": "bf16+packing", "precision": "bf16_autocast", "packing": True},
    {"name": "bf16+lora16", "precision": "bf16_autocast", "lora": 16},
    {"name": "qlora16", "precision": "bf16_autocast", "lora": 16, "qlora": True},
]


def run_once(cfg: TrainConfig, name: str, *, root: str = "runs", collect_env: bool = True) -> dict[str, Any]:
    """Train one configuration under a Run and return its summary."""
    model_id = resolve_model_id(cfg.model)
    tok = None if cfg.model == TINY_RANDOM else build_tokenizer(model_id)
    pad_id = 0 if tok is None else tok.pad_token_id
    samples = load_samples(cfg, tok)
    with Run(LAB, name, cfg.to_dict(), root=root, collect_env=collect_env,
             model={"id": model_id, "precision": cfg.precision, "lora": cfg.lora, "qlora": cfg.qlora},
             workload={"dataset": cfg.dataset, "n_samples": len(samples), "seq_len": cfg.seq_len,
                       "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum, "global_batch": cfg.global_batch,
                       "packing": cfg.packing, "seed": cfg.seed},
             execution={"steps": cfg.steps, "warmup_steps": cfg.warmup_steps}) as run:
        model, info = build_model(cfg)
        run.update(model=info)
        summary = train_loop(model, cfg, samples, pad_id, log=run.log_metric)
        summary.update({"name": name, "config": cfg.to_dict(), "model_info": info, "run_dir": str(run.dir)})
        run.save_summary(summary)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


# --------------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------------
def run_sweep(base: TrainConfig, grid: list[dict[str, Any]], out_dir: Path, root: str = "runs") -> list[dict[str, Any]]:
    results = []
    for item in grid:
        overrides = {k: v for k, v in item.items() if k != "name"}
        cfg = TrainConfig(**{**base.to_dict(), **overrides})
        try:
            s = run_once(cfg, item["name"], root=root)
        except Exception as e:  # keep going; a failed condition is a result too (e.g. OOM)
            s = {"name": item["name"], "config": cfg.to_dict(), "error": f"{type(e).__name__}: {e}"[:500]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(s)
        print(format_row(s), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(results, out_dir / "sweep_results.json")
    (out_dir / "sweep_table.md").write_text(format_table(results))
    plot_sweep(results, out_dir / "memory_vs_throughput.png")
    return results


def format_row(s: dict[str, Any]) -> str:
    if "error" in s:
        return f"{s['name']:>16}  FAILED  {s['error']}"
    return (f"{s['name']:>16}  {s['real_tokens_per_s']:9.0f} tok/s  peak_alloc {s['peak_allocated_gib']:5.2f} GiB  "
            f"reserved {s['peak_reserved_gib']:5.2f} GiB  step {s['mean_step_s']*1000:7.1f} ms  loss {s['mean_loss_last5']:.3f}")


def format_table(results: list[dict[str, Any]]) -> str:
    head = ("| condition | precision | ckpt | accum | packing | lora | trainable % | real tok/s | pad frac | "
            "peak alloc GiB | peak reserved GiB | est. static GiB | step ms | fwd/bwd/opt share | loss (last 5) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for s in results:
        c = s["config"]
        if "error" in s:
            rows.append(f"| {s['name']} | {c['precision']} | {c['grad_checkpointing']} | {c['grad_accum']} | {c['packing']} | "
                        f"{c['lora'] or '-'}{'(q)' if c['qlora'] else ''} | - | FAILED: {s['error'][:60]} | | | | | | | |")
            continue
        ph = s["phase_share"]
        share = "/".join(f"{ph.get(k, 0)*100:.0f}" for k in ("forward", "backward", "optimizer"))
        rows.append(
            f"| {s['name']} | {c['precision']} | {c['grad_checkpointing']} | {c['grad_accum']} | {c['packing']} | "
            f"{c['lora'] or '-'}{'(q)' if c['qlora'] else ''} | {s['model_info']['trainable_frac']*100:.2f} | "
            f"{s['real_tokens_per_s']:.0f} | {s['padding_fraction']:.2f} | {s['peak_allocated_gib']:.2f} | "
            f"{s['peak_reserved_gib']:.2f} | {s['model_info']['est_static_gib']:.2f} | {s['mean_step_s']*1000:.0f} | "
            f"{share} | {s['mean_loss_last5']:.3f} |"
        )
    return head + "\n".join(rows) + "\n"


def plot_sweep(results: list[dict[str, Any]], path: Path) -> None:
    ok = [s for s in results if "error" not in s]
    if not ok:
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, s in enumerate(ok):
        ax.scatter(s["peak_allocated_gib"], s["real_tokens_per_s"], color=PALETTE[i % len(PALETTE)], s=60, label=s["name"])
        ax.annotate(s["name"], (s["peak_allocated_gib"], s["real_tokens_per_s"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("peak allocated memory (GiB)")
    ax.set_ylabel("effective (non-padding) tokens / s")
    ax.set_title("Memory vs throughput, one switch at a time")
    save_fig(fig, path)


# --------------------------------------------------------------------------------------------
# Checkpoint / resume verification
# --------------------------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": {"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, path)


def load_checkpoint(model, optimizer, path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng"]["cpu"])
    if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    return ck["step"]


def resume_test(cfg: TrainConfig, ckpt_dir: Path, atol: float = 1e-5) -> dict[str, Any]:
    """Train `steps` continuously; train `steps//2`, checkpoint, rebuild, resume; compare losses.

    Data order is a function of the seed and the step index, so the resumed run must consume
    exactly the batches the continuous run consumed after the checkpoint: the batch iterator
    is advanced to the checkpoint step before resuming.
    """
    model_id = resolve_model_id(cfg.model)
    tok = None if cfg.model == TINY_RANDOM else build_tokenizer(model_id)
    pad_id = 0 if tok is None else tok.pad_token_id
    samples = load_samples(cfg, tok)
    half = cfg.steps // 2

    def fresh():
        torch.manual_seed(cfg.seed)
        m, _ = build_model(cfg)
        return m, make_optimizer(m, cfg.lr)

    # continuous
    m, opt = fresh()
    cont = train_loop(m, cfg, samples, pad_id, optimizer=opt)["losses"]
    # first half + checkpoint
    m, opt = fresh()
    it = iter_batches(samples, cfg, pad_id)
    half_cfg = TrainConfig(**{**cfg.to_dict(), "steps": half})
    first = train_loop(m, half_cfg, samples, pad_id, optimizer=opt, batch_iter=it)["losses"]
    ckpt = ckpt_dir / "ckpt.pt"
    save_checkpoint(m, opt, half, ckpt)
    del m, opt
    # resume in a fresh model/optimizer, skipping the consumed batches
    m, opt = fresh()
    start = load_checkpoint(m, opt, ckpt)
    it2 = iter_batches(samples, cfg, pad_id)
    for _ in range(start * cfg.grad_accum):
        next(it2)
    second = train_loop(m, cfg, samples, pad_id, optimizer=opt, start_step=start, batch_iter=it2)["losses"]
    resumed = first + second
    diffs = [abs(a - b) for a, b in zip(cont, resumed)]
    max_diff = max(diffs) if diffs else 0.0
    return {
        "steps": cfg.steps, "checkpoint_step": start, "max_abs_loss_diff": max_diff,
        "post_resume_max_abs_diff": max(diffs[start:]) if diffs[start:] else 0.0,
        "passed": max_diff <= atol, "atol": atol,
        "continuous_losses": cont, "resumed_losses": resumed,
        "checkpoint_bytes": ckpt.stat().st_size,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="small", help="registry key, HF id, or 'random-tiny'")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "smoltalk", "wikitext"])
    p.add_argument("--precision", default="bf16_autocast", choices=["fp32", "bf16_autocast", "fp16_autocast", "bf16_weights"])
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    p.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights + LoRA")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny settings, finishes in well under a minute")
    p.add_argument("--sweep", action="store_true", help="run the predefined grid")
    p.add_argument("--resume-test", action="store_true", help="checkpoint/resume verification")
    p.add_argument("--out", default="reports/L05_training_efficiency/results")
    p.add_argument("--runs-root", default="runs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = TrainConfig(model=a.model, dataset=a.dataset, precision=a.precision, grad_checkpointing=a.grad_checkpointing,
                      grad_accum=a.grad_accum, packing=a.packing, lora=a.lora, qlora=a.qlora, batch_size=a.batch_size,
                      seq_len=a.seq_len, steps=a.steps, warmup_steps=a.warmup_steps, lr=a.lr, seed=a.seed,
                      n_samples=a.n_samples, device=a.device)
    if a.smoke:
        cfg = TrainConfig(**{**cfg.to_dict(), "steps": 6, "warmup_steps": 2, "batch_size": 2, "seq_len": 128, "n_samples": 64})
    out = Path(a.out)
    if a.resume_test:
        tmp = Path(a.runs_root) / LAB / "_resume_ckpt"
        res = resume_test(cfg, tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        dump_json(res, out / f"resume_test_{cfg.model.replace('/', '_')}_{cfg.device}.json")
        print(f"resume test: passed={res['passed']} max|Δloss|={res['max_abs_loss_diff']:.3e} (atol {res['atol']}) "
              f"steps={res['steps']} ckpt@{res['checkpoint_step']}")
        return 0 if res["passed"] else 1
    if a.sweep:
        grid = SWEEP_GRID if not a.smoke else SWEEP_GRID[:2]
        run_sweep(cfg, grid, out, root=a.runs_root)
        return 0
    name = a.name or f"{cfg.precision}{'_ckpt' if cfg.grad_checkpointing else ''}{'_pack' if cfg.packing else ''}{'_lora' if cfg.lora else ''}"
    s = run_once(cfg, name, root=a.runs_root)
    print(format_row(s))
    print("run dir:", s["run_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
