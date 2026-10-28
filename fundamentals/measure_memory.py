"""L01 — measure GPU memory of inference (prefill / decode) and a training step, and compare
with the analytic estimate from `memory_calculator.py`.

Three observers, three different numbers:

    torch.cuda.memory_allocated        bytes in live tensors (what the estimate models)
    torch.cuda.memory_reserved         what the caching allocator holds from the driver
                                       (allocated + cached free segments + fragmentation)
    nvidia-smi per-process used        reserved + CUDA context + cuBLAS/attention workspaces
                                       + kernels; what an operator sees on a shared GPU

Peak values are read with `max_memory_allocated` after `reset_peak_memory_stats()` so each
condition is measured independently of what happened before it.

Usage (repo root):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python fundamentals/measure_memory.py --smoke
    ... --model Qwen/Qwen2.5-0.5B-Instruct --prompt-lens 256 1024 --batch-sizes 1 4
    ... --train-model HuggingFaceTB/SmolLM2-135M-Instruct --train-grid 1x256 2x512
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time
from typing import Any

import torch

from fundamentals.memory_calculator import GIB, ModelConfig, estimate_inference, estimate_training
from llm_systems_zoo import Run
from llm_systems_zoo.plotting import PALETTE, plt, save_fig, setup_style

LAB = "L01_memory_anatomy"


def nvidia_smi_process_mib() -> float | None:
    """GPU memory the driver attributes to this process (includes context + workspaces)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return None
    for line in out.splitlines():
        pid, used = [x.strip() for x in line.split(",")]
        if int(pid) == os.getpid():
            return float(used)
    return None


def snapshot(label: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "process_used_nvidia_smi_gib": (nvidia_smi_process_mib() or 0) / 1024,
        "device_free_gib": free / GIB,
        "device_total_gib": total / GIB,
    }


def wait_for_free_memory(need_gib: float, timeout_s: float = 300) -> None:
    t0 = time.time()
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free / GIB >= need_gib:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"only {free / GIB:.1f} GiB free, need {need_gib} GiB (shared GPU)")
        print(f"[wait] {free / GIB:.1f} GiB free < {need_gib} GiB needed; retrying in 15 s")
        time.sleep(15)


def load_model(model_id: str, dtype: torch.dtype, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
    return model.to("cuda").eval()


# ---- inference ---------------------------------------------------------------------------------
@torch.no_grad()
def measure_inference(model, cfg: ModelConfig, prompt_lens: list[int], batch_sizes: list[int], decode_steps: int,
                      weight_dtype: str, run: Run) -> list[dict[str, Any]]:
    rows = []
    vocab = cfg.vocab_size
    gc.collect()
    torch.cuda.empty_cache()
    base = snapshot("weights_loaded")
    for b in batch_sizes:
        for s in prompt_lens:
            ids = torch.randint(0, vocab, (b, s), device="cuda")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0
            prefill = snapshot(f"prefill_b{b}_s{s}")
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(decode_steps):
                out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_ids = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            t_decode = time.perf_counter() - t0
            decode = snapshot(f"decode_b{b}_s{s}+{decode_steps}")
            est_pre = estimate_inference(cfg, b, s, 0, weight_dtype=weight_dtype)
            est_dec = estimate_inference(cfg, b, s, decode_steps, weight_dtype=weight_dtype)
            row = {
                "phase_grid": {"batch": b, "prompt_len": s, "decode_steps": decode_steps},
                "weights_allocated_gib": base["allocated_gib"],
                "prefill": {
                    "estimate_total_gib": est_pre.total / GIB,
                    "estimate_components_gib": est_pre.as_gib(),
                    "measured_peak_allocated_gib": prefill["max_allocated_gib"],
                    "measured_peak_reserved_gib": prefill["max_reserved_gib"],
                    "process_used_nvidia_smi_gib": prefill["process_used_nvidia_smi_gib"],
                    "wall_s": t_prefill,
                    "tokens_per_s": b * s / t_prefill,
                },
                "decode": {
                    "estimate_total_gib": est_dec.total / GIB,
                    "estimate_components_gib": est_dec.as_gib(),
                    "measured_peak_allocated_gib": decode["max_allocated_gib"],
                    "measured_peak_reserved_gib": decode["max_reserved_gib"],
                    "process_used_nvidia_smi_gib": decode["process_used_nvidia_smi_gib"],
                    "wall_s": t_decode,
                    "ms_per_step": 1e3 * t_decode / decode_steps,
                    "tokens_per_s": b * decode_steps / t_decode,
                    "kv_tokens": b * (s + decode_steps),
                    "kv_measured_delta_gib": decode["allocated_gib"] - base["allocated_gib"],
                    "kv_estimate_gib": est_dec.components["kv_cache"] / GIB,
                },
            }
            rows.append(row)
            run.log_metric(kind="inference", batch=b, prompt_len=s,
                           prefill_peak=prefill["max_allocated_gib"], decode_peak=decode["max_allocated_gib"])
            print(f"[inference] b={b:<3} s={s:<5} prefill est {est_pre.total / GIB:.3f} / meas {prefill['max_allocated_gib']:.3f} GiB | "
                  f"decode est {est_dec.total / GIB:.3f} / meas {decode['max_allocated_gib']:.3f} GiB | "
                  f"KV est {row['decode']['kv_estimate_gib'] * 1024:.1f} MiB meas {row['decode']['kv_measured_delta_gib'] * 1024:.1f} MiB | "
                  f"{row['decode']['ms_per_step']:.1f} ms/step")
            del out, past, next_ids, ids
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# ---- training ----------------------------------------------------------------------------------
def measure_training(model_id: str, cfg: ModelConfig, grid: list[tuple[int, int]], modes: list[str], lora_rank: int | None,
                     run: Run, optim_foreach: bool | None = None) -> list[dict[str, Any]]:
    rows = []
    for mode in modes:
        amp = "amp" in mode
        ckpt = "ckpt" in mode
        for b, s in grid:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            param_dtype = "bf16" if lora_rank else "fp32"
            model = load_model(model_id, torch.bfloat16 if lora_rank else torch.float32)
            model.train()
            model.config.use_cache = False
            if lora_rank:
                from peft import LoraConfig, get_peft_model

                model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"))
            if ckpt:
                model.gradient_checkpointing_enable()
                if lora_rank:
                    model.enable_input_require_grads()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=1e-5, foreach=optim_foreach)
            ids = torch.randint(0, cfg.vocab_size, (b, s), device="cuda")
            result: dict[str, Any] = {"mode": mode, "batch": b, "seq_len": s, "lora_rank": lora_rank, "param_dtype": param_dtype,
                                      "optim_foreach": optim_foreach}
            try:
                for step in range(2):  # step 0 allocates optimizer state; step 1 is steady state
                    if step == 1:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        loss = model(input_ids=ids, labels=ids).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                snap = snapshot(f"train_{mode}_b{b}_s{s}")
                est = estimate_training(cfg, b, s, param_dtype=param_dtype, optimizer="adamw", mixed_precision=amp,
                                        gradient_checkpointing=ckpt, flash_attention=True, lora_rank=lora_rank)
                est_naive = estimate_training(cfg, b, s, param_dtype=param_dtype, optimizer="adamw", mixed_precision=amp,
                                              gradient_checkpointing=ckpt, flash_attention=False, lora_rank=lora_rank)
                result.update({
                    "estimate_total_gib": est.total / GIB,
                    "estimate_total_gib_scores_materialised": est_naive.total / GIB,
                    "estimate_components_gib": est.as_gib(),
                    "measured_peak_allocated_gib": snap["max_allocated_gib"],
                    "measured_peak_reserved_gib": snap["max_reserved_gib"],
                    "process_used_nvidia_smi_gib": snap["process_used_nvidia_smi_gib"],
                    "step_wall_s": dt,
                    "tokens_per_s": b * s / dt,
                    "trainable_params": sum(p.numel() for p in params),
                    "status": "ok",
                })
                print(f"[train] {mode:<9} b={b} s={s}: est {est.total / GIB:.3f} GiB (scores materialised {est_naive.total / GIB:.3f}) / "
                      f"meas peak alloc {snap['max_allocated_gib']:.3f} reserved {snap['max_reserved_gib']:.3f} GiB | {dt * 1e3:.0f} ms/step")
            except torch.cuda.OutOfMemoryError as e:
                result.update({"status": "oom", "error": str(e)[:200]})
                print(f"[train] {mode} b={b} s={s}: OOM")
            rows.append(result)
            run.log_metric(kind="training", mode=mode, batch=b, seq_len=s, status=result["status"],
                           peak=result.get("measured_peak_allocated_gib"))
            del model, opt, params, ids
            gc.collect()
            torch.cuda.empty_cache()
    return rows


# ---- figure ------------------------------------------------------------------------------------
def plot_estimate_vs_measured(inference_rows: list[dict], training_rows: list[dict], path) -> None:
    setup_style()
    labels, est, meas_alloc, meas_res = [], [], [], []
    for r in inference_rows:
        g = r["phase_grid"]
        for phase in ("prefill", "decode"):
            labels.append(f"{phase}\nb{g['batch']} s{g['prompt_len']}")
            est.append(r[phase]["estimate_total_gib"])
            meas_alloc.append(r[phase]["measured_peak_allocated_gib"])
            meas_res.append(r[phase]["measured_peak_reserved_gib"])
    for r in training_rows:
        if r.get("status") != "ok":
            continue
        labels.append(f"train {r['mode']}\nb{r['batch']} s{r['seq_len']}")
        est.append(r["estimate_total_gib"])
        meas_alloc.append(r["measured_peak_allocated_gib"])
        meas_res.append(r["measured_peak_reserved_gib"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(labels)), 4.2))
    x = range(len(labels))
    w = 0.27
    ax.bar([i - w for i in x], est, w, label="estimate (calculator)", color=PALETTE[0])
    ax.bar(list(x), meas_alloc, w, label="measured peak allocated", color=PALETTE[1])
    ax.bar([i + w for i in x], meas_res, w, label="measured peak reserved", color=PALETTE[2])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("GiB")
    ax.set_title("Estimate vs measured GPU memory")
    ax.legend(fontsize=8)
    save_fig(fig, path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Measure GPU memory for inference and a training step.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--train-model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                    help="model for the full fine-tuning step (small so it fits a shared GPU)")
    ap.add_argument("--train-grid", nargs="+", default=["1x256", "2x256", "1x512", "2x512"], help="batchxseq")
    ap.add_argument("--train-modes", nargs="+", default=["fp32", "amp", "amp+ckpt"])
    ap.add_argument("--lora-rank", type=int, default=None, help="measure a (Q)LoRA step on --train-model instead of full FT")
    ap.add_argument("--optim-foreach", choices=["default", "true", "false"], default="default",
                    help="AdamW foreach implementation (its temporaries show up as un-estimated peak memory)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--need-free-gib", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="estimate_vs_measured")
    a = ap.parse_args(argv)
    if a.smoke:
        a.prompt_lens, a.batch_sizes, a.decode_steps, a.train_grid, a.train_modes = [128, 256], [1, 2], 8, ["1x128", "2x128"], ["fp32", "amp+ckpt"]

    wait_for_free_memory(a.need_free_gib)
    cfg = ModelConfig.from_pretrained(a.model)
    train_cfg = ModelConfig.from_pretrained(a.train_model)
    config = vars(a).copy()
    with Run(LAB, a.name, config, model={"inference_model": a.model, "train_model": a.train_model, "dtype": a.dtype},
             workload={"prompt_lens": a.prompt_lens, "batch_sizes": a.batch_sizes, "decode_steps": a.decode_steps,
                       "train_grid": a.train_grid, "synthetic_random_tokens": True},
             execution={"shared_gpu": True, "label": "pilot, shared GPU", "steps_per_training_condition": 2}) as run:
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
        inf_rows: list[dict] = []
        train_rows: list[dict] = []
        if not a.skip_inference:
            model = load_model(a.model, torch_dtype)
            inf_rows = measure_inference(model, cfg, a.prompt_lens, a.batch_sizes, a.decode_steps, a.dtype, run)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if not a.skip_train:
            grid = [tuple(int(x) for x in g.split("x")) for g in a.train_grid]
            foreach = {"default": None, "true": True, "false": False}[a.optim_foreach]
            train_rows = measure_training(a.train_model, train_cfg, grid, a.train_modes, a.lora_rank, run, foreach)
        run.save_summary({
            "label": "pilot, shared GPU",
            "gpu": torch.cuda.get_device_name(0),
            "inference_model": {"id": a.model, "n_params": cfg.n_params, "kv_kib_per_token": cfg.kv_bytes_per_token(a.dtype) / 1024},
            "train_model": {"id": a.train_model, "n_params": train_cfg.n_params},
            "inference": inf_rows,
            "training": train_rows,
        })
        plot_estimate_vs_measured(inf_rows, train_rows, run.path("estimate_vs_measured.png"))
        print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
