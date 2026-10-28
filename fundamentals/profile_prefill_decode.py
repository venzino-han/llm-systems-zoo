"""L01 — where does the time go? Prefill vs decode timing and kernel-level profiles.

Prefill processes the whole prompt in one forward pass (large GEMMs, compute bound);
decode produces one token per forward pass (weights re-read every step, memory bound).
This script times both phases with CUDA-synchronised `PhaseTimer`s, compares the decode step
time against the roofline bound from `flops_calculator.py`, and records a torch.profiler
trace for each phase with the top CUDA kernels by time.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python fundamentals/profile_prefill_decode.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from fundamentals.flops_calculator import KNOWN_GPUS, GPUSpec, decode_roofline, prefill_roofline
from fundamentals.measure_memory import load_model, wait_for_free_memory
from fundamentals.memory_calculator import ModelConfig
from llm_systems_zoo import PhaseTimer, Run

LAB = "L01_memory_anatomy"


def match_gpu_spec(name: str) -> GPUSpec | None:
    for key, spec in KNOWN_GPUS.items():
        if key.split()[-1].lower() in name.lower() and key.split()[0].lower() in name.lower():
            return spec
    return None


@torch.no_grad()
def run_phases(model, ids: torch.Tensor, decode_steps: int, timer: PhaseTimer) -> None:
    with timer.phase("prefill"):
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    with timer.phase("decode"):
        for _ in range(decode_steps):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)


def top_kernels(prof, n: int = 15) -> list[dict[str, Any]]:
    rows = []
    for ev in sorted(prof.key_averages(), key=lambda e: getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)), reverse=True):
        t = getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0))
        if t <= 0:
            continue
        rows.append({"name": ev.key[:90], "device_time_ms": t / 1e3, "calls": ev.count,
                     "cpu_time_ms": ev.cpu_time_total / 1e3})
        if len(rows) >= n:
            break
    return rows


@torch.no_grad()
def profile_phase(model, ids, decode_steps: int, phase: str, trace_path) -> list[dict[str, Any]]:
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if phase == "prefill":
            model(input_ids=ids, use_cache=True, logits_to_keep=1)
        else:
            for _ in range(decode_steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    return top_kernels(prof)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Prefill vs decode timing + profiler traces.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--need-free-gib", type=float, default=3.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.prompt_len, a.decode_steps, a.repeats = 256, 16, 2
    wait_for_free_memory(a.need_free_gib)
    cfg = ModelConfig.from_pretrained(a.model)
    gpu_name = torch.cuda.get_device_name(0)
    spec = match_gpu_spec(gpu_name)
    with Run(LAB, f"profile_b{a.batch}_s{a.prompt_len}", vars(a), model={"id": a.model, "dtype": "bf16", "attn": a.attn},
             workload={"batch": a.batch, "prompt_len": a.prompt_len, "decode_steps": a.decode_steps, "synthetic_random_tokens": True},
             execution={"warmup": 1, "repeats": a.repeats, "label": "pilot, shared GPU"}) as run:
        model = load_model(a.model, torch.bfloat16, a.attn)
        ids = torch.randint(0, cfg.vocab_size, (a.batch, a.prompt_len), device="cuda")
        run_phases(model, ids, a.decode_steps, PhaseTimer(sync_cuda=True))  # warm-up
        timer = PhaseTimer(sync_cuda=True)
        for _ in range(a.repeats):
            run_phases(model, ids, a.decode_steps, timer)
        s = timer.summary()
        prefill_s = s["prefill"]["mean_s"]
        decode_step_s = s["decode"]["mean_s"] / a.decode_steps
        result: dict[str, Any] = {
            "label": "pilot, shared GPU",
            "gpu": gpu_name,
            "prefill": {"mean_s": prefill_s, "std_s": float(torch.tensor(timer.samples["prefill"]).std()) if a.repeats > 1 else 0.0,
                        "tokens_per_s": a.batch * a.prompt_len / prefill_s},
            "decode": {"ms_per_step": decode_step_s * 1e3, "tokens_per_s": a.batch / decode_step_s,
                       "share_of_total": s["decode"]["total_s"] / (s["decode"]["total_s"] + s["prefill"]["total_s"])},
        }
        if spec:
            pre = prefill_roofline(cfg, a.batch, a.prompt_len, spec)
            dec = decode_roofline(cfg, a.batch, a.prompt_len + a.decode_steps // 2, spec)
            result["roofline"] = {
                "gpu_spec": spec.__dict__,
                "prefill_bound_s": pre["t_roofline_s"], "prefill_bound": pre["bound"],
                "prefill_fraction_of_roofline": pre["t_roofline_s"] / prefill_s,
                "decode_bound_s_per_step": dec["t_roofline_s"], "decode_bound": dec["bound"],
                "decode_fraction_of_roofline": dec["t_roofline_s"] / decode_step_s,
                "decode_effective_bandwidth_gb_s": (cfg.n_params * 2 + a.batch * (a.prompt_len + a.decode_steps // 2) * cfg.kv_bytes_per_token()) / decode_step_s / 1e9,
            }
        result["top_kernels_prefill"] = profile_phase(model, ids, a.decode_steps, "prefill", run.path("trace_prefill.json"))
        result["top_kernels_decode"] = profile_phase(model, ids, a.decode_steps, "decode", run.path("trace_decode.json"))
        run.save_summary(result)
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("top_kernels")}, indent=2))
        for phase in ("prefill", "decode"):
            print(f"\ntop kernels — {phase}")
            for r in result[f"top_kernels_{phase}"][:8]:
                print(f"  {r['device_time_ms']:8.2f} ms  {r['calls']:5d}×  {r['name']}")
        print(f"run dir: {run.dir}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
