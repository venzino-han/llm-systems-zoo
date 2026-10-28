"""L07.2 — Stateless tasks vs stateful actors for model inference.

Three ways to run N generation requests through a small HF model:

* task_per_call   each Ray task loads the model, generates, and exits -> pays model load every call
* actor_pool      K actors load the model once; `ActorPool` streams inputs one at a time
* actor_batched   K actors process inputs in batches of B                -> amortises per-call overhead

Metrics: time to first result (start-up), end-to-end throughput (inputs/s), model-load time,
and per-actor peak GPU memory (torch allocator) when CUDA is present.

Run:  python distributed/ray_core/actor_vs_task.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray  # noqa: E402
from ray.util import ActorPool  # noqa: E402

from distributed.ray_core._common import cuda_available, env_summary, init_ray  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402
from llm_systems_zoo.registry import MODELS  # noqa: E402


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


def generate(tok, model, prompts: list[str], max_new_tokens: int, device: str) -> list[str]:
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _device() -> str:
    return "cuda" if cuda_available() else "cpu"


def task_body(model_id: str, prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
    device = _device()
    t0 = time.perf_counter()
    tok, model = load_model(model_id, device)
    t_load = time.perf_counter() - t0
    outs = generate(tok, model, prompts, max_new_tokens, device)
    return {"outputs": outs, "load_s": t_load, "pid": os.getpid()}


class ModelActorImpl:
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        self.device = _device()
        t0 = time.perf_counter()
        self.tok, self.model = load_model(model_id, self.device)
        self.load_s = time.perf_counter() - t0
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> dict[str, Any]:
        outs = generate(self.tok, self.model, prompts, self.max_new_tokens, self.device)
        mem = None
        if self.device == "cuda":
            import torch

            mem = torch.cuda.max_memory_allocated() / 2**20
        return {"outputs": outs, "peak_mib": mem, "pid": os.getpid()}

    def load_time(self) -> float:
        return self.load_s


def run_mode(mode: str, prompts: list[str], model_id: str, k: int, batch: int, max_new_tokens: int, gpu_frac: float, cpus_per_worker: float = 2) -> dict[str, Any]:
    # Resources bound how many model copies can be alive at once: 1/gpu_frac on a GPU,
    # num_cpus/cpus_per_worker on CPU. Without this, task_per_call would start one model per task.
    res_opts: dict[str, Any] = {"num_cpus": cpus_per_worker}
    if cuda_available():
        res_opts["num_gpus"] = gpu_frac
    t0 = time.perf_counter()
    t_first = None
    n_done = 0
    loads: list[float] = []
    peaks: list[float] = []
    if mode == "task_per_call":
        task = ray.remote(**res_opts)(task_body)
        refs = [task.remote(model_id, [p], max_new_tokens) for p in prompts]
        pending = list(refs)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            r = ray.get(done[0])
            t_first = t_first or time.perf_counter() - t0
            loads.append(r["load_s"])
            n_done += len(r["outputs"])
    else:
        Actor = ray.remote(**res_opts)(ModelActorImpl)
        actors = [Actor.remote(model_id, max_new_tokens) for _ in range(k)]
        loads = ray.get([a.load_time.remote() for a in actors])
        pool = ActorPool(actors)
        if mode == "actor_pool":
            items = [[p] for p in prompts]
        else:
            items = [prompts[i : i + batch] for i in range(0, len(prompts), batch)]
        for r in pool.map_unordered(lambda a, v: a.generate.remote(v), items):
            t_first = t_first or time.perf_counter() - t0
            n_done += len(r["outputs"])
            if r["peak_mib"] is not None:
                peaks.append(r["peak_mib"])
        for a in actors:
            ray.kill(a)
    wall = time.perf_counter() - t0
    return {
        "mode": mode,
        "n_inputs": n_done,
        "wall_s": wall,
        "inputs_per_s": n_done / wall,
        "time_to_first_result_s": t_first,
        "model_load_s_mean": sum(loads) / len(loads) if loads else None,
        "model_loads": len(loads),
        "peak_gpu_mib_max": max(peaks) if peaks else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--actors", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--gpu-frac", type=float, default=0.25, help="logical GPU share per actor/task")
    ap.add_argument("--modes", nargs="+", default=["task_per_call", "actor_pool", "actor_batched"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model, args.n, args.max_new_tokens, args.batch = MODELS["tiny"].hf_id, 8, 8, 4

    prompts = [f"Question {i}: explain in one sentence why the sky is blue." for i in range(args.n)]
    cfg = vars(args)
    with Run("L07_ray_systems", "actor_vs_task", cfg, model={"id": args.model}, workload={"n": args.n, "max_new_tokens": args.max_new_tokens}) as run:
        gpus = 1 if cuda_available() else 0
        init_ray(num_cpus=8, num_gpus=gpus)
        run.update(software={"ray_env": env_summary()})
        rows = []
        try:
            for mode in args.modes:
                row = run_mode(mode, prompts, args.model, args.actors, args.batch, args.max_new_tokens, args.gpu_frac)
                rows.append(row)
                run.log_metric(**row)
                print(f"{mode:<15} wall={row['wall_s']:.2f}s thr={row['inputs_per_s']:.2f}/s first={row['time_to_first_result_s']:.2f}s loads={row['model_loads']}", flush=True)
        finally:
            ray.shutdown()
        run.log_records(rows)
        plot_bars([r["mode"] for r in rows], [r["inputs_per_s"] for r in rows], ylabel="inputs / s", title=f"{args.model} n={args.n}", path=run.path("throughput.png"))
        run.save_summary({"rows": rows, "device": _device()})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
