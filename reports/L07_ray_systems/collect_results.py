"""Copy the latest run of each L07 experiment into reports/L07_ray_systems/results/ and print
Markdown tables for the README / REPORT. Usage: python reports/L07_ray_systems/collect_results.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "L07_ray_systems"
OUT = Path(__file__).resolve().parent / "results"
EXPERIMENTS = ["task_overhead", "object_store", "placement", "failure_injection", "ray_train", "actor_vs_task", "ray_data_batch_inference", "ray_serve_bench"]


def latest(name: str, must_contain: str | None = None) -> Path | None:
    cands = sorted(p for p in RUNS.glob(f"*_{name}_*") if (p / "summary.json").exists())
    if must_contain:
        cands = [p for p in cands if must_contain in json.dumps(json.load(open(p / "config.json")))]
    return cands[-1] if cands else None


def copy_run(name: str, run: Path, suffix: str = "") -> dict:
    dst = OUT / (name + suffix)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "manifest.json", "config.json"):
        shutil.copy(run / f, dst / f)
    for png in run.glob("*.png"):
        shutil.copy(png, dst / png.name)
    for txt in run.glob("stats_*.txt"):
        shutil.copy(txt, dst / txt.name)
    (dst / "RUN_DIR.txt").write_text(str(run.relative_to(ROOT)) + "\n")
    return json.load(open(run / "summary.json"))


def md_table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def main() -> None:
    out: list[str] = []
    for name in EXPERIMENTS:
        variants = [(None, "")]
        if name in ("ray_serve_bench",):
            variants = [("\"fake\": true", "_fake"), ("SmolLM2", "_tiny")]
        for contains, suffix in variants:
            run = latest(name, contains)
            if run is None:
                out.append(f"### {name}{suffix}\n\n_not run yet_\n")
                continue
            s = copy_run(name, run, suffix)
            rel = run.relative_to(ROOT)
            if name == "task_overhead":
                rows = [[r["granularity_s"], r["n_units"], r["backend"], r["wall_s"], r["speedup_vs_sequential"], r["overhead_per_task_ms"] if r["overhead_per_task_ms"] is not None else "-"] for r in s["rows"]]
                out.append(f"### Task granularity vs overhead (`{rel}`, {s['workers']} workers)\n\n" + md_table(["g (s)", "units", "backend", "wall (s)", "speed-up", "overhead/task (ms)"], rows) + f"\n\nBreak-even granularity (first speed-up ≥ 1): `{s['breakeven_granularity_s']}`\n")
            elif name == "object_store":
                a, sp, b = s["arg_passing"], s["spilling"], s["backpressure"]
                out.append(f"### Object store (`{rel}`)\n\n" + md_table(["experiment", "metric", "value"], [
                    ["arg passing", f"{a['size_mib']} MiB by value to {a['k_tasks']} tasks (s)", a["by_value_s"]],
                    ["arg passing", "ray.put once + refs (s, incl. put)", a["by_ref_total_s"]],
                    ["arg passing", "speed-up", a["speedup"]],
                    ["spilling", f"{sp['total_put_mib']} MiB into a {sp['store_mib']} MiB store: spilled (MiB)", sp["spilled_mib_after_put"]],
                    ["spilling", "put mean / max (ms)", f"{sp['put_s_mean'] * 1e3:.1f} / {sp['put_s_max'] * 1e3:.1f}"],
                    ["spilling", "get mean / max (ms)", f"{sp['get_s_mean'] * 1e3:.1f} / {sp['get_s_max'] * 1e3:.1f}"],
                    ["backpressure", f"{b['n']} x {b['obj_mib']} MiB unbounded: wall (s) / spilled (MiB)", f"{b['unbounded_s']:.2f} / {b['spilled_mib_unbounded']}"],
                    ["backpressure", f"bounded window={b['window']}: wall (s) / spilled (MiB)", f"{b['bounded_s']:.2f} / {b['spilled_mib_bounded']}"],
                ]) + "\n")
            elif name == "placement":
                fg = s["fractional_gpu"]
                acts = fg.get("actors", [])
                rows = [[i, a.get("ray_gpu_ids"), a.get("cuda_visible_devices"), f"{a.get('torch_allocated_mib', 0):.0f}", f"{a.get('torch_reserved_mib', 0):.0f}"] for i, a in enumerate(acts)]
                out.append(f"### Placement groups and fractional GPUs (`{rel}`)\n\nPlacement groups: " + ", ".join(f"{k} ready in {v['ready_s']:.2f}s on nodes {v['nodes']}" for k, v in s["placement_groups"].items()) + f"\n\nFractional GPU: {len(acts)} actors each requesting `num_gpus={json.load(open(run / 'config.json'))['gpu_frac']}` (logical total {fg.get('logical_gpu_requested_total')}); nvidia-smi used MiB before → after: {fg.get('nvidia_smi_used_mib_before')} → {fg.get('nvidia_smi_used_mib_after')}.\n\n" + md_table(["actor", "ray.get_gpu_ids()", "CUDA_VISIBLE_DEVICES", "torch allocated (MiB)", "torch reserved (MiB)"], rows) + "\n")
            elif name == "failure_injection":
                rows = [[r["policy"], r["n_outputs"], r["n_unique"], r["n_missing"], r["n_duplicates"], r["n_failed_refs"], r["n_crashes"], r["n_restarts"], ", ".join(f"{x:.2f}" for x in r["recovery_s"]) or "-", r["exactly_once"], r["wall_s"]] for r in s["rows"]]
                out.append(f"### Failure injection (`{rel}`)\n\n" + md_table(["policy", "outputs", "unique", "missing", "duplicates", "failed refs", "crashes", "restarts", "recovery (s)", "exactly-once", "wall (s)"], rows) + "\n")
            elif name == "ray_train":
                m = s["final_metrics_rank0"]
                out.append(f"### Ray Train (`{rel}`)\n\n{s['workers']} workers, gpu={s['use_gpu']}: final loss {m['loss']:.3f}, {m['tokens_per_s_per_worker']:.0f} tokens/s/worker, wall {s['wall_s']:.1f}s, checkpoint `{Path(s['checkpoint']).name if s['checkpoint'] else None}`; parameter digest on rank 0 `{m['param_digest']}`, identical across workers: {m.get('params_identical_across_workers')}.\n")
            elif name == "actor_vs_task":
                rows = [[r["mode"], r["n_inputs"], r["wall_s"], r["inputs_per_s"], r["time_to_first_result_s"], r["model_loads"], r["model_load_s_mean"], r["peak_gpu_mib_max"] or "-"] for r in s["rows"]]
                out.append(f"### Actor vs task (`{rel}`, device={s['device']})\n\n" + md_table(["mode", "inputs", "wall (s)", "inputs/s", "first result (s)", "model loads", "load (s, mean)", "peak GPU MiB"], rows) + "\n")
            elif name == "ray_data_batch_inference":
                rows = [[r["batch_size"], r["concurrency"], r["slow_preprocess_ms"], r["n_rows"], r["wall_s"], r["rows_per_s"]] for r in s["rows"]]
                out.append(f"### Ray Data batch inference (`{rel}`, device={s['device']})\n\n" + md_table(["batch size", "actors", "slow preprocess (ms)", "rows", "wall (s)", "rows/s"], rows) + "\n")
            elif name == "ray_serve_bench":
                rows = [[r["replicas"], r["batch_wait_ms"], r["concurrency"], r["rps"], r["e2e_p50"], r["e2e_p99"], r["mean_batch"], r["error_rate"]] for r in s["rows"]]
                out.append(f"### Ray Serve bench{suffix} (`{rel}`, device={s['device']}, fake={s['fake']})\n\n" + md_table(["replicas", "batch wait (ms)", "concurrency", "req/s", "p50 (s)", "p99 (s)", "mean batch", "error rate"], rows) + "\n")
    text = "\n".join(out)
    (OUT / "RESULTS.md").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
