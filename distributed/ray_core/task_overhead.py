"""L07.1 — When does distributing work pay off? Task granularity vs scheduling overhead.

Compares four ways of running N independent CPU-bound units of work of duration `g`:

* sequential                 one process, a for loop (the baseline)
* multiprocessing.Pool       local processes, pickled arguments, no scheduler
* ray tasks (1 item/task)    each unit is a Ray task  -> pays scheduling + object store cost per task
* ray tasks (chunked)        `chunk` units per task   -> amortises the per-task cost

For each granularity we report wall time, speed-up over sequential and the *overhead per task*:
    overhead_per_task = (wall - ideal) / n_tasks,  ideal = n_units * g / workers
The curve where speed-up crosses 1 tells you the minimum task size worth distributing.

Run:  python distributed/ray_core/task_overhead.py --smoke
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray  # noqa: E402

from distributed.ray_core._common import busy_wait, env_summary, init_ray  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_lines  # noqa: E402


def _work(g: float) -> int:
    return busy_wait(g)


def _work_chunk(gs: list[float]) -> list[int]:
    return [busy_wait(g) for g in gs]


@ray.remote
def ray_work(g: float) -> int:
    return busy_wait(g)


@ray.remote
def ray_work_chunk(gs: list[float]) -> list[int]:
    return [busy_wait(g) for g in gs]


def run_sequential(g: float, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        _work(g)
    return time.perf_counter() - t0


def run_multiprocessing(g: float, n: int, workers: int, pool: mp.pool.Pool) -> float:
    t0 = time.perf_counter()
    pool.map(_work, [g] * n, chunksize=1)
    return time.perf_counter() - t0


def run_ray_tasks(g: float, n: int) -> float:
    t0 = time.perf_counter()
    ray.get([ray_work.remote(g) for _ in range(n)])
    return time.perf_counter() - t0


def run_ray_chunked(g: float, n: int, chunk: int) -> float:
    chunks = [[g] * chunk for _ in range(n // chunk)]
    if n % chunk:
        chunks.append([g] * (n % chunk))
    t0 = time.perf_counter()
    ray.get([ray_work_chunk.remote(c) for c in chunks])
    return time.perf_counter() - t0


def n_units_for(g: float, budget_s: float, max_units: int) -> int:
    """Pick N so that sequential time is ~budget_s, bounded to keep tiny tasks tractable."""
    return int(max(8, min(max_units, budget_s / g)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--granularities", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--budget-s", type=float, default=4.0, help="target sequential seconds per granularity")
    ap.add_argument("--max-units", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.granularities = [1e-4, 1e-2, 2e-1]
        args.budget_s, args.max_units, args.repeats = 1.0, 400, 1

    cfg = vars(args)
    rows = []
    with Run("L07_ray_systems", "task_overhead", cfg, workload={"work": "busy-wait", **cfg}) as run:
        run.update(software={"ray_env": env_summary()})
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers)
        try:
            pool.map(_work, [1e-4] * args.workers)  # warm the pool
            init_ray(num_cpus=args.workers)
            ray.get([ray_work.remote(1e-4) for _ in range(args.workers)])  # warm Ray workers
            for g in args.granularities:
                n = n_units_for(g, args.budget_s, args.max_units)
                chunk = min(args.chunk, max(1, n // args.workers))
                per_backend: dict[str, list[float]] = {"sequential": [], "multiprocessing": [], "ray_tasks": [], "ray_chunked": []}
                for _ in range(args.repeats):
                    per_backend["sequential"].append(run_sequential(g, n))
                    per_backend["multiprocessing"].append(run_multiprocessing(g, n, args.workers, pool))
                    per_backend["ray_tasks"].append(run_ray_tasks(g, n))
                    per_backend["ray_chunked"].append(run_ray_chunked(g, n, chunk))
                seq = min(per_backend["sequential"])
                ideal = n * g / args.workers
                for backend, times in per_backend.items():
                    wall = min(times)
                    n_tasks = n if backend != "ray_chunked" else -(-n // chunk)
                    row = {
                        "granularity_s": g,
                        "n_units": n,
                        "backend": backend,
                        "chunk": chunk if backend == "ray_chunked" else 1,
                        "n_tasks": n_tasks if backend != "sequential" else 0,
                        "wall_s": wall,
                        "speedup_vs_sequential": seq / wall,
                        "ideal_parallel_s": ideal,
                        "overhead_per_task_ms": None if backend == "sequential" else max(wall - ideal, 0) / n_tasks * 1e3,
                    }
                    rows.append(row)
                    run.log_metric(**row)
                    print(f"g={g:>8.5f}s n={n:>5} {backend:<16} wall={wall:8.3f}s speedup={seq / wall:6.2f}x", flush=True)
        finally:
            pool.close()
            pool.join()
            ray.shutdown()

        run.log_records(rows)
        gs = sorted({r["granularity_s"] for r in rows})
        backends = ["multiprocessing", "ray_tasks", "ray_chunked"]
        speed = {b: [next(r["speedup_vs_sequential"] for r in rows if r["backend"] == b and r["granularity_s"] == g) for g in gs] for b in backends}
        over = {b: [next(r["overhead_per_task_ms"] for r in rows if r["backend"] == b and r["granularity_s"] == g) for g in gs] for b in backends}
        plot_lines(gs, speed, xlabel="task granularity (s, log)", ylabel="speed-up vs sequential", title=f"{args.workers} workers", path=run.path("speedup_vs_granularity.png"), logx=True)
        plot_lines(gs, over, xlabel="task granularity (s, log)", ylabel="overhead per task (ms)", title="scheduling + transfer overhead", path=run.path("overhead_vs_granularity.png"), logx=True, logy=True)
        breakeven = {b: next((g for g, s in zip(gs, speed[b]) if s >= 1.0), None) for b in backends}
        run.save_summary({"rows": rows, "breakeven_granularity_s": breakeven, "workers": args.workers})
        print("break-even granularity (first g with speed-up >= 1):", breakeven)
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
