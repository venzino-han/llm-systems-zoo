"""L07.3 — The object store: by-value vs by-reference, spilling, and backpressure.

Experiments (each in its own Ray session so the object-store size can change):

1. argument passing   a 100 MB array passed by value to K tasks vs `ray.put` once and pass the ref
2. spilling           a small object store (e.g. 200 MB) is filled with 50 MB objects; put/get latency
                      and the MiB spilled to disk are recorded
3. backpressure       a producer submits many 10 MB results; unbounded submission vs `ray.wait`
                      with a bounded in-flight window

Run:  python distributed/ray_core/object_store.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray  # noqa: E402

from distributed.ray_core._common import env_summary, init_ray, spilled_mib  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402

MIB = 1024 * 1024


@ray.remote
def consume(arr: np.ndarray) -> float:
    return float(arr[::4096].sum())


@ray.remote
def produce(mib: int, seed: int) -> np.ndarray:
    return np.full(mib * MIB // 8, seed, dtype=np.float64)


def bounded_map(submit: Callable[[Any], ray.ObjectRef], items: Iterable[Any], max_in_flight: int, consume: Callable[[Any], Any] = lambda x: x) -> list[Any]:
    """Call `submit(item)` (returns an ObjectRef) keeping at most `max_in_flight` results un-consumed.

    This is the canonical backpressure pattern: results are fetched as soon as they are ready and
    reduced by `consume`, which must *not* keep the raw object alive. `ray.get` returns zero-copy
    views into the object store; holding them pins the memory, the store fills, and producers block.
    """
    results: list[Any] = []
    pending: list[ray.ObjectRef] = []
    for item in items:
        pending.append(submit(item))
        if len(pending) >= max_in_flight:
            done, pending = ray.wait(pending, num_returns=1)
            results.append(consume(ray.get(done[0])))
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        results.append(consume(ray.get(done[0])))
    return results


def exp_arg_passing(size_mib: int, k: int) -> dict[str, Any]:
    arr = np.ones(size_mib * MIB // 8, dtype=np.float64)
    t0 = time.perf_counter()
    ray.get([consume.remote(arr) for _ in range(k)])
    by_value = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = ray.put(arr)
    t_put = time.perf_counter() - t0
    t0 = time.perf_counter()
    ray.get([consume.remote(ref) for _ in range(k)])
    by_ref = time.perf_counter() - t0
    return {"size_mib": size_mib, "k_tasks": k, "by_value_s": by_value, "put_s": t_put, "by_ref_s": by_ref, "by_ref_total_s": by_ref + t_put, "speedup": by_value / (by_ref + t_put)}


def exp_spilling(store_mib: int, obj_mib: int, n_objs: int) -> dict[str, Any]:
    refs = []
    put_times = []
    for i in range(n_objs):
        arr = np.full(obj_mib * MIB // 8, i, dtype=np.float64)
        t0 = time.perf_counter()
        refs.append(ray.put(arr))
        put_times.append(time.perf_counter() - t0)
    spilled_after_put = spilled_mib()
    get_times = []
    for r in refs:  # oldest first: these are the ones most likely spilled
        t0 = time.perf_counter()
        ray.get(r)
        get_times.append(time.perf_counter() - t0)
    return {
        "store_mib": store_mib,
        "obj_mib": obj_mib,
        "n_objs": n_objs,
        "total_put_mib": obj_mib * n_objs,
        "put_s_mean": float(np.mean(put_times)),
        "put_s_max": float(np.max(put_times)),
        "get_s_mean": float(np.mean(get_times)),
        "get_s_max": float(np.max(get_times)),
        "spilled_mib_after_put": spilled_after_put,
        "spilled_mib_after_get": spilled_mib(),
    }


def exp_backpressure(n: int, obj_mib: int, window: int) -> dict[str, Any]:
    before = spilled_mib() or 0.0
    t0 = time.perf_counter()
    refs = [produce.remote(obj_mib, i) for i in range(n)]
    total = 0.0
    for r in refs:  # unbounded: every result lives in the store until the loop reaches it
        total += float(ray.get(r)[0])
        del r
    del refs
    unbounded = time.perf_counter() - t0
    spilled_unbounded = (spilled_mib() or 0.0) - before
    before = spilled_mib() or 0.0
    t0 = time.perf_counter()
    res = bounded_map(lambda i: produce.remote(obj_mib, i), range(n), max_in_flight=window, consume=lambda a: float(a[0]))
    total_b = sum(res)
    bounded = time.perf_counter() - t0
    spilled_bounded = (spilled_mib() or 0.0) - before
    assert abs(total - total_b) < 1e-6
    return {"n": n, "obj_mib": obj_mib, "window": window, "unbounded_s": unbounded, "bounded_s": bounded, "spilled_mib_unbounded": spilled_unbounded, "spilled_mib_bounded": spilled_bounded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arg-mib", type=int, default=100)
    ap.add_argument("--k-tasks", type=int, default=16)
    ap.add_argument("--store-mib", type=int, default=256)
    ap.add_argument("--obj-mib", type=int, default=50)
    ap.add_argument("--n-objs", type=int, default=12)
    ap.add_argument("--bp-n", type=int, default=120)
    ap.add_argument("--bp-obj-mib", type=int, default=10)
    ap.add_argument("--bp-window", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.arg_mib, args.k_tasks, args.n_objs, args.bp_n = 20, 4, 8, 30

    cfg = vars(args)
    with Run("L07_ray_systems", "object_store", cfg, workload=cfg) as run:
        run.update(software={"ray_env": env_summary()})
        init_ray(num_cpus=4)
        try:
            a = exp_arg_passing(args.arg_mib, args.k_tasks)
            print(f"[args] by_value={a['by_value_s']:.3f}s  put+by_ref={a['by_ref_total_s']:.3f}s  speedup={a['speedup']:.1f}x")
        finally:
            ray.shutdown()
        init_ray(num_cpus=4, object_store_memory=args.store_mib * MIB)
        try:
            s = exp_spilling(args.store_mib, args.obj_mib, args.n_objs)
            print(f"[spill] put {s['total_put_mib']} MiB into {s['store_mib']} MiB store: spilled={s['spilled_mib_after_put']} MiB get_max={s['get_s_max'] * 1e3:.1f} ms")
            b = exp_backpressure(args.bp_n, args.bp_obj_mib, args.bp_window)
            print(f"[backpressure] unbounded={b['unbounded_s']:.2f}s spilled={b['spilled_mib_unbounded']} MiB | bounded(window={b['window']})={b['bounded_s']:.2f}s spilled={b['spilled_mib_bounded']} MiB")
        finally:
            ray.shutdown()
        summary = {"arg_passing": a, "spilling": s, "backpressure": b}
        run.save_summary(summary)
        plot_bars(["by value", "ray.put + ref"], [a["by_value_s"], a["by_ref_total_s"]], ylabel="seconds", title=f"{args.arg_mib} MiB arg to {args.k_tasks} tasks", path=run.path("arg_passing.png"))
        plot_bars(["unbounded", f"bounded (window={args.bp_window})"], [b["unbounded_s"], b["bounded_s"]], ylabel="seconds", title=f"{args.bp_n} x {args.bp_obj_mib} MiB results", path=run.path("backpressure.png"))
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
