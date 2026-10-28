"""L07.5 — Failure injection: lost results, duplicated results, and exactly-once output.

A list of items is processed by a pool of worker actors that write results into a sink actor.
Faults are injected deterministically:

* worker death   the worker writes its result, then calls `os._exit(1)` (crash *after* the side effect)
* transient error the first attempt of some items raises an exception

Three policies are compared:

| policy          | actor restarts | task retries | sink              | expected outcome                 |
|-----------------|----------------|--------------|-------------------|----------------------------------|
| naive           | 0              | 0            | append list       | results on the dead worker are lost |
| at_least_once   | unlimited      | unlimited    | append list       | nothing lost, crashed items duplicated |
| exactly_once    | unlimited      | unlimited    | keyed (idempotent)| nothing lost, nothing duplicated |

The driver verifies `set(outputs) == set(inputs)` and counts duplicates; the sink records crash and
restart timestamps so recovery time can be measured from state that survives the crash.

Run:  python distributed/ray_core/failure_injection.py --smoke
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
from ray.exceptions import RayActorError, RayTaskError  # noqa: E402

from distributed.ray_core._common import env_summary, init_ray  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402


class SinkImpl:
    """Durable state that outlives workers: results, attempt counters, crash/restart times."""

    def __init__(self, idempotent: bool) -> None:
        self.idempotent = idempotent
        self.results: list[tuple[int, int]] = []
        self.keyed: dict[int, int] = {}
        self.attempts: dict[int, int] = {}
        self.events: list[dict[str, Any]] = []

    def record_attempt(self, item: int) -> int:
        self.attempts[item] = self.attempts.get(item, 0) + 1
        return self.attempts[item]

    def has(self, item: int) -> bool:
        return item in self.keyed

    def write(self, item: int, value: int) -> None:
        if self.idempotent:
            self.keyed[item] = value  # overwriting with the same value is a no-op in effect
        else:
            self.results.append((item, value))

    def event(self, kind: str, worker: str) -> None:
        self.events.append({"kind": kind, "worker": worker, "t": time.time()})

    def outputs(self) -> list[tuple[int, int]]:
        return list(self.keyed.items()) if self.idempotent else list(self.results)

    def get_events(self) -> list[dict[str, Any]]:
        return list(self.events)


class WorkerImpl:
    def __init__(self, name: str, sink: Any, crash_items: list[int], flaky_items: list[int], work_s: float, check_before: bool) -> None:
        self.name = name
        self.sink = sink
        self.crash_items = set(crash_items)
        self.flaky_items = set(flaky_items)
        self.work_s = work_s
        self.check_before = check_before
        ray.get(sink.event.remote("start", name))

    def process(self, item: int) -> int:
        if self.check_before and ray.get(self.sink.has.remote(item)):
            return item * 2  # already done by a previous attempt: skip the side effect
        attempt = ray.get(self.sink.record_attempt.remote(item))
        time.sleep(self.work_s)
        if item in self.flaky_items and attempt == 1:
            raise RuntimeError(f"transient failure on item {item}")
        value = item * 2
        ray.get(self.sink.write.remote(item, value))  # side effect happens here ...
        if item in self.crash_items and attempt == 1:
            ray.get(self.sink.event.remote("crash", self.name))
            os._exit(1)  # ... and the worker dies before returning
        return value


POLICIES = {
    "naive": {"max_restarts": 0, "max_task_retries": 0, "retry_exceptions": False, "idempotent": False, "check_before": False},
    "at_least_once": {"max_restarts": -1, "max_task_retries": -1, "retry_exceptions": True, "idempotent": False, "check_before": False},
    "exactly_once": {"max_restarts": -1, "max_task_retries": -1, "retry_exceptions": True, "idempotent": True, "check_before": True},
}


def analyse(items: list[int], outputs: list[tuple[int, int]], lost_refs: int) -> dict[str, Any]:
    """Pure helper (unit-tested): compare the output multiset with the input set."""
    seen = [k for k, _ in outputs]
    uniq = set(seen)
    wrong = [k for k, v in outputs if v != k * 2]
    return {
        "n_items": len(items),
        "n_outputs": len(seen),
        "n_unique": len(uniq),
        "n_duplicates": len(seen) - len(uniq),
        "n_missing": len(set(items) - uniq),
        "n_wrong_value": len(wrong),
        "n_failed_refs": lost_refs,
        "exactly_once": len(seen) == len(items) and uniq == set(items) and not wrong,
    }


def run_policy(policy: str, items: list[int], n_workers: int, crash_items: list[int], flaky_items: list[int], work_s: float) -> dict[str, Any]:
    p = POLICIES[policy]
    Sink = ray.remote(max_restarts=0)(SinkImpl)
    sink = Sink.remote(p["idempotent"])
    Worker = ray.remote(max_restarts=p["max_restarts"], max_task_retries=p["max_task_retries"])(WorkerImpl)
    workers = [Worker.remote(f"w{i}", sink, crash_items, flaky_items, work_s, p["check_before"]) for i in range(n_workers)]
    ray.get([sink.get_events.remote()])
    t0 = time.perf_counter()
    refs = {}
    for i, item in enumerate(items):
        w = workers[i % n_workers]
        refs[w.process.options(retry_exceptions=p["retry_exceptions"]).remote(item)] = item
    failed = 0
    errors: dict[str, int] = {}
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        try:
            ray.get(done[0])
        except (RayActorError, RayTaskError) as e:
            failed += 1
            errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1
    wall = time.perf_counter() - t0
    outputs = ray.get(sink.outputs.remote())
    events = ray.get(sink.get_events.remote())
    recov = []
    crashes = [e for e in events if e["kind"] == "crash"]
    for c in crashes:
        later = [e["t"] for e in events if e["kind"] == "start" and e["worker"] == c["worker"] and e["t"] > c["t"]]
        if later:
            recov.append(min(later) - c["t"])
    for w in workers:
        try:
            ray.kill(w)
        except Exception:
            pass
    ray.kill(sink)
    res = analyse(items, outputs, failed)
    res.update({"policy": policy, "wall_s": wall, "n_crashes": len(crashes), "n_restarts": sum(1 for e in events if e["kind"] == "start") - n_workers, "recovery_s": recov, "driver_errors": errors})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--work-s", type=float, default=0.01)
    ap.add_argument("--crash-items", type=int, nargs="+", default=[7, 101])
    ap.add_argument("--flaky-items", type=int, nargs="+", default=[3, 50, 150])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_items, args.crash_items, args.flaky_items = 40, [5], [3, 20]

    items = list(range(args.n_items))
    cfg = vars(args)
    with Run("L07_ray_systems", "failure_injection", cfg, workload=cfg) as run:
        init_ray(num_cpus=max(4, args.workers + 2))
        run.update(software={"ray_env": env_summary()})
        rows = []
        try:
            for policy in args.policies:
                r = run_policy(policy, items, args.workers, args.crash_items, args.flaky_items, args.work_s)
                rows.append(r)
                run.log_metric(**{k: v for k, v in r.items() if not isinstance(v, (list, dict))})
                print(f"{policy:<14} outputs={r['n_outputs']:>4} unique={r['n_unique']:>4} missing={r['n_missing']:>3} dup={r['n_duplicates']:>2} failed_refs={r['n_failed_refs']:>3} crashes={r['n_crashes']} recovery_s={[round(x, 2) for x in r['recovery_s']]} exactly_once={r['exactly_once']}", flush=True)
        finally:
            ray.shutdown()
        run.log_records(rows)
        plot_bars([f"{r['policy']}\nmissing" for r in rows] + [f"{r['policy']}\ndup" for r in rows], [r["n_missing"] for r in rows] + [r["n_duplicates"] for r in rows], ylabel="items", title=f"{args.n_items} items, {len(args.crash_items)} crashes, {len(args.flaky_items)} transient errors", path=run.path("loss_and_duplication.png"))
        run.save_summary({"rows": rows})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
