"""L07.4 — Placement groups and fractional GPUs: Ray resources are logical, not physical.

* Placement groups with PACK vs SPREAD strategies (on one node both land on the same node; the
  rendered spec shows what a multi-node scheduler would receive).
* Several actors each request `num_gpus=0.25`. Ray co-locates them on one device and sets
  CUDA_VISIBLE_DEVICES, but nothing stops an actor from allocating more than a quarter of the
  memory: `nvidia-smi` sees the sum of what each process actually allocated.

Run:  python distributed/ray_core/placement.py --smoke
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
from ray.util.placement_group import placement_group, remove_placement_group  # noqa: E402
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # noqa: E402

from distributed.ray_core._common import cuda_available, env_summary, init_ray, nvidia_smi_used_mib  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402


def render_bundles(n_workers: int, cpus_per_worker: float, gpus_per_worker: float, strategy: str) -> dict[str, Any]:
    """Pure helper: the bundle list a trainer / inference pool would request (unit-tested)."""
    bundle = {"CPU": cpus_per_worker}
    if gpus_per_worker > 0:
        bundle["GPU"] = gpus_per_worker
    return {"bundles": [dict(bundle) for _ in range(n_workers)], "strategy": strategy, "total": {"CPU": n_workers * cpus_per_worker, "GPU": n_workers * gpus_per_worker}}


class ProbeImpl:
    def probe(self, alloc_mib: int) -> dict[str, Any]:
        info: dict[str, Any] = {
            "pid": os.getpid(),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "node": ray.get_runtime_context().get_node_id()[:8],
        }
        if cuda_available() and alloc_mib > 0:
            import torch

            self.keep = torch.empty(alloc_mib * 2**20, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            info["torch_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
            info["torch_reserved_mib"] = torch.cuda.memory_reserved() / 2**20
            info["device_total_mib"] = torch.cuda.get_device_properties(0).total_memory / 2**20
        return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actors", type=int, default=3)
    ap.add_argument("--gpu-frac", type=float, default=0.25)
    ap.add_argument("--alloc-mib", type=int, default=1024, help="memory each actor really allocates")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.actors, args.alloc_mib = 2, 256

    cfg = vars(args)
    with Run("L07_ray_systems", "placement", cfg) as run:
        gpus = 1 if cuda_available() else 0
        init_ray(num_cpus=8, num_gpus=gpus)
        run.update(software={"ray_env": env_summary()})
        summary: dict[str, Any] = {"placement_groups": {}, "fractional_gpu": {}}
        try:
            for strategy in ("PACK", "SPREAD"):
                spec = render_bundles(2, 1, 0, strategy)
                pg = placement_group(spec["bundles"], strategy=strategy)
                t0 = time.perf_counter()
                ray.get(pg.ready(), timeout=30)
                Probe = ray.remote(num_cpus=1)(ProbeImpl)
                actors = [Probe.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(2)]
                infos = ray.get([a.probe.remote(0) for a in actors])
                summary["placement_groups"][strategy] = {"spec": spec, "ready_s": time.perf_counter() - t0, "nodes": [i["node"] for i in infos]}
                print(f"[pg {strategy}] ready in {time.perf_counter() - t0:.2f}s, actors on nodes {[i['node'] for i in infos]}")
                for a in actors:
                    ray.kill(a)
                remove_placement_group(pg)

            used_before = nvidia_smi_used_mib()
            res = {"num_gpus": args.gpu_frac} if gpus else {}
            Probe = ray.remote(num_cpus=1, **res)(ProbeImpl)
            actors = [Probe.remote() for _ in range(args.actors)]
            infos = ray.get([a.probe.remote(args.alloc_mib) for a in actors])
            time.sleep(1.0)
            used_after = nvidia_smi_used_mib()
            logical_total = args.actors * args.gpu_frac
            summary["fractional_gpu"] = {
                "actors": infos,
                "logical_gpu_requested_total": logical_total,
                "nvidia_smi_used_mib_before": used_before,
                "nvidia_smi_used_mib_after": used_after,
                "actual_alloc_mib_sum": args.actors * args.alloc_mib if gpus else 0,
            }
            print(f"[fractional] {args.actors} actors x num_gpus={args.gpu_frac}: ray_gpu_ids={[i['ray_gpu_ids'] for i in infos]}")
            print(f"[fractional] nvidia-smi used MiB before={used_before} after={used_after} (each actor allocated {args.alloc_mib} MiB)")
            for a in actors:
                ray.kill(a)
        finally:
            ray.shutdown()
        run.save_summary(summary)
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
