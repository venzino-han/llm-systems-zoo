"""L07.8b — Closed-loop benchmark of the Serve deployment: 1 vs 2 replicas, with and without
Serve-level dynamic batching. Starts the deployment in-process, drives it with an async client at
several concurrency levels, and records per-request latency with the shared metrics module.

Run:  python distributed/ray_serve/bench.py --fake --smoke
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402
import ray  # noqa: E402
from ray import serve  # noqa: E402

from distributed.ray_core._common import cuda_available, env_summary, init_ray  # noqa: E402
from distributed.ray_serve.deployment import build_app, start  # noqa: E402
from llm_systems_zoo import RequestRecord, Run  # noqa: E402
from llm_systems_zoo.metrics import full_report  # noqa: E402
from llm_systems_zoo.plotting import plot_lines  # noqa: E402
from llm_systems_zoo.registry import MODELS  # noqa: E402


async def closed_loop(url: str, prompts: list[str], concurrency: int, max_tokens: int) -> list[RequestRecord]:
    """Each of `concurrency` clients sends its next request as soon as the previous one returns."""
    records: list[RequestRecord] = []
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))

    async with httpx.AsyncClient(timeout=300) as client:

        async def worker() -> None:
            while True:
                try:
                    i, p = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = RequestRecord(request_id=str(i), t_submit=time.perf_counter(), prompt_tokens=len(p.split()))
                try:
                    r = await client.post(url, json={"prompt": p, "max_tokens": max_tokens})
                    r.raise_for_status()
                    j = r.json()
                    rec.t_end = time.perf_counter()
                    rec.t_first_token = rec.t_end  # non-streaming endpoint: TTFT == E2E
                    rec.output_tokens = int(j.get("output_tokens", 0))
                    rec.extra = {"batch": j.get("batch"), "replica_s": j.get("replica_s")}
                except Exception as e:  # noqa: BLE001
                    rec.t_end = time.perf_counter()
                    rec.error = type(e).__name__
                records.append(rec)

        await asyncio.gather(*[worker() for _ in range(concurrency)])
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-ms", type=float, default=50)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--batch-wait-ms", type=float, nargs="+", default=[0, 10])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n-requests", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-ongoing", type=int, default=16)
    ap.add_argument("--gpu-frac", type=float, default=0.2)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.replicas, args.batch_wait_ms, args.concurrency, args.n_requests = [1, 2], [0], [1, 4], 24
        if not args.fake and not args.proxy:
            args.model, args.max_new_tokens = MODELS["tiny"].hf_id, 8

    prompts = [f"Request {i}: describe a distributed system in one sentence." for i in range(args.n_requests)]
    cfg = vars(args)
    with Run("L07_ray_systems", "ray_serve_bench", cfg, model={"id": None if args.fake else args.model}, workload={"n_requests": args.n_requests, "closed_loop": True}) as run:
        init_ray(num_cpus=8, num_gpus=1 if cuda_available() else 0)
        run.update(software={"ray_env": env_summary()})
        rows: list[dict[str, Any]] = []
        try:
            for bw in args.batch_wait_ms:
                for n_rep in args.replicas:
                    app = build_app(None if args.fake else args.model, n_rep, args.max_ongoing, args.max_new_tokens, args.fake_ms if args.fake else 0, args.proxy, bw, args.max_batch if bw > 0 else 1, args.gpu_frac)
                    url = start(app, args.port)
                    asyncio.run(closed_loop(url, prompts[:4], 2, args.max_new_tokens))  # warm-up
                    for conc in args.concurrency:
                        recs = asyncio.run(closed_loop(url, prompts, conc, args.max_new_tokens))
                        rep = full_report(recs)
                        row = {"replicas": n_rep, "batch_wait_ms": bw, "concurrency": conc, "rps": rep["throughput"]["requests_per_s"], "e2e_p50": rep["latency"]["e2e_s"].get("p50"), "e2e_p99": rep["latency"]["e2e_s"].get("p99"), "error_rate": rep["latency"]["error_rate"], "mean_batch": sum((r.extra.get("batch") or 1) for r in recs) / len(recs)}
                        rows.append(row)
                        run.log_metric(**row)
                        run.log_records(recs, filename=f"records_r{n_rep}_bw{int(bw)}_c{conc}.jsonl")
                        print(f"replicas={n_rep} batch_wait={bw:>4}ms conc={conc:>3} rps={row['rps']:.2f} p50={row['e2e_p50']:.3f}s p99={row['e2e_p99']:.3f}s mean_batch={row['mean_batch']:.1f}", flush=True)
                    serve.shutdown()
        finally:
            serve.shutdown()
            ray.shutdown()
        run.log_records(rows, filename="rows.jsonl")
        series = {f"r={r['replicas']} bw={r['batch_wait_ms']}": [x["rps"] for x in rows if x["replicas"] == r["replicas"] and x["batch_wait_ms"] == r["batch_wait_ms"]] for r in rows}
        plot_lines(args.concurrency, series, xlabel="closed-loop concurrency", ylabel="requests / s", title="Ray Serve: replicas x Serve-level batching", path=run.path("rps_vs_concurrency.png"))
        run.save_summary({"rows": rows, "device": "cuda" if cuda_available() else "cpu", "fake": args.fake})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
