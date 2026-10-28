"""Async load generator for OpenAI-compatible servers (vLLM, SGLang, TGI, the fake server).

Measures, per request: submit time, first-token time, end time, prompt/output tokens, errors.
Supports *closed-loop* load (fixed concurrency) and *open-loop* load (requests fired at
pre-computed Poisson/burst arrival times regardless of the server's state), warm-up
exclusion, per-request time-outs, and an optional client-side *admission limit* that rejects
a request immediately when too many are already in flight.

Only the client's view is recorded: TTFT includes queueing inside the server, and output
token counts come from the `usage` field when the server sends one (vLLM does, with
`stream_options.include_usage`), otherwise from counting streamed chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx

from llm_systems_zoo import SLO, RequestRecord, Run
from llm_systems_zoo.metrics import full_report
from serving.workloads import GeneratedRequest, WorkloadGenerator, WorkloadSpec, describe


@dataclass
class LoadConfig:
    endpoint: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    api: str = "completions"  # completions | chat
    mode: str = "closed"  # closed | open
    concurrency: int = 8
    rate: float = 4.0
    duration_s: float = 30.0
    max_requests: int | None = None
    warmup: int = 4
    timeout_s: float = 120.0
    max_in_flight: int | None = None  # client-side admission control (open loop only)
    ignore_eos: bool = True
    temperature: float = 0.0
    system_prompt: str | None = None  # chat api only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Stats:
    def __init__(self) -> None:
        self.in_flight = 0
        self.rejected = 0
        self.max_in_flight = 0


async def _stream_request(client: httpx.AsyncClient, cfg: LoadConfig, req: GeneratedRequest) -> RequestRecord:
    """Send one streaming request and record its timeline."""
    body: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": req.max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cfg.ignore_eos:
        body["ignore_eos"] = True  # vLLM extension: keeps output length = max_tokens
    if cfg.api == "chat":
        msgs = []
        if cfg.system_prompt:
            msgs.append({"role": "system", "content": cfg.system_prompt})
        msgs.append({"role": "user", "content": req.prompt})
        body["messages"] = msgs
        url = f"{cfg.endpoint}/v1/chat/completions"
    else:
        body["prompt"] = req.prompt
        url = f"{cfg.endpoint}/v1/completions"

    rec = RequestRecord(request_id=req.request_id, t_submit=time.perf_counter(), prompt_tokens=req.prompt_tokens_target,
                        extra={"request_class": req.request_class, "max_tokens": req.max_tokens, "arrival_offset_s": req.arrival_offset_s})
    chunks_with_text = 0
    usage: dict[str, Any] | None = None
    try:
        async with client.stream("POST", url, json=body, timeout=cfg.timeout_s) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:200]
                rec.error = f"http_{resp.status_code}: {text}"
                rec.t_end = time.perf_counter()
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    c = choices[0]
                    text = c.get("text") if cfg.api == "completions" else (c.get("delta") or {}).get("content")
                    if text:
                        chunks_with_text += 1
                        if rec.t_first_token is None:
                            rec.t_first_token = time.perf_counter()
        rec.t_end = time.perf_counter()
    except (httpx.TimeoutException, asyncio.TimeoutError):
        rec.error = "timeout"
        rec.t_end = time.perf_counter()
    except Exception as e:  # connection refused, protocol errors, ...
        rec.error = f"{type(e).__name__}: {e}"[:200]
        rec.t_end = time.perf_counter()
        return rec

    if usage:
        rec.prompt_tokens = int(usage.get("prompt_tokens", rec.prompt_tokens))
        rec.output_tokens = int(usage.get("completion_tokens", chunks_with_text))
    else:
        rec.output_tokens = chunks_with_text
    if rec.error is None and rec.t_first_token is None:
        rec.error = "no_tokens"
    return rec


async def _closed_loop(client: httpx.AsyncClient, cfg: LoadConfig, requests: list[GeneratedRequest], stats: _Stats) -> list[RequestRecord]:
    queue: asyncio.Queue[GeneratedRequest | None] = asyncio.Queue()
    for r in requests:
        queue.put_nowait(r)
    for _ in range(cfg.concurrency):
        queue.put_nowait(None)
    records: list[RequestRecord] = []

    async def worker() -> None:
        while True:
            req = await queue.get()
            if req is None:
                return
            stats.in_flight += 1
            stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
            try:
                records.append(await _stream_request(client, cfg, req))
            finally:
                stats.in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
    return records


async def _open_loop(client: httpx.AsyncClient, cfg: LoadConfig, requests: list[GeneratedRequest], stats: _Stats) -> list[RequestRecord]:
    """Fire each request at its arrival offset. Never waits for the server."""
    records: list[RequestRecord] = []
    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []

    async def fire(req: GeneratedRequest) -> None:
        delay = t0 + req.arrival_offset_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if cfg.max_in_flight is not None and stats.in_flight >= cfg.max_in_flight:
            stats.rejected += 1
            now = time.perf_counter()
            records.append(RequestRecord(req.request_id, now, req.prompt_tokens_target, 0, None, now, "rejected_admission",
                                         {"request_class": req.request_class, "max_tokens": req.max_tokens, "arrival_offset_s": req.arrival_offset_s}))
            return
        stats.in_flight += 1
        stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
        try:
            records.append(await _stream_request(client, cfg, req))
        finally:
            stats.in_flight -= 1

    for req in requests:
        tasks.append(asyncio.create_task(fire(req)))
    await asyncio.gather(*tasks)
    return records


def plan_requests(spec: WorkloadSpec, cfg: LoadConfig, tokenizer: Any | None = None, dataset_prompts: Any | None = None) -> tuple[list[GeneratedRequest], list[GeneratedRequest]]:
    """Materialise warm-up + measured requests for the configured mode."""
    spec.arrival.mode = cfg.mode
    spec.arrival.concurrency = cfg.concurrency
    spec.arrival.rate = cfg.rate
    spec.arrival.duration_s = cfg.duration_s
    if cfg.max_requests:
        n = cfg.max_requests
    elif cfg.mode == "open":
        n = max(1, int(math.ceil(cfg.rate * cfg.duration_s)))
    else:
        n = spec.num_requests
    gen = WorkloadGenerator(spec, tokenizer=tokenizer, dataset_prompts=dataset_prompts)
    measured = gen.requests(n)
    warm_spec = WorkloadSpec.from_dict(spec.to_dict())
    warm_spec.seed = spec.seed + 1000
    warm_spec.arrival.mode = "closed"
    warm = WorkloadGenerator(warm_spec, tokenizer=tokenizer, dataset_prompts=dataset_prompts).requests(cfg.warmup) if cfg.warmup else []
    return warm, measured


async def run_load_async(cfg: LoadConfig, warm: list[GeneratedRequest], measured: list[GeneratedRequest]) -> tuple[list[RequestRecord], dict[str, Any]]:
    # Open loop must never be throttled by the client's connection pool, or pool waiting would
    # masquerade as server queueing; closed loop needs exactly `concurrency` connections.
    if cfg.mode == "closed":
        limits = httpx.Limits(max_connections=cfg.concurrency + 8, max_keepalive_connections=cfg.concurrency + 8)
    else:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
    stats = _Stats()
    async with httpx.AsyncClient(limits=limits, timeout=cfg.timeout_s) as client:
        if warm:
            warm_cfg = LoadConfig(**{**cfg.to_dict(), "mode": "closed", "concurrency": min(cfg.concurrency, len(warm), 8)})
            await _closed_loop(client, warm_cfg, warm, _Stats())
        t_start = time.perf_counter()
        if cfg.mode == "closed":
            records = await _closed_loop(client, cfg, measured, stats)
        else:
            records = await _open_loop(client, cfg, measured, stats)
        t_end = time.perf_counter()
    records.sort(key=lambda r: r.t_submit)
    meta = {"wall_s": t_end - t_start, "max_in_flight_observed": stats.max_in_flight, "rejected_admission": stats.rejected}
    return records, meta


def run_load(
    cfg: LoadConfig,
    spec: WorkloadSpec,
    *,
    lab: str = "L03_serving_slo",
    run_name: str | None = None,
    slo: SLO | None = None,
    tokenizer: Any | None = None,
    dataset_prompts: Any | None = None,
    server_config: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
    run_root: str | Path = "runs",
    collect_env: bool = True,
    quiet: bool = False,
) -> Run:
    """Run one load point and persist records + report in a harness `Run` directory."""
    warm, measured = plan_requests(spec, cfg, tokenizer, dataset_prompts)
    config = {"load": cfg.to_dict(), "workload": describe(spec), "server": server_config or {}, "slo": slo.to_dict() if slo else None, **(extra_config or {})}
    name = run_name or f"{spec.name}_{cfg.mode}_{cfg.concurrency if cfg.mode == 'closed' else cfg.rate}"
    with Run(lab, name, config, root=run_root, model={"id": cfg.model, "server": server_config or {}}, workload=describe(spec),
             execution={"warmup_requests": len(warm), "measured_requests": len(measured), "timeout_s": cfg.timeout_s, "mode": cfg.mode}, collect_env=collect_env) as run:
        records, meta = asyncio.run(run_load_async(cfg, warm, measured))
        run.log_records([r.to_dict() for r in records])
        report = full_report(records, slo)
        report["offered_load"] = cfg.concurrency if cfg.mode == "closed" else cfg.rate
        report["mode"] = cfg.mode
        report["meta"] = meta
        report["p99_undersampled"] = report["latency"]["n_success"] < 100
        run.save_summary(report)
        if not quiet:
            print(format_summary(report))
    return run


def format_summary(rep: dict[str, Any]) -> str:
    lat, th = rep["latency"], rep["throughput"]
    lines = [
        f"offered_load={rep.get('offered_load')} mode={rep.get('mode')}  requests: {lat['n_success']}/{lat['n_submitted']} ok  "
        f"(error_rate={lat['error_rate']:.3f})  window={th['window_s']:.1f}s",
        f"  throughput: {th['requests_per_s']:.2f} req/s  in={th['input_tokens_per_s']:.0f} tok/s  out={th['output_tokens_per_s']:.0f} tok/s",
    ]
    for key in ("ttft_s", "tpot_s", "e2e_s"):
        s = lat[key]
        if s.get("n"):
            lines.append(f"  {key:7s} mean={s['mean']*1000:7.1f}ms p50={s['p50']*1000:7.1f}ms p95={s['p95']*1000:7.1f}ms p99={s['p99']*1000:7.1f}ms")
    if "goodput" in rep:
        g = rep["goodput"]
        lines.append(f"  goodput: {g['goodput_rps']:.2f} req/s  slo_attainment={g['slo_attainment']:.3f}  (SLO {g['slo']})")
    if rep.get("p99_undersampled"):
        lines.append("  note: fewer than 100 successful requests; p99 is under-sampled")
    return "\n".join(lines)


def parse_slo(text: str | None) -> SLO | None:
    """'ttft=0.5,tpot=0.05,e2e=10' -> SLO"""
    if not text:
        return None
    kv = dict(part.split("=") for part in text.split(","))
    return SLO(ttft_s=float(kv["ttft"]) if "ttft" in kv else None, tpot_s=float(kv["tpot"]) if "tpot" in kv else None, e2e_s=float(kv["e2e"]) if "e2e" in kv else None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--api", choices=["completions", "chat"], default="completions")
    p.add_argument("--workload", type=Path, default=Path("benchmarks/workloads/short_qa.yaml"))
    p.add_argument("--mode", choices=["closed", "open"], default="closed")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rate", type=float, default=4.0, help="open-loop arrival rate (req/s)")
    p.add_argument("--duration", type=float, default=30.0, help="open-loop duration (s); requests = rate*duration")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-in-flight", type=int, default=None, help="client-side admission limit (open loop)")
    p.add_argument("--no-ignore-eos", action="store_true")
    p.add_argument("--slo", default="ttft=0.5,tpot=0.05", help="e.g. ttft=0.5,tpot=0.05,e2e=10")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id for exact prompt lengths (optional)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-root", default="runs")
    p.add_argument("--smoke", action="store_true", help="8 requests, closed loop, concurrency 2")
    return p


def main(argv: Sequence[str] | None = None) -> Run:
    a = build_parser().parse_args(argv)
    cfg = LoadConfig(endpoint=a.endpoint, model=a.model, api=a.api, mode=a.mode, concurrency=a.concurrency, rate=a.rate,
                     duration_s=a.duration, max_requests=a.max_requests, warmup=a.warmup, timeout_s=a.timeout,
                     max_in_flight=a.max_in_flight, ignore_eos=not a.no_ignore_eos)
    if a.smoke:
        cfg.mode, cfg.concurrency, cfg.max_requests, cfg.warmup = "closed", 2, 8, 1
    spec = WorkloadSpec.load(a.workload)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    run = run_load(cfg, spec, slo=parse_slo(a.slo), tokenizer=tok, run_name=a.run_name, run_root=a.run_root)
    print("run dir:", run.dir)
    return run


if __name__ == "__main__":
    main()
