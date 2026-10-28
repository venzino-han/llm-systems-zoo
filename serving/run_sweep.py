"""Sweep orchestrator for the Serving SLO lab.

    rate-sweep         open-loop Poisson arrivals at several rates -> saturation curve
    concurrency-sweep  closed-loop at several concurrencies -> throughput/latency curve
    compare            two or more server configs at one fixed load (A/B of scheduler knobs)
    overload           one rate beyond saturation, with and without a client admission limit

Each point is a harness `Run` (records.jsonl, summary.json, manifest.json). The sweep writes
sweep.json, saturation.md and saturation.png under runs/L03_serving_slo/sweeps/ and, with
--report-dir, copies them into reports/.

`--endpoint URL` reuses a running server instead of launching one; `--fake` launches the
in-process fake server (CPU-only smoke of the whole pipeline).
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from llm_systems_zoo.io import dump_json
from serving.load_generator import LoadConfig, parse_slo, run_load
from serving.server import ServerConfig, VLLMServer
from serving.slo_analysis import analyze, markdown_table, plot_saturation
from serving.workloads import WorkloadSpec


@contextlib.contextmanager
def endpoint_for(args: argparse.Namespace, server_cfg: ServerConfig | None) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (endpoint_url, model_name, server_info) for the chosen backend."""
    if args.fake:
        from serving.fake_server import FakeEngine, FakeServerThread

        engine = FakeEngine(max_concurrency=args.fake_capacity, ttft_s=args.fake_ttft, tpot_s=args.fake_tpot)
        with FakeServerThread(engine) as srv:
            yield srv.url, "fake-model", {"label": server_cfg.label if server_cfg else "fake", "backend": "fake",
                                           "max_concurrency": args.fake_capacity, "ttft_s": args.fake_ttft, "tpot_s": args.fake_tpot}
        return
    if args.endpoint:
        yield args.endpoint, args.model, {"label": args.label or "external", "backend": "external", "endpoint": args.endpoint}
        return
    assert server_cfg is not None
    with VLLMServer(server_cfg, startup_timeout_s=args.startup_timeout) as srv:
        info = {**server_cfg.to_dict(), "backend": "vllm", "startup_s": srv.startup_s, "pid": srv.pid}
        yield server_cfg.url, server_cfg.model_name, info


class Sweep:
    def __init__(self, name: str, root: str | Path = "runs", lab: str = "L03_serving_slo") -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / lab / "sweeps" / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.points: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    def add(self, run_dir: Path, label: str, **meta: Any) -> None:
        self.points.append({"run_dir": str(run_dir), "label": label, **meta})

    def finish(self, slo, title: str, report_dir: Path | None, notes: list[str]) -> list[dict[str, Any]]:
        rows = analyze([p["run_dir"] for p in self.points], slo)
        table = markdown_table(rows, slo)
        header = f"# {title}\n\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"
        (self.dir / "saturation.md").write_text(header + table + "\n")
        dump_json({"points": self.points, "rows": rows, "notes": notes, "elapsed_s": time.perf_counter() - self.t0}, self.dir / "sweep.json")
        if rows:
            plot_saturation(rows, self.dir / "saturation.png", title, slo)
        print("\n" + table)
        print(f"\nsweep dir: {self.dir}")
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            for f in ("saturation.md", "saturation.png", "sweep.json"):
                if (self.dir / f).exists():
                    shutil.copy(self.dir / f, report_dir / f)
            for p in self.points:  # curated copy of each point's summary
                src = Path(p["run_dir"]) / "summary.json"
                if src.exists():
                    shutil.copy(src, report_dir / f"summary_{Path(p['run_dir']).name}.json")
            print(f"copied to {report_dir}")
        return rows


def _server_cfg(args: argparse.Namespace) -> ServerConfig | None:
    """The server config is only needed when we launch vLLM ourselves."""
    if args.fake or args.endpoint:
        return None
    return ServerConfig.load(args.server_config)


def _load_cfg(args: argparse.Namespace, endpoint: str, model: str, **over: Any) -> LoadConfig:
    cfg = LoadConfig(endpoint=endpoint, model=model, api=args.api, warmup=args.warmup, timeout_s=args.timeout,
                     ignore_eos=not args.fake and not args.no_ignore_eos)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _tokenizer(args: argparse.Namespace):
    if args.fake or not args.tokenizer:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.tokenizer)


def _notes(args: argparse.Namespace, server_info: dict[str, Any]) -> list[str]:
    notes = [f"workload: {args.workload}", f"backend: {server_info.get('backend')} ({server_info.get('label')})", f"SLO: {args.slo}"]
    if args.tag:
        notes.append(f"tag: {args.tag}")
    return notes


def cmd_rate_sweep(args: argparse.Namespace) -> None:
    rates = [float(r) for r in args.rates.split(",")]
    if args.smoke:
        rates, args.duration = rates[:2], min(args.duration, 3.0)
    server_cfg = _server_cfg(args)
    sweep = Sweep(f"rate_sweep_{Path(args.workload).stem}")
    with endpoint_for(args, server_cfg) as (url, model, info):
        tok = _tokenizer(args)
        for rate in rates:
            spec = WorkloadSpec.load(args.workload)
            cfg = _load_cfg(args, url, model, mode="open", rate=rate, duration_s=args.duration, max_requests=args.max_requests)
            run = run_load(cfg, spec, slo=parse_slo(args.slo), tokenizer=tok, server_config=info, extra_config={"label": info.get("label"), "sweep": str(sweep.dir)},
                           run_name=f"{spec.name}_open_r{rate:g}", collect_env=not args.fake)
            sweep.add(run.dir, info.get("label", ""), rate=rate)
    sweep.finish(parse_slo(args.slo), f"Open-loop rate sweep: {Path(args.workload).stem}", args.report_dir, _notes(args, info))


def cmd_concurrency_sweep(args: argparse.Namespace) -> None:
    concs = [int(c) for c in args.concurrencies.split(",")]
    if args.smoke:
        concs, args.max_requests = concs[:2], 8
    server_cfg = _server_cfg(args)
    sweep = Sweep(f"concurrency_sweep_{Path(args.workload).stem}")
    with endpoint_for(args, server_cfg) as (url, model, info):
        tok = _tokenizer(args)
        for c in concs:
            spec = WorkloadSpec.load(args.workload)
            cfg = _load_cfg(args, url, model, mode="closed", concurrency=c, max_requests=args.max_requests or max(4 * c, 32))
            run = run_load(cfg, spec, slo=parse_slo(args.slo), tokenizer=tok, server_config=info, extra_config={"label": info.get("label"), "sweep": str(sweep.dir)},
                           run_name=f"{spec.name}_closed_c{c}", collect_env=not args.fake)
            sweep.add(run.dir, info.get("label", ""), concurrency=c)
    sweep.finish(parse_slo(args.slo), f"Closed-loop concurrency sweep: {Path(args.workload).stem}", args.report_dir, _notes(args, info))


def cmd_compare(args: argparse.Namespace) -> None:
    """A/B of server configs at one fixed load. Servers are launched one after another."""
    cfg_paths = [Path(p) for p in args.server_configs.split(",")]
    if args.smoke:
        args.duration, args.max_requests = min(args.duration, 3.0), args.max_requests or 8
    sweep = Sweep(f"compare_{Path(args.workload).stem}")
    last_info: dict[str, Any] = {}
    for path in cfg_paths:
        server_cfg = ServerConfig.load(path)
        with endpoint_for(args, server_cfg) as (url, model, info):
            last_info = info
            tok = _tokenizer(args)
            spec = WorkloadSpec.load(args.workload)
            if args.mode == "open":
                cfg = _load_cfg(args, url, model, mode="open", rate=args.rate, duration_s=args.duration, max_requests=args.max_requests)
            else:
                cfg = _load_cfg(args, url, model, mode="closed", concurrency=args.concurrency, max_requests=args.max_requests or 64)
            run = run_load(cfg, spec, slo=parse_slo(args.slo), tokenizer=tok, server_config=info, extra_config={"label": server_cfg.label, "sweep": str(sweep.dir)},
                           run_name=f"{spec.name}_{server_cfg.label}", collect_env=not args.fake)
            sweep.add(run.dir, server_cfg.label, server_config=str(path))
    sweep.finish(parse_slo(args.slo), f"Server config comparison: {Path(args.workload).stem} @ {args.mode} {args.rate if args.mode == 'open' else args.concurrency}", args.report_dir, _notes(args, last_info))


def cmd_overload(args: argparse.Namespace) -> None:
    """Run at `factor` × the given (saturation) rate: without admission control, then with
    `--max-in-flight` limits. Shows queue growth vs. shedding."""
    limits = [None] + [int(x) for x in args.max_in_flight.split(",")] if args.max_in_flight else [None]
    rate = args.rate * args.factor
    if args.smoke:
        args.duration = min(args.duration, 3.0)
    server_cfg = _server_cfg(args)
    sweep = Sweep(f"overload_{Path(args.workload).stem}")
    with endpoint_for(args, server_cfg) as (url, model, info):
        tok = _tokenizer(args)
        for lim in limits:
            spec = WorkloadSpec.load(args.workload)
            cfg = _load_cfg(args, url, model, mode="open", rate=rate, duration_s=args.duration, max_requests=args.max_requests, max_in_flight=lim)
            label = "no_admission" if lim is None else f"max_in_flight_{lim}"
            run = run_load(cfg, spec, slo=parse_slo(args.slo), tokenizer=tok, server_config=info, extra_config={"label": label, "sweep": str(sweep.dir)},
                           run_name=f"{spec.name}_overload_{label}", collect_env=not args.fake)
            sweep.add(run.dir, label, rate=rate, max_in_flight=lim)
    sweep.finish(parse_slo(args.slo), f"Overload at {rate:g} req/s ({args.factor}× {args.rate:g}): admission control", args.report_dir, _notes(args, info))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--workload", default="benchmarks/workloads/short_qa.yaml")
        sp.add_argument("--server-config", default="configs/l03_server_baseline.yaml")
        sp.add_argument("--endpoint", default=None, help="reuse a running server (skips launching)")
        sp.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="model name for --endpoint")
        sp.add_argument("--label", default=None, help="label for runs against --endpoint (e.g. the server config name)")
        sp.add_argument("--api", choices=["completions", "chat"], default="completions")
        sp.add_argument("--tokenizer", default=None, help="HF tokenizer for exact prompt lengths")
        sp.add_argument("--slo", default="ttft=0.5,tpot=0.05")
        sp.add_argument("--warmup", type=int, default=8)
        sp.add_argument("--timeout", type=float, default=120.0)
        sp.add_argument("--max-requests", type=int, default=None)
        sp.add_argument("--duration", type=float, default=30.0)
        sp.add_argument("--no-ignore-eos", action="store_true")
        sp.add_argument("--startup-timeout", type=float, default=900)
        sp.add_argument("--report-dir", type=Path, default=None)
        sp.add_argument("--tag", default="")
        sp.add_argument("--smoke", action="store_true")
        sp.add_argument("--fake", action="store_true", help="use the in-process fake server (CPU)")
        sp.add_argument("--fake-capacity", type=int, default=8)
        sp.add_argument("--fake-ttft", type=float, default=0.02)
        sp.add_argument("--fake-tpot", type=float, default=0.005)

    sp = sub.add_parser("rate-sweep", help="open-loop Poisson rate sweep")
    common(sp)
    sp.add_argument("--rates", default="1,2,4,8,16,32")
    sp.set_defaults(func=cmd_rate_sweep)

    sp = sub.add_parser("concurrency-sweep", help="closed-loop concurrency sweep")
    common(sp)
    sp.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    sp.set_defaults(func=cmd_concurrency_sweep)

    sp = sub.add_parser("compare", help="A/B server configs at fixed load")
    common(sp)
    sp.add_argument("--server-configs", required=True, help="comma-separated YAML paths")
    sp.add_argument("--mode", choices=["open", "closed"], default="open")
    sp.add_argument("--rate", type=float, default=8.0)
    sp.add_argument("--concurrency", type=int, default=16)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("overload", help="beyond-saturation behaviour with/without admission control")
    common(sp)
    sp.add_argument("--rate", type=float, required=True, help="saturation rate (req/s)")
    sp.add_argument("--factor", type=float, default=2.0)
    sp.add_argument("--max-in-flight", default="", help="comma-separated client admission limits to try")
    sp.set_defaults(func=cmd_overload)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
