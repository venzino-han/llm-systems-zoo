"""TP vs replicas vs single GPU: which layout serves this model/workload best on N GPUs?

For a given GPU count N the script runs, one after another (never concurrently):

    single    one vLLM server on GPU 0                           (baseline; needs the model to fit)
    tp        one vLLM server with --tensor-parallel-size N      (one model copy, sharded)
    replica   N single-GPU vLLM servers behind the round-robin proxy (N model copies)

and drives each with the *same* L03 workload and load generator, so throughput, per-GPU
throughput, TTFT/TPOT percentiles and SLO goodput are directly comparable. Each layout is a
harness `Run`; the comparison table/figure lands in runs/L04_multi_gpu_inference/compare_*/
and, with --report-dir, in reports/.

    python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 --layouts single,tp,replica --dry-run
    python -m inference.multi_gpu.bench_tp_vs_replica --gpus 0,1 --model Qwen/Qwen2.5-7B-Instruct \
        --workload benchmarks/workloads/short_qa.yaml --mode closed --concurrencies 4,16,64

Heterogeneous GPUs (e.g. an A6000 next to a Blackwell card) are a confound: TP shards
memory evenly and runs at the slower card's pace, while each replica runs at its own pace.
Report replica results per device and never average a TP result over "N GPUs" as if they
were identical. `--homogeneous-check` refuses to run TP when the device names differ.
"""

from __future__ import annotations

import argparse
import contextlib
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from llm_systems_zoo.hardware import gpu_inventory, gpu_topology
from llm_systems_zoo.io import dump_json
from llm_systems_zoo.plotting import PALETTE, plt, save_fig, setup_style
from serving.load_generator import LoadConfig, parse_slo, run_load
from serving.server import ServerConfig, VLLMServer, wait_for_health
from serving.slo_analysis import analyze, markdown_table
from serving.workloads import WorkloadSpec

LAB = "L04_multi_gpu_inference"


# ---------------------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------------------
def layout_configs(layout: str, gpus: Sequence[int], base: ServerConfig, port0: int) -> list[ServerConfig]:
    """Server configs for a layout. Replica layout returns one config per GPU."""
    env_base = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if layout == "single":
        c = ServerConfig(**{**base.to_dict(), "label": "single", "port": port0, "tensor_parallel_size": 1,
                            "env": {**env_base, "CUDA_VISIBLE_DEVICES": str(gpus[0])}})
        return [c]
    if layout == "tp":
        c = ServerConfig(**{**base.to_dict(), "label": f"tp{len(gpus)}", "port": port0, "tensor_parallel_size": len(gpus),
                            "env": {**env_base, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}})
        return [c]
    if layout == "replica":
        return [
            ServerConfig(**{**base.to_dict(), "label": f"replica{len(gpus)}_gpu{g}", "port": port0 + 1 + i, "tensor_parallel_size": 1,
                            "env": {**env_base, "CUDA_VISIBLE_DEVICES": str(g)}})
            for i, g in enumerate(gpus)
        ]
    raise ValueError(f"unknown layout {layout}")


def proxy_command(port: int, backends: Sequence[str], python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "inference.multi_gpu.proxy", "--port", str(port), "--backends", ",".join(backends)]


def render_plan(layouts: Sequence[str], gpus: Sequence[int], base: ServerConfig, port0: int) -> list[dict[str, Any]]:
    """What would run, as shell commands (used by --dry-run and tests)."""
    plan = []
    for layout in layouts:
        cfgs = layout_configs(layout, gpus, base, port0)
        cmds = [f"{' '.join(f'{k}={v}' for k, v in c.env.items())} {shlex.join(c.command())}" for c in cfgs]
        endpoint = cfgs[0].url
        if layout == "replica":
            cmds.append(shlex.join(proxy_command(port0, [c.url for c in cfgs])))
            endpoint = f"http://127.0.0.1:{port0}"
        plan.append({"layout": layout, "gpus": list(gpus) if layout != "single" else [gpus[0]], "n_gpus": 1 if layout == "single" else len(gpus),
                     "servers": [c.to_dict() for c in cfgs], "commands": cmds, "endpoint": endpoint})
    return plan


@contextlib.contextmanager
def launch_layout(layout: str, gpus: Sequence[int], base: ServerConfig, port0: int, startup_timeout: float) -> Iterator[tuple[str, dict[str, Any]]]:
    """Start the servers (and proxy) for a layout; yield (endpoint, info)."""
    cfgs = layout_configs(layout, gpus, base, port0)
    servers: list[VLLMServer] = []
    proxy: subprocess.Popen | None = None
    try:
        for c in cfgs:
            servers.append(VLLMServer(c, startup_timeout_s=startup_timeout).start())
        endpoint = cfgs[0].url
        if layout == "replica":
            proxy = subprocess.Popen(proxy_command(port0, [c.url for c in cfgs]), start_new_session=True)
            endpoint = f"http://127.0.0.1:{port0}"
            wait_for_health(endpoint, 60, proxy)
        info = {"layout": layout, "n_gpus": 1 if layout == "single" else len(gpus), "gpus": list(gpus), "servers": [c.to_dict() for c in cfgs],
                "startup_s": [s.startup_s for s in servers], "tensor_parallel_size": cfgs[0].tensor_parallel_size, "replicas": len(cfgs)}
        yield endpoint, info
    finally:
        if proxy is not None and proxy.poll() is None:
            proxy.terminate()
            try:
                proxy.wait(10)
            except subprocess.TimeoutExpired:
                proxy.kill()
        for s in servers:
            s.stop()


# ---------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------
def compare_table(rows: Sequence[dict[str, Any]], n_gpus: dict[str, int]) -> str:
    lines = ["| layout | GPUs | load | req/s | out tok/s | out tok/s per GPU | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | goodput req/s | SLO attain. |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = n_gpus.get(r["label"], 1)
        ms = lambda x: "-" if x is None else f"{x*1000:.0f}"  # noqa: E731
        lines.append(f"| {r['label']} | {n} | {r['offered_load']:g} | {r['requests_per_s']:.2f} | {r['output_tokens_per_s']:.0f} | {r['output_tokens_per_s']/n:.0f} | "
                     f"{ms(r['ttft_p50'])}/{ms(r['ttft_p99'])} | {ms(r['tpot_p50'])}/{ms(r['tpot_p99'])} | {r.get('goodput_rps', float('nan')):.2f} | {r.get('slo_attainment', float('nan')):.2f} |")
    return "\n".join(lines)


def plot_compare(rows: Sequence[dict[str, Any]], n_gpus: dict[str, int], path: Path, title: str) -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = sorted({r["label"] for r in rows})
    for i, label in enumerate(labels):
        sub = sorted([r for r in rows if r["label"] == label], key=lambda r: r["offered_load"])
        x = [r["offered_load"] for r in sub]
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(x, [r["output_tokens_per_s"] for r in sub], marker="o", color=c, label=label)
        axes[1].plot(x, [r["output_tokens_per_s"] / n_gpus.get(label, 1) for r in sub], marker="o", color=c, label=label)
        axes[2].plot(x, [(r["tpot_p50"] or 0) * 1000 for r in sub], marker="o", color=c, label=f"{label} TPOT p50")
        axes[2].plot(x, [(r["ttft_p99"] or 0) * 1000 for r in sub], marker="s", linestyle="--", color=c, label=f"{label} TTFT p99")
    xlabel = "offered concurrency" if rows and rows[0]["mode"] == "closed" else "offered rate (req/s)"
    axes[0].set(xlabel=xlabel, ylabel="output tok/s", title="Aggregate throughput")
    axes[1].set(xlabel=xlabel, ylabel="output tok/s per GPU", title="Per-GPU throughput")
    axes[2].set(xlabel=xlabel, ylabel="ms", title="Latency")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(title)
    return save_fig(fig, path)


def run(args: argparse.Namespace) -> Path:
    gpus = [int(g) for g in args.gpus.split(",")]
    layouts = args.layouts.split(",")
    base = ServerConfig.load(args.server_config) if args.server_config else ServerConfig()
    base.model = args.model or base.model
    base.gpu_memory_utilization = args.gpu_memory_utilization or base.gpu_memory_utilization
    base.max_model_len = args.max_model_len or base.max_model_len
    inv = gpu_inventory()
    names = {g["index"]: g["name"] for g in inv}
    hetero = len({names.get(g, "?") for g in gpus}) > 1
    plan = render_plan(layouts, gpus, base, args.port)
    if args.dry_run:
        print(f"GPUs {gpus}: " + ", ".join(f"[{g}] {names.get(g, 'unknown')}" for g in gpus) + ("  (HETEROGENEOUS: TP results are bounded by the slower/smaller card)" if hetero else ""))
        for p in plan:
            print(f"\n== layout {p['layout']} ({p['n_gpus']} GPU) -> endpoint {p['endpoint']}")
            for c in p["commands"]:
                print("  " + c)
        return Path(".")
    if hetero and args.homogeneous_check and "tp" in layouts:
        raise SystemExit("refusing TP on heterogeneous GPUs (--no-homogeneous-check to override)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / LAB / f"compare_{stamp}_{Path(args.workload).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loads = [int(x) for x in args.concurrencies.split(",")] if args.mode == "closed" else [float(x) for x in args.rates.split(",")]
    if args.smoke:
        loads = loads[:1]
    slo = parse_slo(args.slo)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    run_dirs: list[Path] = []
    n_gpus: dict[str, int] = {}
    topo = gpu_topology()
    for layout in layouts:
        with launch_layout(layout, gpus, base, args.port, args.startup_timeout) as (endpoint, info):
            label = info["layout"] if info["layout"] == "single" else f"{info['layout']}{info['n_gpus']}"
            n_gpus[label] = info["n_gpus"]
            info["gpu_names"] = [names.get(g, "?") for g in (gpus if layout != "single" else gpus[:1])]
            info["interconnect"] = {k: v for k, v in topo.items() if k != "raw"}
            info["heterogeneous"] = hetero
            for load in loads:
                spec = WorkloadSpec.load(args.workload)
                if args.mode == "closed":
                    cfg = LoadConfig(endpoint=endpoint, model=base.model_name, mode="closed", concurrency=int(load), max_requests=args.max_requests or max(4 * int(load), 32), warmup=args.warmup, timeout_s=args.timeout)
                else:
                    cfg = LoadConfig(endpoint=endpoint, model=base.model_name, mode="open", rate=float(load), duration_s=args.duration, max_requests=args.max_requests, warmup=args.warmup, timeout_s=args.timeout)
                if args.smoke:
                    cfg.max_requests, cfg.warmup = 8, 1
                r = run_load(cfg, spec, lab=LAB, slo=slo, tokenizer=tok, server_config=info, extra_config={"label": label, "compare": str(out_dir)},
                             run_name=f"{spec.name}_{label}_{args.mode}_{load:g}")
                run_dirs.append(r.dir)
            time.sleep(2)
    rows = analyze(run_dirs, slo)
    table = compare_table(rows, n_gpus)
    notes = [f"model: {base.model}", f"workload: {args.workload}", "GPUs: " + ", ".join(f"[{g}] {names.get(g, '?')}" for g in gpus),
             f"interconnect: {topo.get('links', {})} (NVLink: {topo.get('has_nvlink')})", f"heterogeneous: {hetero}", f"SLO: {args.slo}"]
    (out_dir / "compare.md").write_text(f"# TP vs replica vs single: {Path(args.workload).stem}\n\n" + "\n".join(f"- {n}" for n in notes) + "\n\n" + table + "\n\n" + markdown_table(rows, slo) + "\n")
    dump_json({"rows": rows, "n_gpus": n_gpus, "run_dirs": [str(d) for d in run_dirs], "notes": notes, "plan": plan}, out_dir / "compare.json")
    plot_compare(rows, n_gpus, out_dir / "compare.png", f"{base.model} on {len(gpus)} GPUs")
    print("\n".join(notes) + "\n\n" + table + f"\n\nwrote {out_dir}")
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        for f in ("compare.md", "compare.json", "compare.png"):
            shutil.copy(out_dir / f, args.report_dir / f)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0,1", help="GPU indices (PCI_BUS_ID order)")
    p.add_argument("--layouts", default="single,tp,replica")
    p.add_argument("--model", default=None)
    p.add_argument("--server-config", default="configs/l04_server_base.yaml")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--port", type=int, default=8030, help="base port; replicas use port+1..port+N")
    p.add_argument("--workload", default="benchmarks/workloads/short_qa.yaml")
    p.add_argument("--mode", choices=["closed", "open"], default="closed")
    p.add_argument("--concurrencies", default="1,8,32,128")
    p.add_argument("--rates", default="2,8,32")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--slo", default="ttft=0.5,tpot=0.05")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--startup-timeout", type=float, default=1200)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--homogeneous-check", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> Path:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
