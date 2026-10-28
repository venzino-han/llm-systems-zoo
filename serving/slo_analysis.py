"""Turn one or many load-generator runs into an SLO report and a saturation curve.

Input: run directories written by `serving.load_generator` (records.jsonl + summary.json).
Output: a markdown table, a JSON summary and a three-panel figure:
    offered load -> throughput, offered load -> p99 TTFT/TPOT, offered load -> goodput.

Goodput and attainment are recomputed here from the raw records, so a different SLO can be
applied after the fact without re-running the benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

from llm_systems_zoo import SLO, RequestRecord
from llm_systems_zoo.io import dump_json, read_jsonl
from llm_systems_zoo.metrics import full_report
from llm_systems_zoo.plotting import PALETTE, plt, save_fig, setup_style

UNDERSAMPLED_P99 = 100


def record_from_dict(d: dict[str, Any]) -> RequestRecord:
    keys = {"request_id", "t_submit", "t_first_token", "t_end", "prompt_tokens", "output_tokens", "error"}
    extra = {k: v for k, v in d.items() if k not in keys and k not in {"ttft", "tpot", "e2e"}}
    return RequestRecord(
        request_id=str(d["request_id"]),
        t_submit=float(d["t_submit"]),
        prompt_tokens=int(d.get("prompt_tokens") or 0),
        output_tokens=int(d.get("output_tokens") or 0),
        t_first_token=d.get("t_first_token"),
        t_end=d.get("t_end"),
        error=d.get("error"),
        extra=extra,
    )


def load_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    records = [record_from_dict(d) for d in read_jsonl(run_dir / "records.jsonl")]
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    return {"dir": run_dir, "records": records, "summary": summary, "config": config}


def analyze_run(run: dict[str, Any], slo: SLO | None) -> dict[str, Any]:
    rep = full_report(run["records"], slo)
    lat = rep["latency"]
    load_cfg = run["config"].get("load", {})
    offered = run["summary"].get("offered_load")
    if offered is None:
        offered = load_cfg.get("concurrency") if load_cfg.get("mode") == "closed" else load_cfg.get("rate")
    row: dict[str, Any] = {
        "run": run["dir"].name,
        "mode": load_cfg.get("mode", run["summary"].get("mode")),
        "offered_load": offered,
        "label": run["config"].get("label"),
        "n_submitted": lat["n_submitted"],
        "n_success": lat["n_success"],
        "error_rate": lat["error_rate"],
        "requests_per_s": rep["throughput"]["requests_per_s"],
        "output_tokens_per_s": rep["throughput"]["output_tokens_per_s"],
        "input_tokens_per_s": rep["throughput"]["input_tokens_per_s"],
        "ttft_mean": lat["ttft_s"].get("mean"),
        "ttft_p50": lat["ttft_s"].get("p50"),
        "ttft_p95": lat["ttft_s"].get("p95"),
        "ttft_p99": lat["ttft_s"].get("p99"),
        "tpot_mean": lat["tpot_s"].get("mean"),
        "tpot_p50": lat["tpot_s"].get("p50"),
        "tpot_p95": lat["tpot_s"].get("p95"),
        "tpot_p99": lat["tpot_s"].get("p99"),
        "e2e_p50": lat["e2e_s"].get("p50"),
        "e2e_p99": lat["e2e_s"].get("p99"),
        "p99_undersampled": lat["n_success"] < UNDERSAMPLED_P99,
        "max_in_flight": run["summary"].get("meta", {}).get("max_in_flight_observed"),
        "rejected": run["summary"].get("meta", {}).get("rejected_admission", 0),
    }
    if slo is not None:
        row["goodput_rps"] = rep["goodput"]["goodput_rps"]
        row["slo_attainment"] = rep["goodput"]["slo_attainment"]
    return row


def analyze(run_dirs: Sequence[str | Path], slo: SLO | None) -> list[dict[str, Any]]:
    rows = [analyze_run(load_run(d), slo) for d in run_dirs]
    rows.sort(key=lambda r: ((r["label"] or ""), r["offered_load"] if r["offered_load"] is not None else 0))
    return rows


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.0f}"


def markdown_table(rows: Sequence[dict[str, Any]], slo: SLO | None) -> str:
    cols = ["label", "mode", "offered_load", "n_success/n_submitted", "req/s", "out tok/s", "TTFT p50/p99 (ms)", "TPOT p50/p99 (ms)", "E2E p99 (s)", "err"]
    if slo is not None:
        cols += ["goodput req/s", "SLO attain."]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [
            str(r.get("label") or ""),
            str(r["mode"]),
            f"{r['offered_load']:g}" if isinstance(r["offered_load"], (int, float)) else str(r["offered_load"]),
            f"{r['n_success']}/{r['n_submitted']}" + (" *" if r["p99_undersampled"] else ""),
            f"{r['requests_per_s']:.2f}",
            f"{r['output_tokens_per_s']:.0f}",
            f"{_ms(r['ttft_p50'])}/{_ms(r['ttft_p99'])}",
            f"{_ms(r['tpot_p50'])}/{_ms(r['tpot_p99'])}",
            f"{r['e2e_p99']:.2f}" if r["e2e_p99"] is not None else "-",
            f"{r['error_rate']:.2f}" + (f" ({r['rejected']} rej)" if r.get("rejected") else ""),
        ]
        if slo is not None:
            cells += [f"{r['goodput_rps']:.2f}", f"{r['slo_attainment']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    if any(r["p99_undersampled"] for r in rows):
        lines.append("")
        lines.append(f"\\* fewer than {UNDERSAMPLED_P99} successful requests: p99 is under-sampled.")
    if slo is not None:
        lines.append("")
        lines.append(f"SLO: {slo.to_dict()}. Attainment is over *submitted* requests (failures and rejections count).")
    return "\n".join(lines)


def plot_saturation(rows: Sequence[dict[str, Any]], path: str | Path, title: str = "", slo: SLO | None = None) -> Path:
    setup_style()
    labels = sorted({r["label"] or "" for r in rows})
    fig, axes = plt.subplots(1, 3 if slo is not None else 2, figsize=(13 if slo is not None else 9, 3.8))
    for i, label in enumerate(labels):
        sub = sorted([r for r in rows if (r["label"] or "") == label], key=lambda r: r["offered_load"])
        x = [r["offered_load"] for r in sub]
        c = PALETTE[i % len(PALETTE)]
        name = label or "runs"
        axes[0].plot(x, [r["requests_per_s"] for r in sub], marker="o", color=c, label=f"{name}: achieved req/s")
        axes[0].plot(x, [r["output_tokens_per_s"] / 100 for r in sub], marker="s", linestyle="--", color=c, alpha=0.6, label=f"{name}: out tok/s ÷100")
        axes[1].plot(x, [(r["ttft_p99"] or 0) * 1000 for r in sub], marker="o", color=c, label=f"{name}: TTFT p99")
        axes[1].plot(x, [(r["tpot_p99"] or 0) * 1000 for r in sub], marker="s", linestyle="--", color=c, label=f"{name}: TPOT p99")
        if slo is not None:
            axes[2].plot(x, [r["goodput_rps"] for r in sub], marker="o", color=c, label=f"{name}: goodput")
            axes[2].plot(x, [r["slo_attainment"] * max(r["offered_load"] for r in sub) for r in sub], linestyle=":", color=c, alpha=0.5, label=f"{name}: attainment×max load")
    mode = rows[0]["mode"] if rows else ""
    xlabel = "offered concurrency" if mode == "closed" else "offered arrival rate (req/s)"
    if mode == "open":
        lim = max(r["offered_load"] for r in rows)
        axes[0].plot([0, lim], [0, lim], color="gray", linewidth=0.8, linestyle=":", label="ideal (achieved = offered)")
    axes[0].set(xlabel=xlabel, ylabel="throughput", title="Throughput")
    axes[1].set(xlabel=xlabel, ylabel="ms", title="Tail latency (p99)")
    axes[1].set_yscale("log")
    if slo is not None:
        axes[2].set(xlabel=xlabel, ylabel="req/s", title=f"SLO goodput (TTFT≤{slo.ttft_s}s, TPOT≤{slo.tpot_s}s)")
        axes[2].legend(fontsize=7)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    return save_fig(fig, path)


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories or globs (e.g. 'runs/L03_serving_slo/*short_qa_open*')")
    p.add_argument("--slo", default="ttft=0.5,tpot=0.05")
    p.add_argument("--out", type=Path, default=None, help="directory for saturation.md/.json/.png")
    p.add_argument("--title", default="")
    a = p.parse_args(argv)
    from serving.load_generator import parse_slo

    dirs: list[str] = []
    for pattern in a.runs:
        matches = sorted(glob.glob(pattern))
        dirs.extend(matches or [pattern])
    slo = parse_slo(a.slo)
    rows = analyze(dirs, slo)
    table = markdown_table(rows, slo)
    print(table)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "saturation.md").write_text(table + "\n")
        dump_json(rows, a.out / "saturation.json")
        plot_saturation(rows, a.out / "saturation.png", a.title, slo)
        print("wrote", a.out)
    return rows


if __name__ == "__main__":
    main()
