"""Turn DDP/FSDP run summaries into a scaling-efficiency table and figure.

    python training/distributed/scaling_analysis.py --runs-root runs --out reports/L05_training_efficiency/results

Efficiency definitions:
  strong scaling: speedup(N) = step_time(1) / step_time(N); efficiency = speedup / N
  weak scaling:   efficiency = aggregate_tokens_per_s(N) / (N * tokens_per_s(1))
The world-size-1 run of the *same* strategy and scaling mode is the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_systems_zoo.io import dump_json
from llm_systems_zoo.plotting import plot_lines


def load_summaries(runs_root: Path) -> list[dict]:
    out = []
    for p in sorted((runs_root / "L05_training_efficiency").glob("*/summary.json")):
        s = json.loads(p.read_text())
        if "strategy" in s and "world_size" in s:
            s["_path"] = str(p.parent)
            out.append(s)
    return out


def scaling_table(summaries: list[dict]) -> list[dict]:
    rows = []
    by_key: dict[tuple, dict[int, dict]] = {}
    for s in summaries:
        by_key.setdefault((s["strategy"], s["scaling"]), {})[s["world_size"]] = s
    for (strategy, scaling), runs in sorted(by_key.items()):
        ref = runs.get(1)
        for n, s in sorted(runs.items()):
            row = {"strategy": strategy, "scaling": scaling, "world_size": n, "global_batch": s["global_batch"],
                   "aggregate_tok_s": s["aggregate_real_tokens_per_s"], "step_ms": s["mean_step_s"] * 1000,
                   "peak_alloc_gib_max": s["peak_allocated_gib_max"], "run": s["_path"]}
            probe = s["per_rank"][0]
            for k in ("exposed_comm_s", "comm_kernel_s_per_step", "total_kernel_s_per_step"):
                if k in probe:
                    row[k] = probe[k]
            if ref is not None:
                if scaling == "strong":
                    speedup = ref["mean_step_s"] / s["mean_step_s"]
                    row["speedup"] = speedup
                    row["efficiency"] = speedup / n
                else:
                    row["efficiency"] = s["aggregate_real_tokens_per_s"] / (n * ref["aggregate_real_tokens_per_s"])
            rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = ["strategy", "scaling", "world_size", "global_batch", "aggregate_tok_s", "step_ms", "peak_alloc_gib_max",
            "speedup", "efficiency", "exposed_comm_s", "comm_kernel_s_per_step"]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3g}" if isinstance(v, float) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="reports/L05_training_efficiency/results")
    a = p.parse_args(argv)
    rows = scaling_table(load_summaries(Path(a.runs_root)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(rows, out / "scaling_table.json")
    (out / "scaling_table.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    series: dict[str, list[float]] = {}
    xs: list[int] = []
    for r in rows:
        if "efficiency" in r:
            key = f"{r['strategy']}/{r['scaling']}"
            series.setdefault(key, []).append(r["efficiency"])
            if r["world_size"] not in xs:
                xs.append(r["world_size"])
    if series and all(len(v) == len(xs) for v in series.values()):
        plot_lines(xs, series, xlabel="GPUs", ylabel="scaling efficiency", title="Scaling efficiency", path=out / "scaling_efficiency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
