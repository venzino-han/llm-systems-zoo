"""Aggregate every run directory under runs/ into one table.

Usage:
    python benchmarks/analysis/aggregate_runs.py [--root runs] [--lab L03] [--json out.json] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten(d: dict[str, Any], prefix: str = "", max_depth: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and max_depth > 0:
            out.update(_flatten(v, f"{key}.", max_depth - 1))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def collect(root: Path, lab: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if lab and not m.get("lab", "").startswith(lab):
            continue
        row = {
            "lab": m.get("lab"),
            "name": m.get("name"),
            "status": m.get("status"),
            "created_at": m.get("created_at"),
            "duration_s": m.get("duration_s"),
            "config_hash": m.get("config_hash"),
            "commit": (m.get("provenance") or {}).get("commit", "")[:8] if m.get("provenance") else "",
            "gpus": ",".join(g["name"] for g in (m.get("hardware") or {}).get("gpus", [])),
            "dir": str(manifest_path.parent),
        }
        summary_path = manifest_path.parent / "summary.json"
        if summary_path.exists():
            try:
                row.update({f"summary.{k}": v for k, v in _flatten(json.loads(summary_path.read_text())).items()})
            except json.JSONDecodeError:
                pass
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--lab", default=None, help="prefix filter on the lab name, e.g. L03")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--wide", action="store_true", help="include flattened summary columns")
    args = ap.parse_args()

    rows = collect(Path(args.root), args.lab)
    if not rows:
        print(f"no runs under {args.root}")
        return
    df = pd.DataFrame(rows)
    base_cols = ["lab", "name", "status", "created_at", "duration_s", "config_hash", "gpus", "dir"]
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200):
        print(df if args.wide else df[[c for c in base_cols if c in df.columns]].to_string(index=False))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
