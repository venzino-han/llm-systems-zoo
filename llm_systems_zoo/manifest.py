"""Run manifests: every experiment writes who/what/where/how before it writes a number.

A `Run` owns one directory under `runs/<lab>/<timestamp>_<name>_<cfghash>/` containing:

    manifest.json   hardware, software, model, workload, execution, provenance, status
    config.json     the exact configuration the script ran with
    metrics.jsonl   time-stamped scalar metrics logged during the run
    records.jsonl   per-request / per-step raw records (optional, written by the lab)
    summary.json    the final aggregate the report quotes

The benchmark protocol (docs/benchmark_protocol.md) explains why each field exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .hardware import hardware_inventory, software_inventory
from .io import dump_json, sha256_of, write_jsonl


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"])) if commit else None
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": dirty,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }


def config_hash(config: dict[str, Any]) -> str:
    return sha256_of(config)


@dataclass
class RunManifest:
    run_id: str
    lab: str
    name: str
    created_at: str
    config: dict[str, Any]
    config_hash: str
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)      # model id, revision, dtype, quant, parallelism
    workload: dict[str, Any] = field(default_factory=dict)   # dataset, seed, lengths, concurrency/arrival
    execution: dict[str, Any] = field(default_factory=dict)  # warm-up, repeats, timeout, cache state
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    finished_at: str | None = None
    duration_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    """Context manager that creates the run directory and manifest.

    >>> with Run("L01_memory", "prefill_vs_decode", config) as run:
    ...     run.log_metric(step=1, tokens_per_s=1234.5)
    ...     run.save_summary({"tokens_per_s": 1234.5})
    """

    def __init__(
        self,
        lab: str,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        root: str | Path = "runs",
        model: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        collect_env: bool = True,
    ) -> None:
        config = dict(config or {})
        self._t0 = time.perf_counter()
        chash = config_hash(config)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / lab / f"{stamp}_{name}_{chash}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.manifest = RunManifest(
            run_id=uuid.uuid4().hex[:12],
            lab=lab,
            name=name,
            created_at=_now_iso(),
            config=config,
            config_hash=chash,
            hardware=hardware_inventory() if collect_env else {},
            software=software_inventory() if collect_env else {},
            model=dict(model or {}),
            workload=dict(workload or {}),
            execution=dict(execution or {}),
            provenance=git_provenance(),
        )
        dump_json(config, self.dir / "config.json")
        self.save_manifest()

    # ---- paths -------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / name

    # ---- logging -----------------------------------------------------------------------
    def log_metric(self, **values: Any) -> None:
        row = {"t": round(time.perf_counter() - self._t0, 6), **values}
        write_jsonl([row], self.dir / "metrics.jsonl", append=True)

    def log_records(self, records: Iterable[Any], filename: str = "records.jsonl", append: bool = False) -> Path:
        rows = [r if isinstance(r, dict) else asdict(r) for r in records]
        return write_jsonl(rows, self.dir / filename, append=append)

    def save_json(self, name: str, obj: Any) -> Path:
        return dump_json(obj, self.dir / name)

    def save_summary(self, summary: dict[str, Any]) -> Path:
        return dump_json(summary, self.dir / "summary.json")

    def note(self, text: str) -> None:
        self.manifest.notes.append(text)
        self.save_manifest()

    def update(self, **sections: dict[str, Any]) -> None:
        """Merge extra model/workload/execution info discovered after start."""
        for key, value in sections.items():
            getattr(self.manifest, key).update(value)
        self.save_manifest()

    def save_manifest(self) -> Path:
        return dump_json(self.manifest.to_dict(), self.dir / "manifest.json")

    # ---- lifecycle ---------------------------------------------------------------------
    def finish(self, status: str = "completed") -> None:
        self.manifest.status = status
        self.manifest.finished_at = _now_iso()
        self.manifest.duration_s = round(time.perf_counter() - self._t0, 3)
        self.save_manifest()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("failed" if exc_type else "completed")
