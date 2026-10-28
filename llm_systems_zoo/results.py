"""Copy the curated part of a run directory into reports/<lab>/results/."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

CURATED_DEFAULT = ("summary.json", "manifest.json", "config.json")


def copy_results(
    run_dir: str | Path,
    reports_dir: str | Path,
    *,
    name: str | None = None,
    files: Iterable[str] = CURATED_DEFAULT,
    figures: bool = True,
) -> Path:
    """Copy summary/manifest/config (and figures) of a run into a reports results folder.

    Raw records and traces stay in `runs/` (git-ignored); only the small curated files move.
    Returns the destination directory.
    """
    run_dir = Path(run_dir)
    dest = Path(reports_dir) / (name or run_dir.name)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = run_dir / f
        if src.exists():
            shutil.copy2(src, dest / f)
    if figures:
        for src in list(run_dir.glob("*.png")) + list(run_dir.glob("*.svg")) + list(run_dir.glob("*.md")):
            shutil.copy2(src, dest / src.name)
    return dest
