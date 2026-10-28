"""Small, dependency-free file helpers shared by every lab."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False, default=_json_default)
        f.write("\n")
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict], path: str | Path, append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    return path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_of(obj: Any, length: int = 12) -> str:
    """Stable short hash of any JSON-serialisable object (used for config hashes)."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):  # numpy scalars / arrays
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
