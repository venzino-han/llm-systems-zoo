"""Hardware and software inventory for run manifests.

Every benchmark report must state the hardware it ran on. This module collects that
automatically so a run can never be published without its environment.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from typing import Any


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gib() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kib = int(line.split()[1])
                    return round(kib / 2**20, 1)
    except OSError:
        pass
    return None


def gpu_inventory() -> list[dict[str, Any]]:
    """GPU list from nvidia-smi (works without torch). Empty list on CPU-only hosts."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict[str, Any]] = []
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_total_gib": round(int(parts[2]) / 1024, 1),
            "driver_version": parts[3],
            "pci_bus_id": parts[4],
            "compute_capability": parts[5],
        })
    return gpus


def gpu_topology() -> dict[str, Any]:
    """Parsed `nvidia-smi topo -m`: link type between each GPU pair (NV#, PIX, PHB, SYS...)."""
    out = _run(["nvidia-smi", "topo", "-m"])
    if not out:
        return {}
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)  # nvidia-smi underlines the header with ANSI codes
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # The matrix is tab separated; column names such as "CPU Affinity" contain spaces.
    header = [h.strip() for h in lines[0].split("\t") if h.strip()]
    gpu_cols = [h for h in header if re.fullmatch(r"GPU\d+", h)]
    links: dict[str, str] = {}
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("\t")]
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        src = cells[0]
        for col, val in zip(gpu_cols, cells[1 : 1 + len(gpu_cols)]):
            if src != col:
                links[f"{src}-{col}"] = val
    has_nvlink = any(v.startswith("NV") for v in links.values())
    return {"links": links, "has_nvlink": has_nvlink, "raw": out}


def cuda_driver_version() -> str | None:
    out = _run(["nvidia-smi"])
    if not out:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


TRACKED_PACKAGES = (
    "torch", "transformers", "vllm", "sglang", "trl", "peft", "accelerate", "datasets",
    "bitsandbytes", "ray", "triton", "flashinfer-python", "flash-attn", "xformers",
    "sentence-transformers", "faiss-cpu", "numpy",
)


def software_inventory() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_driver_api": cuda_driver_version(),
        "packages": {p: v for p in TRACKED_PACKAGES if (v := _pkg_version(p))},
        "env": {
            k: os.environ[k]
            for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NCCL_DEBUG", "VLLM_ATTENTION_BACKEND")
            if k in os.environ
        },
    }
    try:  # torch is optional for the harness
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        pass
    return info


def hardware_inventory() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "gpus": gpu_inventory(),
        "gpu_topology": {k: v for k, v in gpu_topology().items() if k != "raw"},
        "num_nodes": 1,
    }


if __name__ == "__main__":
    import json

    print(json.dumps({"hardware": hardware_inventory(), "software": software_inventory()}, indent=2))
