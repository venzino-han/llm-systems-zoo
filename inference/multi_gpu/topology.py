"""Print and interpret the GPU interconnect topology for multi-GPU inference decisions.

Tensor parallelism (TP) all-reduces the activations of *every* layer across the TP group,
so the link between the GPUs bounds the achievable per-token latency. Replicas (data
parallelism) exchange nothing, so they are indifferent to the link, but each replica must
hold a full copy of the weights and gets 1/N of the KV-cache memory.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from llm_systems_zoo.hardware import gpu_inventory, gpu_topology

# Nominal bidirectional bandwidths (GB/s) for each nvidia-smi link class. These are the
# vendor numbers for the *best* case; measure with comm_bench.py before trusting them.
LINK_CLASSES: dict[str, tuple[str, float]] = {
    "NV": ("NVLink", 600.0),            # per NVLink generation this varies (NV1..NV18); 600 is A100-class
    "PIX": ("same PCIe switch", 64.0),  # PCIe 4.0 x16 ≈ 32 GB/s each direction
    "PXB": ("PCIe across switches", 64.0),
    "PHB": ("PCIe via host bridge (CPU)", 64.0),
    "NODE": ("PCIe + NUMA node hop", 48.0),
    "SYS": ("PCIe + inter-socket (QPI/UPI)", 32.0),
}


def classify(link: str) -> tuple[str, float]:
    for prefix, (name, bw) in LINK_CLASSES.items():
        if link.startswith(prefix):
            return name, bw
    return ("unknown", 0.0)


def summarize_topology(links: dict[str, str], gpus: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Interpret parsed `nvidia-smi topo -m` links."""
    pairs = []
    for key, link in sorted(links.items()):
        a, b = key.split("-")
        if a < b:  # one row per unordered pair
            name, bw = classify(link)
            pairs.append({"pair": f"{a}-{b}", "link": link, "class": name, "nominal_gbps_bidir": bw})
    has_nvlink = any(p["link"].startswith("NV") for p in pairs)
    hetero = False
    if gpus:
        names = {g["name"] for g in gpus}
        hetero = len(names) > 1
    worst = min((p["nominal_gbps_bidir"] for p in pairs), default=0.0)
    advice = []
    if not pairs:
        advice.append("single GPU: TP/replica comparisons are not possible on this host")
    else:
        if has_nvlink:
            advice.append("NVLink present: TP across these GPUs pays little per-layer all-reduce latency; TP is usually the right default for a model that fits")
        else:
            advice.append("no NVLink: every TP all-reduce crosses PCIe/host; expect TP to lose to replicas on throughput for small models and to win only when the model does not fit on one GPU or at low concurrency where latency dominates")
        if hetero:
            advice.append("heterogeneous GPUs: TP runs at the pace of the slowest device and splits memory evenly, so the small GPU bounds both; replica results must be reported per device, not averaged")
    return {"pairs": pairs, "has_nvlink": has_nvlink, "heterogeneous": hetero, "worst_link_nominal_gbps": worst, "advice": advice}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    gpus = gpu_inventory()
    topo = gpu_topology()
    summary = summarize_topology(topo.get("links", {}), gpus)
    summary["gpus"] = gpus
    if a.json:
        print(json.dumps(summary, indent=2))
        return summary
    print("GPUs:")
    for g in gpus:
        print(f"  [{g['index']}] {g['name']}  {g['memory_total_gib']} GiB  sm_{g['compute_capability'].replace('.', '')}  bus {g['pci_bus_id']}")
    print("Links:")
    for pr in summary["pairs"]:
        print(f"  {pr['pair']}: {pr['link']:5s} {pr['class']} (nominal ≤ {pr['nominal_gbps_bidir']:.0f} GB/s bidir)")
    print("Reading:")
    for line in summary["advice"]:
        print(f"  - {line}")
    if topo.get("raw"):
        print("\nnvidia-smi topo -m:\n" + topo["raw"])
    return summary


if __name__ == "__main__":
    main()
