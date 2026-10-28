"""L02 — drive the allocators with synthetic workloads and measure waste, sharing and capacity.

A discrete-time simulation: every step each active sequence decodes one token; new requests are
admitted when the allocator can hold their prompt; finished sequences free their memory. Time
is measured in decode steps, so "makespan" is the number of steps to drain the request queue —
a proxy for throughput at a fixed per-step cost. Nothing here is a latency measurement; the
vLLM benchmark in `bench_prefix_caching.py` provides the real-engine numbers.

Experiments (each writes a JSON table and a figure under --out):

    waste        internal fragmentation and makespan vs block size, paged vs contiguous
    prefix       prefix-cache hit rate vs fraction of requests sharing a system prompt
    capacity     achieved concurrency and makespan vs KV memory budget, paged vs contiguous
    eviction     hit rate vs number of distinct system prompts under a fixed cache budget
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from inference.kv_cache.paged_allocator import ContiguousAllocator, OutOfMemory, PagedAllocator
from llm_systems_zoo.plotting import plot_lines


@dataclass
class Request:
    rid: int
    prompt: list[int]
    gen_len: int
    generated: int = 0
    start_step: int | None = None
    end_step: int | None = None
    preemptions: int = 0


@dataclass
class WorkloadSpec:
    n_requests: int = 300
    prompt_len_mean: int = 512
    prompt_len_sigma: float = 0.5      # lognormal spread in log space
    prompt_len_max: int = 4096
    gen_len_mean: int = 128
    gen_len_max: int = 512
    shared_prefix_len: int = 256
    shared_prefix_ratio: float = 0.5   # fraction of requests that start with a shared system prompt
    n_prefix_variants: int = 1         # distinct system prompts (cycled round-robin)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def generate_requests(spec: WorkloadSpec) -> list[Request]:
    rng = np.random.default_rng(spec.seed)
    prefixes = [[10_000_000 + v * 100_000 + i for i in range(spec.shared_prefix_len)] for v in range(spec.n_prefix_variants)]
    reqs = []
    for i in range(spec.n_requests):
        body_len = int(np.clip(rng.lognormal(np.log(spec.prompt_len_mean), spec.prompt_len_sigma), 8, spec.prompt_len_max))
        gen_len = int(np.clip(rng.exponential(spec.gen_len_mean), 1, spec.gen_len_max))
        body = rng.integers(0, 1_000_000, size=body_len).tolist()
        use_prefix = spec.shared_prefix_len > 0 and rng.random() < spec.shared_prefix_ratio
        prompt = (prefixes[i % spec.n_prefix_variants] + body) if use_prefix else body
        reqs.append(Request(i, prompt[: spec.prompt_len_max], gen_len))
    return reqs


@dataclass
class SimResult:
    makespan_steps: int
    tokens_generated: int          # useful tokens (recomputed tokens after preemption excluded)
    recomputed_tokens: int
    tokens_per_step: float
    mean_active: float
    peak_active: int
    mean_utilisation: float
    mean_internal_fragmentation: float
    preemptions: int
    dropped: int
    allocator_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def simulate(allocator: ContiguousAllocator | PagedAllocator, requests: list[Request], max_steps: int = 1_000_000,
             admission_reserve: float = 0.05) -> SimResult:
    """`admission_reserve` keeps that fraction of blocks free for decode growth (vLLM's watermark),
    which limits preemption thrash; the contiguous allocator reserves full sequences already."""
    pending: deque[Request] = deque(requests)
    reserve_blocks = int(admission_reserve * allocator.num_blocks) if isinstance(allocator, PagedAllocator) else 0
    active: dict[int, Request] = {}
    step = 0
    tokens = 0
    recomputed = 0
    preemptions = dropped = 0
    active_hist: list[int] = []
    util_hist: list[float] = []
    frag_hist: list[float] = []
    while (pending or active) and step < max_steps:
        # admission: FCFS, stop at the first request that does not fit (no reordering)
        while pending:
            r = pending[0]
            if isinstance(allocator, PagedAllocator) and active and \
                    allocator.blocks_needed(len(r.prompt)) + reserve_blocks > allocator.num_available_blocks:
                break
            try:
                allocator.allocate(r.rid, r.prompt)
            except OutOfMemory:
                if not active:  # cannot ever fit → drop (an engine would reject the request)
                    pending.popleft()
                    dropped += 1
                    continue
                break
            pending.popleft()
            r.start_step = step
            active[r.rid] = r
        # one decode step for every active sequence
        for r in list(active.values()):
            try:
                allocator.append(r.rid, 5_000_000 + r.rid * 1000 + r.generated)  # unique generated token
            except OutOfMemory:
                # recompute-style preemption: release everything and go back to the queue head
                allocator.free(r.rid)
                del active[r.rid]
                recomputed += r.generated  # those tokens will be generated again
                r.generated = 0
                r.preemptions += 1
                preemptions += 1
                pending.appendleft(r)
                continue
            r.generated += 1
            if r.generated >= r.gen_len:
                tokens += r.gen_len
                allocator.free(r.rid)
                del active[r.rid]
                r.end_step = step
        st = allocator.stats()
        active_hist.append(len(active))
        util_hist.append(st["utilisation"])
        frag_hist.append(st["internal_fragmentation"])
        step += 1
    return SimResult(
        makespan_steps=step,
        tokens_generated=tokens,
        recomputed_tokens=recomputed,
        tokens_per_step=tokens / step if step else 0.0,
        mean_active=float(np.mean(active_hist)) if active_hist else 0.0,
        peak_active=max(active_hist, default=0),
        mean_utilisation=float(np.mean(util_hist)) if util_hist else 0.0,
        mean_internal_fragmentation=float(np.mean(frag_hist)) if frag_hist else 0.0,
        preemptions=preemptions,
        dropped=dropped,
        allocator_stats=allocator.stats(),
    )


def _max_seq_len(reqs: list[Request]) -> int:
    return max(len(r.prompt) + r.gen_len for r in reqs)


def run_paged(spec: WorkloadSpec, capacity_tokens: int, block_size: int, prefix_caching: bool = True) -> SimResult:
    reqs = generate_requests(spec)
    alloc = PagedAllocator(num_blocks=capacity_tokens // block_size, block_size=block_size, enable_prefix_caching=prefix_caching)
    return simulate(alloc, reqs)


def run_contiguous(spec: WorkloadSpec, capacity_tokens: int, max_seq_len: int | None = None) -> SimResult:
    reqs = generate_requests(spec)
    alloc = ContiguousAllocator(capacity_tokens, max_seq_len or _max_seq_len(reqs))
    return simulate(alloc, reqs)


# ---- experiments -----------------------------------------------------------------------------
def exp_waste(spec: WorkloadSpec, capacity: int, block_sizes: list[int], out: Path) -> list[dict]:
    rows = []
    c = run_contiguous(spec, capacity)
    rows.append({"allocator": "contiguous", "block_size": None, **c.to_dict()})
    for bs in block_sizes:
        r = run_paged(spec, capacity, bs)
        rows.append({"allocator": "paged", "block_size": bs, **r.to_dict()})
    paged = [r for r in rows if r["allocator"] == "paged"]
    plot_lines(block_sizes, {"paged": [r["mean_internal_fragmentation"] for r in paged],
                             "contiguous (max_seq_len reservation)": [c.mean_internal_fragmentation] * len(block_sizes)},
               xlabel="block size (tokens)", ylabel="mean internal fragmentation", title="Waste vs block size", logx=True,
               path=out / "waste_vs_block_size.png")
    plot_lines(block_sizes, {"paged": [r["makespan_steps"] for r in paged], "contiguous": [c.makespan_steps] * len(block_sizes)},
               xlabel="block size (tokens)", ylabel="makespan (decode steps)", title="Makespan vs block size", logx=True,
               path=out / "makespan_vs_block_size.png")
    return rows


def exp_prefix(spec: WorkloadSpec, capacity: int, block_size: int, ratios: list[float], prefix_lens: list[int], out: Path) -> list[dict]:
    rows = []
    series: dict[str, list[float]] = {}
    for plen in prefix_lens:
        ys = []
        for ratio in ratios:
            s = WorkloadSpec(**{**spec.to_dict(), "shared_prefix_len": plen, "shared_prefix_ratio": ratio})
            r = run_paged(s, capacity, block_size)
            hit_frac = r.allocator_stats["hit_token_fraction"]
            rows.append({"prefix_len": plen, "shared_ratio": ratio, "hit_token_fraction": hit_frac,
                         "block_hit_rate": r.allocator_stats["cache_hit_rate"], **r.to_dict()})
            ys.append(hit_frac)
        series[f"prefix {plen} tok"] = ys
    plot_lines(ratios, series, xlabel="fraction of requests sharing the system prompt", ylabel="prompt tokens served from cache",
               title=f"Prefix-cache hit fraction (block {block_size})", path=out / "prefix_hit_vs_shared_ratio.png")
    return rows


def exp_capacity(spec: WorkloadSpec, capacities: list[int], block_size: int, out: Path) -> list[dict]:
    rows = []
    for cap in capacities:
        c = run_contiguous(spec, cap)
        p = run_paged(spec, cap, block_size)
        rows.append({"capacity_tokens": cap, "allocator": "contiguous", **c.to_dict()})
        rows.append({"capacity_tokens": cap, "allocator": "paged", "block_size": block_size, **p.to_dict()})
    cont = [r for r in rows if r["allocator"] == "contiguous"]
    pag = [r for r in rows if r["allocator"] == "paged"]
    plot_lines(capacities, {"contiguous": [r["mean_active"] for r in cont], "paged": [r["mean_active"] for r in pag]},
               xlabel="KV capacity (tokens)", ylabel="mean concurrent sequences", title="Concurrency vs memory budget", logx=True,
               path=out / "concurrency_vs_capacity.png")
    plot_lines(capacities, {"contiguous": [r["tokens_per_step"] for r in cont], "paged": [r["tokens_per_step"] for r in pag]},
               xlabel="KV capacity (tokens)", ylabel="generated tokens per decode step", title="Throughput proxy vs memory budget", logx=True,
               path=out / "throughput_vs_capacity.png")
    return rows


def exp_eviction(spec: WorkloadSpec, capacity: int, block_size: int, variants: list[int], out: Path) -> list[dict]:
    rows = []
    for nv in variants:
        s = WorkloadSpec(**{**spec.to_dict(), "n_prefix_variants": nv, "shared_prefix_ratio": 1.0})
        r = run_paged(s, capacity, block_size)
        rows.append({"n_prefix_variants": nv, "hit_token_fraction": r.allocator_stats["hit_token_fraction"],
                     "evictions": r.allocator_stats["evictions"], **r.to_dict()})
    plot_lines(variants, {"hit fraction": [r["hit_token_fraction"] for r in rows]}, xlabel="distinct system prompts",
               ylabel="prompt tokens served from cache", title=f"Cache pressure (capacity {capacity} tok)", logx=True,
               path=out / "hit_vs_prefix_variants.png")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KV-cache allocator simulations (CPU).")
    ap.add_argument("--exp", nargs="+", default=["waste", "prefix", "capacity", "eviction"])
    ap.add_argument("--out", type=Path, default=Path("runs/L02_kv_cache/sim"))
    ap.add_argument("--n-requests", type=int, default=300)
    ap.add_argument("--capacity", type=int, default=65_536, help="KV budget in tokens for waste/prefix/eviction")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    spec = WorkloadSpec(n_requests=40 if a.smoke else a.n_requests)
    results: dict[str, Any] = {"spec": spec.to_dict(), "capacity": a.capacity, "block_size": a.block_size}
    if "waste" in a.exp:
        results["waste"] = exp_waste(spec, a.capacity, [1, 4, 8, 16, 32, 64, 128, 256] if not a.smoke else [8, 64], a.out)
    if "prefix" in a.exp:
        results["prefix"] = exp_prefix(spec, a.capacity, a.block_size, [0.0, 0.25, 0.5, 0.75, 1.0], [128, 512, 1024] if not a.smoke else [256], a.out)
    if "capacity" in a.exp:
        caps = [8_192, 16_384, 32_768, 65_536, 131_072] if not a.smoke else [8_192, 32_768]
        results["capacity"] = exp_capacity(spec, caps, a.block_size, a.out)
    if "eviction" in a.exp:
        results["eviction"] = exp_eviction(WorkloadSpec(**{**spec.to_dict(), "shared_prefix_len": 1024}), 16_384, a.block_size,
                                           [1, 2, 4, 8, 16, 32] if not a.smoke else [1, 8], a.out)
    (a.out / "simulation_results.json").write_text(json.dumps(results, indent=2))
    for name in ("waste", "prefix", "capacity", "eviction"):
        if name in results:
            print(f"\n== {name} ==")
            keys = [k for k in results[name][0] if k != "allocator_stats"]
            print(" | ".join(keys))
            for r in results[name]:
                print(" | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
