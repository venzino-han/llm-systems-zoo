"""Workload generation for the serving labs (L03, L04).

A *workload* is (a) a set of requests, each with a prompt and an output length budget, and
(b) an *arrival process* that says when those requests are submitted.

Length distributions: `fixed`, `uniform`, `lognormal`, `dataset` (prompt lengths taken from
real chat prompts). Arrival processes: `closed` (fixed concurrency; each client waits for
its reply), `open` (Poisson arrivals at a fixed rate, independent of the server), `burst`
(alternating on/off windows with different rates).

Everything is seeded so that a workload YAML + seed reproduces the exact same request list.
Specs are stored in `benchmarks/workloads/*.yaml`.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# A small vocabulary of mostly single-token English words. Used when no tokenizer is given:
# each word is ~1 token for BPE tokenizers, so N words ≈ N tokens (verified against Qwen2.5
# tokenizer: 1000 words -> 1000±20 tokens).
_WORDS = (
    "time year people way day man thing woman life child world school state family student "
    "group country problem hand part place case week company system program question work "
    "government number night point home water room mother area money story fact month lot "
    "right study book eye job word business issue side kind head house service friend father "
    "power hour game line end member law car city community name president team minute idea "
    "kid body information back parent face others level office door health person art war "
    "history party result change morning reason research girl guy moment air teacher force "
    "education foot boy age policy process music market sense nation plan college interest "
    "death experience effect use class control care field development role effort rate heart "
    "drug show leader light voice wife police mind price report decision son view relationship "
    "town road arm difference value building action model season society tax director position "
    "player record paper space ground form event official matter center couple site project "
    "activity star table need court oil situation cost industry figure street image phone data "
    "picture practice piece land product doctor wall patient worker news test movie north love "
    "support technology step baby computer type attention film tree source language region"
).split()


# --------------------------------------------------------------------------------------
# Length distributions
# --------------------------------------------------------------------------------------
@dataclass
class LengthDist:
    """Distribution over token counts.

    kind=fixed      -> `value`
    kind=uniform    -> integer in [low, high]
    kind=lognormal  -> exp(N(mu, sigma)) clipped to [low, high]   (mu/sigma in log space)
    kind=dataset    -> sampled from real prompt lengths (filled in by `DatasetPrompts`)
    """

    kind: str = "fixed"
    value: int = 128
    low: int = 1
    high: int = 4096
    mu: float = 5.0
    sigma: float = 0.5

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return int(self.value)
        if self.kind == "uniform":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "lognormal":
            x = int(round(math.exp(rng.gauss(self.mu, self.sigma))))
            return max(int(self.low), min(int(self.high), x))
        if self.kind == "dataset":
            # Dataset-backed classes draw lengths from the prompts themselves; the sampler
            # returns the target cap and `WorkloadGenerator` substitutes the real length.
            return int(self.high)
        raise ValueError(f"unknown length distribution kind: {self.kind}")

    def mean(self) -> float:
        if self.kind == "fixed":
            return float(self.value)
        if self.kind == "uniform":
            return (self.low + self.high) / 2
        if self.kind == "lognormal":
            return math.exp(self.mu + self.sigma**2 / 2)
        return float(self.high)


@dataclass
class RequestClass:
    """A class of requests (e.g. 'short_qa', 'long_doc') with its own length distributions.

    `shared_prefix_tokens` prepends the same text to every request of the class, which is
    how we model a shared system prompt / document prefix for prefix-caching experiments.
    """

    name: str
    weight: float = 1.0
    prompt_len: LengthDist = field(default_factory=lambda: LengthDist("fixed", 128))
    output_len: LengthDist = field(default_factory=lambda: LengthDist("fixed", 128))
    shared_prefix_tokens: int = 0
    prompt_source: str = "synthetic"  # synthetic | dataset
    dataset: str | None = None
    dataset_split: str | None = None


@dataclass
class ArrivalSpec:
    """How requests are submitted.

    mode=closed : `concurrency` clients, each sends its next request when the previous completes.
    mode=open   : Poisson arrivals at `rate` req/s for `duration_s` seconds (server state ignored).
    mode=burst  : alternate `on_s` seconds at `on_rate` and `off_s` seconds at `off_rate`.
    """

    mode: str = "closed"
    concurrency: int = 8
    rate: float = 4.0
    duration_s: float = 30.0
    on_s: float = 5.0
    off_s: float = 5.0
    on_rate: float = 16.0
    off_rate: float = 1.0

    def offered_load(self) -> float:
        """A single number describing the load for plots: concurrency or mean arrival rate."""
        if self.mode == "closed":
            return float(self.concurrency)
        if self.mode == "open":
            return float(self.rate)
        period = self.on_s + self.off_s
        return (self.on_rate * self.on_s + self.off_rate * self.off_s) / period


@dataclass
class WorkloadSpec:
    name: str
    seed: int = 0
    classes: list[RequestClass] = field(default_factory=list)
    arrival: ArrivalSpec = field(default_factory=ArrivalSpec)
    num_requests: int = 100
    description: str = ""

    # ---- (de)serialisation ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkloadSpec:
        classes = []
        for c in d.get("classes", []):
            c = dict(c)
            c["prompt_len"] = LengthDist(**c.get("prompt_len", {}))
            c["output_len"] = LengthDist(**c.get("output_len", {}))
            classes.append(RequestClass(**c))
        arrival = ArrivalSpec(**d.get("arrival", {}))
        return cls(
            name=d["name"],
            seed=int(d.get("seed", 0)),
            classes=classes,
            arrival=arrival,
            num_requests=int(d.get("num_requests", 100)),
            description=d.get("description", ""),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> WorkloadSpec:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def mean_prompt_tokens(self) -> float:
        w = sum(c.weight for c in self.classes) or 1.0
        return sum(c.weight * (c.prompt_len.mean() + c.shared_prefix_tokens) for c in self.classes) / w

    def mean_output_tokens(self) -> float:
        w = sum(c.weight for c in self.classes) or 1.0
        return sum(c.weight * c.output_len.mean() for c in self.classes) / w


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------
@dataclass
class GeneratedRequest:
    request_id: str
    request_class: str
    prompt: str
    prompt_tokens_target: int
    max_tokens: int
    arrival_offset_s: float  # seconds after the start of the run (0 for closed loop)
    shared_prefix_tokens: int = 0


def synthetic_text(n_tokens: int, rng: random.Random) -> str:
    """~n_tokens of pseudo-English. With a BPE tokenizer this lands within a few % of n."""
    if n_tokens <= 0:
        return ""
    return " ".join(rng.choice(_WORDS) for _ in range(n_tokens))


class TokenBudget:
    """Fit synthetic text to an exact token count when a tokenizer is available."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def text(self, n_tokens: int, rng: random.Random) -> str:
        if self.tokenizer is None or n_tokens <= 0:
            return synthetic_text(n_tokens, rng)
        # Decode random vocabulary ids (the approach vLLM's random dataset uses), then trim
        # until re-tokenisation gives at most n tokens. Avoid special tokens.
        vocab = self.tokenizer.vocab_size
        ids = [rng.randrange(0, vocab) for _ in range(n_tokens)]
        special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        ids = [i for i in ids if i not in special]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        for _ in range(8):
            n = self.count(text)
            if n <= n_tokens:
                break
            text = self.tokenizer.decode(self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens])
        return text


# --------------------------------------------------------------------------------------
# Arrival schedules
# --------------------------------------------------------------------------------------
def poisson_arrivals(n: int, rate: float, rng: random.Random) -> list[float]:
    """n arrival offsets with exponential inter-arrival times of mean 1/rate."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    t = 0.0
    out = []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def burst_arrivals(n: int, on_s: float, off_s: float, on_rate: float, off_rate: float, rng: random.Random) -> list[float]:
    """Poisson arrivals whose rate alternates between on_rate (for on_s) and off_rate (for off_s)."""
    out: list[float] = []
    t = 0.0
    while len(out) < n:
        for window, rate in ((on_s, on_rate), (off_s, off_rate)):
            end = t + window
            while len(out) < n:
                if rate <= 0:
                    t = end
                    break
                t += rng.expovariate(rate)
                if t > end:
                    t = end
                    break
                out.append(t)
            if len(out) >= n:
                break
    return out


def arrival_offsets(spec: ArrivalSpec, n: int, rng: random.Random) -> list[float]:
    if spec.mode == "closed":
        return [0.0] * n
    if spec.mode == "open":
        return poisson_arrivals(n, spec.rate, rng)
    if spec.mode == "burst":
        return burst_arrivals(n, spec.on_s, spec.off_s, spec.on_rate, spec.off_rate, rng)
    raise ValueError(f"unknown arrival mode: {spec.mode}")


# --------------------------------------------------------------------------------------
# Dataset prompts
# --------------------------------------------------------------------------------------
class DatasetPrompts:
    """Real user prompts from a Hugging Face chat dataset (default: ultrachat_200k test_sft).

    Prompts are filtered to [min_tokens, max_tokens] and optionally truncated to
    `truncate_to` tokens so that a class can target a length band while keeping real text.
    Needs `datasets` and a tokenizer; only used when a class has prompt_source=dataset.
    """

    def __init__(
        self,
        dataset: str = "HuggingFaceH4/ultrachat_200k",
        split: str = "test_sft",
        tokenizer: Any | None = None,
        min_tokens: int = 1,
        max_tokens: int = 4096,
        truncate_to: int | None = None,
        limit: int = 2000,
    ) -> None:
        from datasets import load_dataset  # lazy: network/heavy

        ds = load_dataset(dataset, split=split, streaming=True)
        self.prompts: list[tuple[str, int]] = []
        for row in ds:
            text = _first_user_turn(row)
            if not text:
                continue
            n = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else len(text.split())
            if truncate_to and n > truncate_to:
                if tokenizer:
                    text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:truncate_to])
                else:
                    text = " ".join(text.split()[:truncate_to])
                n = truncate_to
            if min_tokens <= n <= max_tokens:
                self.prompts.append((text, n))
            if len(self.prompts) >= limit:
                break
        if not self.prompts:
            raise RuntimeError("no prompts matched the length filter")

    def sample(self, rng: random.Random) -> tuple[str, int]:
        return rng.choice(self.prompts)


def _first_user_turn(row: dict[str, Any]) -> str | None:
    msgs = row.get("messages") or []
    for m in msgs:
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return row.get("prompt")


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------
class WorkloadGenerator:
    """Turns a WorkloadSpec into a deterministic list of GeneratedRequest."""

    def __init__(
        self,
        spec: WorkloadSpec,
        tokenizer: Any | None = None,
        dataset_prompts: dict[str, DatasetPrompts] | None = None,
    ) -> None:
        self.spec = spec
        self.budget = TokenBudget(tokenizer)
        self.dataset_prompts = dataset_prompts or {}
        self._prefix_cache: dict[str, str] = {}

    def _prefix(self, cls: RequestClass, rng: random.Random) -> str:
        if cls.shared_prefix_tokens <= 0:
            return ""
        if cls.name not in self._prefix_cache:
            # A dedicated RNG keyed on the class keeps the prefix identical across runs.
            prng = random.Random(f"{self.spec.seed}:{cls.name}:prefix")
            self._prefix_cache[cls.name] = self.budget.text(cls.shared_prefix_tokens, prng)
        return self._prefix_cache[cls.name]

    def requests(self, n: int | None = None) -> list[GeneratedRequest]:
        n = n or self.spec.num_requests
        rng = random.Random(self.spec.seed)
        if not self.spec.classes:
            raise ValueError("workload has no request classes")
        weights = [c.weight for c in self.spec.classes]
        offsets = arrival_offsets(self.spec.arrival, n, random.Random(self.spec.seed + 1))
        out: list[GeneratedRequest] = []
        for i in range(n):
            cls = rng.choices(self.spec.classes, weights=weights, k=1)[0]
            prefix = self._prefix(cls, rng)
            if cls.prompt_source == "dataset":
                dp = self.dataset_prompts.get(cls.name)
                if dp is None:
                    raise ValueError(f"class {cls.name} needs DatasetPrompts (prompt_source=dataset)")
                body, n_body = dp.sample(rng)
            else:
                n_body = cls.prompt_len.sample(rng)
                body = self.budget.text(n_body, rng)
            prompt = (prefix + "\n\n" + body) if prefix else body
            out.append(
                GeneratedRequest(
                    request_id=f"{self.spec.name}-{i:05d}",
                    request_class=cls.name,
                    prompt=prompt,
                    prompt_tokens_target=n_body + cls.shared_prefix_tokens,
                    max_tokens=max(1, cls.output_len.sample(rng)),
                    arrival_offset_s=offsets[i],
                    shared_prefix_tokens=cls.shared_prefix_tokens,
                )
            )
        return out


def describe(spec: WorkloadSpec) -> dict[str, Any]:
    """Compact description for run manifests."""
    return {
        "name": spec.name,
        "seed": spec.seed,
        "num_requests": spec.num_requests,
        "arrival": asdict(spec.arrival),
        "offered_load": spec.arrival.offered_load(),
        "mean_prompt_tokens": spec.mean_prompt_tokens(),
        "mean_output_tokens": spec.mean_output_tokens(),
        "classes": [
            {
                "name": c.name,
                "weight": c.weight,
                "prompt_len": asdict(c.prompt_len),
                "output_len": asdict(c.output_len),
                "shared_prefix_tokens": c.shared_prefix_tokens,
                "prompt_source": c.prompt_source,
            }
            for c in spec.classes
        ],
    }


# --------------------------------------------------------------------------------------
# Built-in specs (written to benchmarks/workloads/ by `python -m serving.workloads --write`)
# --------------------------------------------------------------------------------------
def builtin_specs() -> dict[str, WorkloadSpec]:
    short_qa = WorkloadSpec(
        name="short_qa",
        description="Chat-style short questions, short answers. Decode-dominated.",
        seed=0,
        num_requests=200,
        classes=[
            RequestClass(
                "short_qa",
                prompt_len=LengthDist("lognormal", mu=math.log(128), sigma=0.4, low=16, high=512),
                output_len=LengthDist("lognormal", mu=math.log(128), sigma=0.3, low=16, high=256),
            )
        ],
        arrival=ArrivalSpec(mode="open", rate=4.0, duration_s=60),
    )
    long_prefix = WorkloadSpec(
        name="long_prefix_shared",
        description="A 1024-token shared system prompt + short user turn. Prefill-dominated unless prefix caching is on.",
        seed=0,
        num_requests=200,
        classes=[
            RequestClass(
                "shared_prefix",
                prompt_len=LengthDist("uniform", low=32, high=128),
                output_len=LengthDist("fixed", value=64),
                shared_prefix_tokens=1024,
            )
        ],
        arrival=ArrivalSpec(mode="open", rate=4.0, duration_s=60),
    )
    mixed_burst = WorkloadSpec(
        name="mixed_burst",
        description="80% short chat + 20% long-document requests under a bursty arrival process. Shows head-of-line blocking.",
        seed=0,
        num_requests=200,
        classes=[
            RequestClass(
                "short",
                weight=0.8,
                prompt_len=LengthDist("lognormal", mu=math.log(96), sigma=0.4, low=16, high=256),
                output_len=LengthDist("lognormal", mu=math.log(96), sigma=0.3, low=16, high=192),
            ),
            RequestClass(
                "long",
                weight=0.2,
                prompt_len=LengthDist("uniform", low=1500, high=3000),
                output_len=LengthDist("fixed", value=256),
            ),
        ],
        arrival=ArrivalSpec(mode="burst", on_s=5, off_s=10, on_rate=12.0, off_rate=1.0),
    )
    return {s.name: s for s in (short_qa, long_prefix, mixed_burst)}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Inspect or write workload specs")
    p.add_argument("--write", type=Path, default=None, help="write built-in specs to this directory")
    p.add_argument("--spec", type=Path, default=None, help="load a spec and print a sample of requests")
    p.add_argument("-n", type=int, default=5)
    a = p.parse_args(argv)
    if a.write:
        for name, spec in builtin_specs().items():
            print("wrote", spec.save(a.write / f"{name}.yaml"))
    if a.spec:
        spec = WorkloadSpec.load(a.spec)
        print(yaml.safe_dump(describe(spec), sort_keys=False))
        for r in WorkloadGenerator(spec).requests(a.n):
            print(f"{r.request_id} class={r.request_class} prompt~{r.prompt_tokens_target} tok "
                  f"max_tokens={r.max_tokens} t={r.arrival_offset_s:.2f}s")


def iter_classes(spec: WorkloadSpec) -> Iterable[RequestClass]:
    return iter(spec.classes)


if __name__ == "__main__":
    main()
