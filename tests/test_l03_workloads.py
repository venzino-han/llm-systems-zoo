"""L03: workload generators are deterministic and their distributions behave as specified."""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

import pytest

from serving.workloads import (
    ArrivalSpec,
    LengthDist,
    RequestClass,
    WorkloadGenerator,
    WorkloadSpec,
    arrival_offsets,
    builtin_specs,
    burst_arrivals,
    describe,
    poisson_arrivals,
    synthetic_text,
)

WORKLOAD_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "workloads"


def _spec(**kw) -> WorkloadSpec:
    base = dict(
        name="t",
        seed=7,
        num_requests=50,
        classes=[RequestClass("a", prompt_len=LengthDist("fixed", 32), output_len=LengthDist("fixed", 8))],
        arrival=ArrivalSpec(mode="closed", concurrency=4),
    )
    base.update(kw)
    return WorkloadSpec(**base)


def test_generator_is_deterministic_for_same_seed():
    a = WorkloadGenerator(_spec()).requests()
    b = WorkloadGenerator(_spec()).requests()
    assert [r.prompt for r in a] == [r.prompt for r in b]
    assert [r.arrival_offset_s for r in a] == [r.arrival_offset_s for r in b]
    c = WorkloadGenerator(_spec(seed=8)).requests()
    assert [r.prompt for r in a] != [r.prompt for r in c]


def test_fixed_and_uniform_lengths():
    rng = random.Random(0)
    assert all(LengthDist("fixed", 32).sample(rng) == 32 for _ in range(10))
    xs = [LengthDist("uniform", low=10, high=20).sample(rng) for _ in range(500)]
    assert min(xs) >= 10 and max(xs) <= 20 and len(set(xs)) > 5


def test_lognormal_lengths_clip_and_centre():
    rng = random.Random(0)
    d = LengthDist("lognormal", mu=math.log(128), sigma=0.3, low=16, high=512)
    xs = [d.sample(rng) for _ in range(4000)]
    assert min(xs) >= 16 and max(xs) <= 512
    assert 110 < statistics.median(xs) < 150  # median of lognormal = exp(mu)


def test_synthetic_text_word_count_matches_tokens():
    rng = random.Random(1)
    assert len(synthetic_text(200, rng).split()) == 200
    assert synthetic_text(0, rng) == ""


def test_poisson_inter_arrival_mean():
    rng = random.Random(3)
    rate = 10.0
    t = poisson_arrivals(5000, rate, rng)
    gaps = [b - a for a, b in zip(t, t[1:])]
    assert abs(statistics.mean(gaps) - 1 / rate) < 0.01
    assert all(b > a for a, b in zip(t, t[1:]))


def test_burst_arrivals_respect_windows():
    rng = random.Random(0)
    t = burst_arrivals(400, on_s=2.0, off_s=2.0, on_rate=50.0, off_rate=1.0, rng=rng)
    in_on = sum(1 for x in t if (x % 4.0) < 2.0)
    assert in_on > 0.9 * len(t)  # almost everything lands in the "on" windows
    assert t == sorted(t)


def test_arrival_offsets_by_mode():
    rng = random.Random(0)
    assert arrival_offsets(ArrivalSpec(mode="closed"), 5, rng) == [0.0] * 5
    assert len(arrival_offsets(ArrivalSpec(mode="open", rate=5), 7, rng)) == 7
    with pytest.raises(ValueError):
        arrival_offsets(ArrivalSpec(mode="nope"), 1, rng)


def test_class_mix_and_shared_prefix():
    spec = _spec(
        classes=[
            RequestClass("short", weight=0.8, prompt_len=LengthDist("fixed", 10), output_len=LengthDist("fixed", 4)),
            RequestClass("long", weight=0.2, prompt_len=LengthDist("fixed", 40), output_len=LengthDist("fixed", 4), shared_prefix_tokens=20),
        ],
        num_requests=1000,
    )
    reqs = WorkloadGenerator(spec).requests()
    share = sum(r.request_class == "long" for r in reqs) / len(reqs)
    assert 0.15 < share < 0.25
    longs = [r for r in reqs if r.request_class == "long"]
    prefixes = {r.prompt.split("\n\n")[0] for r in longs}
    assert len(prefixes) == 1  # identical prefix across the class
    assert all(r.prompt_tokens_target == 60 for r in longs)
    assert all(len(r.prompt.split()) == 60 for r in longs)


def test_yaml_round_trip(tmp_path: Path):
    spec = builtin_specs()["mixed_burst"]
    path = spec.save(tmp_path / "w.yaml")
    loaded = WorkloadSpec.load(path)
    assert loaded.to_dict() == spec.to_dict()
    assert describe(loaded)["offered_load"] == pytest.approx(spec.arrival.offered_load())


def test_builtin_specs_in_repo_match_code():
    for name, spec in builtin_specs().items():
        on_disk = WorkloadSpec.load(WORKLOAD_DIR / f"{name}.yaml")
        assert on_disk.to_dict() == spec.to_dict(), f"{name}.yaml is stale: run `python -m serving.workloads --write benchmarks/workloads`"


def test_offered_load_summary():
    assert ArrivalSpec(mode="closed", concurrency=16).offered_load() == 16
    assert ArrivalSpec(mode="open", rate=3.5).offered_load() == 3.5
    assert ArrivalSpec(mode="burst", on_s=5, off_s=5, on_rate=10, off_rate=0).offered_load() == 5
