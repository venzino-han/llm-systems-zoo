"""L07 Ray Core: pure helpers plus a few tests on a tiny local Ray cluster (CPU, no network)."""

from __future__ import annotations

import time

import pytest

ray = pytest.importorskip("ray")

from distributed.ray_core._common import busy_wait  # noqa: E402
from distributed.ray_core.failure_injection import analyse, run_policy  # noqa: E402
from distributed.ray_core.object_store import bounded_map  # noqa: E402
from distributed.ray_core.placement import render_bundles  # noqa: E402
from distributed.ray_core.task_overhead import (  # noqa: E402
    n_units_for,
    run_ray_chunked,
    run_ray_tasks,
    run_sequential,
)


@pytest.fixture(scope="module")
def ray_local():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_busy_wait_runs_for_requested_duration():
    t0 = time.perf_counter()
    busy_wait(0.02)
    assert time.perf_counter() - t0 >= 0.02


def test_n_units_for_is_bounded():
    assert n_units_for(1e-6, 4.0, 4000) == 4000
    assert n_units_for(10.0, 4.0, 4000) == 8


def test_render_bundles():
    spec = render_bundles(3, 2, 0.5, "PACK")
    assert len(spec["bundles"]) == 3 and spec["bundles"][0] == {"CPU": 2, "GPU": 0.5}
    assert spec["total"] == {"CPU": 6, "GPU": 1.5} and spec["strategy"] == "PACK"
    assert "GPU" not in render_bundles(1, 1, 0, "SPREAD")["bundles"][0]


def test_analyse_detects_loss_duplicates_and_wrong_values():
    items = [0, 1, 2, 3]
    assert analyse(items, [(0, 0), (1, 2), (2, 4), (3, 6)], 0)["exactly_once"]
    r = analyse(items, [(0, 0), (1, 2), (1, 2), (3, 6)], 1)
    assert r["n_missing"] == 1 and r["n_duplicates"] == 1 and r["n_failed_refs"] == 1 and not r["exactly_once"]
    assert analyse(items, [(0, 0), (1, 2), (2, 5), (3, 6)], 0)["n_wrong_value"] == 1


def test_bounded_map_returns_every_result(ray_local):
    @ray.remote
    def sq(i):
        return i * i

    out = bounded_map(lambda i: sq.remote(i), range(20), max_in_flight=3, consume=lambda v: v + 0)
    assert sorted(out) == [i * i for i in range(20)]


def test_task_overhead_backends_agree_on_small_sizes(ray_local):
    assert run_sequential(1e-4, 4) > 0
    assert run_ray_tasks(1e-4, 4) > 0
    assert run_ray_chunked(1e-4, 5, chunk=2) > 0


@pytest.mark.timeout(120)
def test_naive_policy_loses_results_and_exactly_once_does_not(ray_local):
    items = list(range(12))
    naive = run_policy("naive", items, n_workers=2, crash_items=[3], flaky_items=[5], work_s=0.005)
    assert naive["n_crashes"] == 1 and naive["n_missing"] > 0 and not naive["exactly_once"]
    eo = run_policy("exactly_once", items, n_workers=2, crash_items=[3], flaky_items=[5], work_s=0.005)
    assert eo["exactly_once"] and eo["n_missing"] == 0 and eo["n_duplicates"] == 0 and eo["n_restarts"] >= 1
