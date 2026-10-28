"""CPU tests for the torch.compile helpers (dynamo only, no Inductor / C++ compiler needed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inference.compilation.bench_torch_compile import dynamo_stats, recompiles_between  # noqa: E402


def test_recompile_counter_detects_shape_driven_recompiles():
    import torch._dynamo

    torch._dynamo.reset()

    def f(x):
        return (x * 2).sum()

    g = torch.compile(f, backend="eager", dynamic=False)
    before = dynamo_stats()
    g(torch.ones(4))
    mid = dynamo_stats()
    g(torch.ones(4))  # same shape: no recompile
    same = dynamo_stats()
    g(torch.ones(7))  # new static shape: recompile
    after = dynamo_stats()
    assert mid["unique_graphs"] - before["unique_graphs"] == 1
    assert recompiles_between(mid, same) == 0
    assert recompiles_between(same, after) == 1
    torch._dynamo.reset()
