"""Kernel lab tests: the CPU-checkable reference math, plus Triton kernels on GPU (--gpu)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inference.kernels.attention_bench import naive_attention  # noqa: E402


def test_naive_attention_matches_sdpa_math_on_cpu():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(naive_attention(q, k, v), ref, atol=1e-5)


@pytest.mark.gpu
def test_triton_softmax_correct():
    from inference.kernels.triton_softmax import check_correctness

    res = check_correctness()
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]


@pytest.mark.gpu
def test_triton_rmsnorm_correct():
    from inference.kernels.triton_rmsnorm import check_correctness

    res = check_correctness()
    assert all(r["ok"] and r["fused_ok"] for r in res), [r for r in res if not (r["ok"] and r["fused_ok"])]
