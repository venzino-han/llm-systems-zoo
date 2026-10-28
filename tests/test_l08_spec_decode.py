"""CPU tests for the minimal speculative decoder (tiny random models, no downloads)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from inference.speculative_decoding.spec_decode_minimal import (  # noqa: E402
    plain_generate,
    speculative_generate,
    verify_sampled,
)


def _tiny(seed: int, vocab: int = 64):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def models():
    return _tiny(0), _tiny(1)


def test_same_model_as_draft_accepts_everything(models):
    target, _ = models
    ids = torch.tensor([[1, 5, 9, 2, 7]])
    out, st = speculative_generate(target, target, ids, max_new_tokens=16, k=4)
    assert out == plain_generate(target, ids, 16)
    assert st.acceptance_rate == 1.0 and st.accepted == st.proposed
    # 1 prefill forward + 4 verification rounds emitting 5, 5, 5, 1 tokens (k accepted + 1 bonus)
    assert st.target_forwards == 5 and st.emitted == len(out) == 16


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_greedy_output_equals_target_greedy_for_any_draft(models, k):
    target, draft = models
    for prompt in ([[3, 4, 5]], [[7]], [[1, 2, 3, 4, 5, 6, 7, 8]]):
        ids = torch.tensor(prompt)
        out, st = speculative_generate(target, draft, ids, max_new_tokens=20, k=k)
        assert out == plain_generate(target, ids, 20), f"k={k} prompt={prompt}"
        assert st.emitted == 20 and st.proposed <= st.draft_forwards
        assert 0.0 <= st.acceptance_rate <= 1.0
        assert len(st.acceptance_rate_per_position) == k


def test_eos_stops_generation(models):
    target, draft = models
    ids = torch.tensor([[3, 4, 5]])
    ref = plain_generate(target, ids, 30)
    eos = ref[5]  # pretend the 6th generated token is EOS
    out, _ = speculative_generate(target, draft, ids, max_new_tokens=30, k=3, eos_token_id=eos)
    assert out == ref[: ref.index(eos) + 1]


def test_sampling_mode_runs_and_has_right_length(models):
    target, draft = models
    gen = torch.Generator().manual_seed(0)
    out, st = speculative_generate(target, draft, torch.tensor([[2, 3]]), max_new_tokens=12, k=3, mode="sample", generator=gen)
    assert len(out) == 12 and st.emitted == 12


def test_rejection_sampling_recovers_target_distribution():
    """Empirical distribution of accepted-or-corrected tokens must match p, not q."""
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])  # deliberately very different draft
    gen = torch.Generator().manual_seed(0)
    n = 40000
    counts = torch.zeros(4)
    accepted = 0
    for _ in range(n):
        tok = int(torch.multinomial(q, 1, generator=gen))
        ok, corrected = verify_sampled(p, q, tok, gen)
        counts[tok if ok else corrected] += 1
        accepted += ok
    emp = counts / n
    assert torch.allclose(emp, p, atol=0.015), emp
    # expected acceptance = sum_x min(p(x), q(x))
    assert abs(accepted / n - float(torch.minimum(p, q).sum())) < 0.015


def test_acceptance_probability_is_min_one_p_over_q():
    p = torch.tensor([0.8, 0.2])
    q = torch.tensor([0.2, 0.8])
    gen = torch.Generator().manual_seed(1)
    acc0 = sum(verify_sampled(p, q, 0, gen)[0] for _ in range(2000)) / 2000  # p/q = 4 -> always accept
    acc1 = sum(verify_sampled(p, q, 1, gen)[0] for _ in range(4000)) / 4000  # p/q = 0.25
    assert acc0 == 1.0 and abs(acc1 - 0.25) < 0.03
