"""Packing correctness (L05): bins, masks, positions, label masking, accounting. CPU only."""

from __future__ import annotations

import torch

from training.packing import (
    IGNORE_INDEX,
    block_diagonal_causal_mask,
    build_packed_batch,
    build_padded_batch,
    cu_seqlens,
    first_fit,
    first_fit_decreasing,
    packing_efficiency,
    restart_position_ids,
)


def test_first_fit_respects_capacity_and_order():
    lengths = [4, 3, 5, 2, 6, 1]
    bins = first_fit(lengths, max_len=8)
    for b in bins:
        assert sum(lengths[i] for i in b) <= 8
    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
    assert bins[0][0] == 0  # arrival order preserved


def test_first_fit_decreasing_uses_no_more_bins():
    lengths = [7, 2, 6, 3, 5, 4, 1, 8]
    assert len(first_fit_decreasing(lengths, 8)) <= len(first_fit(lengths, 8))


def test_block_mask_has_no_cross_sample_attention():
    m = block_diagonal_causal_mask([3, 2], max_len=6)
    assert m[0:3, 0:3].equal(torch.ones(3, 3, dtype=torch.bool).tril())
    assert not m[3:5, 0:3].any()  # sample 2 cannot see sample 1
    assert not m[0:3, 3:5].any()  # sample 1 cannot see sample 2 (causal anyway)
    assert not m[5].any()  # padding row attends to nothing


def test_position_ids_restart_and_cu_seqlens():
    assert restart_position_ids([3, 2], 6).tolist() == [0, 1, 2, 0, 1, 0]
    assert cu_seqlens([3, 2]).tolist() == [0, 3, 5]


def test_packed_batch_labels_mask_sample_starts_and_padding():
    samples = [[10, 11, 12], [20, 21], [30]]
    bins = first_fit([3, 2, 1], max_len=6)  # all in one row
    pb = build_packed_batch(samples, bins, max_len=6, pad_id=0, mask_mode="block")
    assert pb.input_ids.tolist() == [[10, 11, 12, 20, 21, 30]]
    # token 20 (start of sample 2) and 30 (start of sample 3) are not targets
    assert pb.labels.tolist() == [[10, 11, 12, IGNORE_INDEX, 21, IGNORE_INDEX]]
    assert pb.position_ids.tolist() == [[0, 1, 2, 0, 1, 0]]
    assert pb.attention_mask.shape == (1, 1, 6, 6)
    assert pb.n_real_tokens == 6 and pb.n_pad_tokens == 0 and pb.n_samples == 3


def test_packed_batch_cu_seqlens_mode_and_padding_accounting():
    samples = [[1, 2], [3, 4, 5]]
    pb = build_packed_batch(samples, [[0, 1]], max_len=8, pad_id=0, mask_mode="cu_seqlens")
    assert pb.attention_mask is None
    assert pb.cu_seqlens[0].tolist() == [0, 2, 5]
    assert pb.labels[0, 5:].tolist() == [IGNORE_INDEX] * 3
    assert pb.n_pad_tokens == 3


def test_padded_batch_baseline_accounting():
    pb = build_padded_batch([[1, 2, 3], [4]], pad_id=0)
    assert pb.input_ids.shape == (2, 3)
    assert pb.n_real_tokens == 4 and pb.n_pad_tokens == 2
    assert pb.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_packing_efficiency_improves_over_dynamic_padding():
    lengths = [100, 20, 300, 50, 10, 250, 400, 5] * 4
    r = packing_efficiency(lengths, max_len=512, padded_batch_size=4)
    assert r["real_tokens"] == sum(lengths)
    assert r["packed_efficiency"] > r["padded_efficiency"]
    assert r["packed_rows"] < r["padded_rows"]


def test_packed_forward_matches_separate_forward_on_tiny_model():
    """A packed row must produce the same logits for each sample as running the samples alone."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    samples = [[5, 6, 7, 8], [9, 10, 11]]
    pb = build_packed_batch(samples, [[0, 1]], max_len=8, pad_id=0, mask_mode="block")
    with torch.no_grad():
        packed = model(input_ids=pb.input_ids, attention_mask=pb.attention_mask, position_ids=pb.position_ids).logits[0]
        a = model(input_ids=torch.tensor([samples[0]])).logits[0]
        b = model(input_ids=torch.tensor([samples[1]])).logits[0]
    assert torch.allclose(packed[0:4], a, atol=1e-4)
    assert torch.allclose(packed[4:7], b, atol=1e-4)
