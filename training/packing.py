"""Educational sequence packing (L05).

Why packing exists
------------------
A training batch is a dense [B, L] tensor. Samples are shorter than L and have different lengths,
so a padded batch wastes `sum(L - len_i)` positions on every forward/backward pass. Packing puts
several samples into one row of length L so the model does real work on (almost) every position.

Correctness requirements when several samples share a row
--------------------------------------------------------
1. Attention must not cross sample boundaries → block-diagonal causal mask, or `cu_seqlens`
   for a varlen (flash-attention style) kernel that never materialises the mask.
2. Position ids must restart at 0 for each sample, otherwise RoPE/absolute embeddings see positions
   the sample would never have at inference time.
3. The first token of every sample (except the first sample of the row) must not be a training
   target: with the usual shifted-label loss it would be predicted from the *previous* sample's
   last token. Padding positions are never targets either.

This module is the "Implement" depth of the lab: plain PyTorch, CPU-testable, no kernels. The
production path (TRL `padding_free=True` / `packing_strategy="bfd"`, flash-attn varlen) avoids the
O(L^2) mask entirely; see README section 1 for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

IGNORE_INDEX = -100


# --------------------------------------------------------------------------------------------
# Bin packing
# --------------------------------------------------------------------------------------------
def first_fit(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """Greedy first-fit: each sample (in arrival order) goes into the first bin with room.

    Returns bins as lists of sample indices. O(n * bins); fine for a few thousand samples per
    pass, which is how a streaming data loader would use it (pack per shard, not per dataset).
    """
    bins: list[list[int]] = []
    remaining: list[int] = []
    for idx, n in enumerate(lengths):
        if n > max_len:
            raise ValueError(f"sample {idx} has length {n} > max_len {max_len}; truncate first")
        for b, room in enumerate(remaining):
            if n <= room:
                bins[b].append(idx)
                remaining[b] -= n
                break
        else:
            bins.append([idx])
            remaining.append(max_len - n)
    return bins


def first_fit_decreasing(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """First-fit on samples sorted by decreasing length: tighter bins, but the batch order no
    longer follows the data order (a mild shuffling side-effect, usually acceptable)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    sorted_bins = first_fit([lengths[i] for i in order], max_len)
    return [[order[j] for j in b] for b in sorted_bins]


def truncate(samples: Iterable[Sequence[int]], max_len: int) -> list[list[int]]:
    return [list(s[:max_len]) for s in samples]


# --------------------------------------------------------------------------------------------
# Masks and positions
# --------------------------------------------------------------------------------------------
def block_diagonal_causal_mask(lengths: Sequence[int], max_len: int) -> torch.Tensor:
    """Boolean [max_len, max_len] mask, True where query i may attend key j.

    Samples are laid out back to back; padding (positions >= sum(lengths)) attends to nothing
    (all False row), which the HF 4-D mask path handles as fully masked.
    """
    mask = torch.zeros(max_len, max_len, dtype=torch.bool)
    start = 0
    for n in lengths:
        end = start + n
        mask[start:end, start:end] = torch.ones(n, n, dtype=torch.bool).tril()
        start = end
    return mask


def restart_position_ids(lengths: Sequence[int], max_len: int) -> torch.Tensor:
    pos = torch.zeros(max_len, dtype=torch.long)
    start = 0
    for n in lengths:
        pos[start : start + n] = torch.arange(n)
        start += n
    return pos


def cu_seqlens(lengths: Sequence[int]) -> torch.Tensor:
    """Cumulative sequence lengths in the flash-attn varlen convention: [0, l1, l1+l2, ...]."""
    return torch.tensor([0, *torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0).tolist()], dtype=torch.int32)


# --------------------------------------------------------------------------------------------
# Batch construction
# --------------------------------------------------------------------------------------------
@dataclass
class PackedBatch:
    input_ids: torch.Tensor           # [B, L]
    labels: torch.Tensor              # [B, L], IGNORE_INDEX on pad and on sample-start positions
    position_ids: torch.Tensor        # [B, L], restarts per sample
    attention_mask: torch.Tensor | None = None  # [B, 1, L, L] bool, block-diagonal causal
    cu_seqlens: list[torch.Tensor] = field(default_factory=list)  # per row, varlen boundaries
    lengths: list[list[int]] = field(default_factory=list)
    n_real_tokens: int = 0
    n_pad_tokens: int = 0
    n_samples: int = 0

    def to(self, device) -> PackedBatch:
        self.input_ids = self.input_ids.to(device)
        self.labels = self.labels.to(device)
        self.position_ids = self.position_ids.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        return self

    def model_kwargs(self) -> dict:
        kw = {"input_ids": self.input_ids, "labels": self.labels, "position_ids": self.position_ids}
        if self.attention_mask is not None:
            kw["attention_mask"] = self.attention_mask
        return kw


def mask_cross_sample_labels(labels: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Set the first token of every sample after the first one to IGNORE_INDEX (in place).

    With HF-style shifted labels, labels[t] is predicted from inputs[:t]. The first token of
    sample k (k >= 1) sits right after sample k-1's last token, so it would be a target
    conditioned on foreign context. The very first token of the row is dropped by the shift
    itself, so it needs no explicit mask.
    """
    start = 0
    for k, n in enumerate(lengths):
        if k > 0:
            labels[start] = IGNORE_INDEX
        start += n
    return labels


def build_packed_batch(
    samples: Sequence[Sequence[int]],
    bins: Sequence[Sequence[int]],
    max_len: int,
    pad_id: int,
    mask_mode: str = "block",
) -> PackedBatch:
    """Materialise packed rows from a bin assignment.

    mask_mode="block"      → 4-D block-diagonal causal mask (works with eager/sdpa attention)
    mask_mode="cu_seqlens" → no mask tensor; returns per-row cu_seqlens for a varlen kernel
    """
    rows, labels, pos, masks, cus, lens = [], [], [], [], [], []
    n_real = 0
    for b in bins:
        ids: list[int] = []
        ls: list[int] = []
        for idx in b:
            s = list(samples[idx])
            ids.extend(s)
            ls.append(len(s))
        n_real += len(ids)
        pad = max_len - len(ids)
        row = torch.tensor(ids + [pad_id] * pad, dtype=torch.long)
        lab = torch.tensor(ids + [IGNORE_INDEX] * pad, dtype=torch.long)
        mask_cross_sample_labels(lab, ls)
        rows.append(row)
        labels.append(lab)
        pos.append(restart_position_ids(ls, max_len))
        lens.append(ls)
        if mask_mode == "block":
            masks.append(block_diagonal_causal_mask(ls, max_len))
        elif mask_mode == "cu_seqlens":
            cus.append(cu_seqlens(ls))
        else:
            raise ValueError(mask_mode)
    B = len(bins)
    return PackedBatch(
        input_ids=torch.stack(rows),
        labels=torch.stack(labels),
        position_ids=torch.stack(pos),
        attention_mask=torch.stack(masks).unsqueeze(1) if masks else None,
        cu_seqlens=cus,
        lengths=lens,
        n_real_tokens=n_real,
        n_pad_tokens=B * max_len - n_real,
        n_samples=sum(len(b) for b in bins),
    )


def build_padded_batch(samples: Sequence[Sequence[int]], pad_id: int, max_len: int | None = None) -> PackedBatch:
    """The baseline: one sample per row, right-padded to the longest sample (or max_len)."""
    L = max_len or max(len(s) for s in samples)
    rows, labels, pos = [], [], []
    n_real = 0
    for s in samples:
        s = list(s[:L])
        n_real += len(s)
        pad = L - len(s)
        rows.append(torch.tensor(s + [pad_id] * pad, dtype=torch.long))
        labels.append(torch.tensor(s + [IGNORE_INDEX] * pad, dtype=torch.long))
        pos.append(torch.tensor(list(range(len(s))) + [0] * pad, dtype=torch.long))
    ids = torch.stack(rows)
    return PackedBatch(
        input_ids=ids,
        labels=torch.stack(labels),
        position_ids=torch.stack(pos),
        attention_mask=(ids != pad_id).long() if pad_id is not None else None,  # 2-D key-padding mask
        lengths=[[len(s)] for s in samples],
        n_real_tokens=n_real,
        n_pad_tokens=ids.numel() - n_real,
        n_samples=len(samples),
    )


# --------------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------------
def packing_efficiency(lengths: Sequence[int], max_len: int, padded_batch_size: int, strategy: str = "first_fit") -> dict:
    """Compare tokens processed by padded batching vs packing for the same samples.

    Padded batching pads each batch of `padded_batch_size` samples to its longest member
    (dynamic padding), which is already better than padding everything to max_len.
    """
    lengths = [min(n, max_len) for n in lengths]
    real = sum(lengths)
    padded_total = 0
    for i in range(0, len(lengths), padded_batch_size):
        chunk = lengths[i : i + padded_batch_size]
        padded_total += max(chunk) * len(chunk)
    packer = first_fit if strategy == "first_fit" else first_fit_decreasing
    bins = packer(lengths, max_len)
    packed_total = len(bins) * max_len
    return {
        "n_samples": len(lengths),
        "real_tokens": real,
        "padded_tokens": padded_total,
        "padded_efficiency": real / padded_total if padded_total else 0.0,
        "packed_rows": len(bins),
        "packed_tokens": packed_total,
        "packed_efficiency": real / packed_total if packed_total else 0.0,
        "padded_rows": len(lengths),
    }
