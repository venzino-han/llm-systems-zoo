"""L02 tests: paged/contiguous allocator simulator and KV capacity model (CPU, offline)."""

from __future__ import annotations

import pytest

from fundamentals.memory_calculator import KNOWN_CONFIGS
from inference.kv_cache.kv_capacity import (
    capacity_table,
    max_concurrency,
    max_kv_tokens,
    mha_equivalent,
)
from inference.kv_cache.paged_allocator import ContiguousAllocator, OutOfMemory, PagedAllocator
from inference.kv_cache.simulate_workloads import (
    WorkloadSpec,
    generate_requests,
    run_contiguous,
    run_paged,
)

QWEN05 = KNOWN_CONFIGS["Qwen/Qwen2.5-0.5B-Instruct"]


# ---- contiguous ----------------------------------------------------------------------------------
def test_contiguous_reserves_max_seq_len_and_fragments():
    a = ContiguousAllocator(capacity_tokens=100, max_seq_len=40)
    a.allocate("s1", list(range(10)))
    a.allocate("s2", list(range(10)))
    assert a.allocated_slots == 80 and a.used_tokens == 20
    assert a.stats()["internal_fragmentation"] == pytest.approx(0.75)
    with pytest.raises(OutOfMemory):  # 20 free slots < 40 reservation although 80 tokens are unused
        a.allocate("s3", [1])
    a.free("s1")
    a.allocate("s3", [1])  # freed range is reused
    assert a.stats()["active_sequences"] == 2


def test_contiguous_append_bounded_by_reservation_and_coalesces_free_ranges():
    a = ContiguousAllocator(capacity_tokens=30, max_seq_len=10)
    for s in ("a", "b", "c"):
        a.allocate(s, [0] * 9)
    a.append("a", 1)
    with pytest.raises(OutOfMemory):
        a.append("a", 2)
    for s in ("a", "c", "b"):
        a.free(s)
    assert a.largest_free_range == 30


# ---- paged ---------------------------------------------------------------------------------------
def test_paged_block_table_maps_logical_to_physical_and_reconstructs_tokens():
    p = PagedAllocator(num_blocks=8, block_size=4, enable_prefix_caching=False)
    toks = list(range(10))
    p.allocate("s", toks)
    assert len(p.block_table("s")) == 3 and p.tokens("s") == toks
    assert p.allocated_slots == 12 and p.stats()["internal_fragmentation"] == pytest.approx(2 / 12)
    for t in range(10, 13):
        p.append("s", t)
    assert len(p.block_table("s")) == 4 and p.tokens("s") == list(range(13))
    p.free("s")
    assert p.num_free_blocks == 8 and p.num_evictable_blocks == 0  # no caching → blocks go straight to the free list


def test_paged_out_of_memory_rolls_back_partial_allocation():
    p = PagedAllocator(num_blocks=3, block_size=4)
    with pytest.raises(OutOfMemory):
        p.allocate("big", list(range(16)))  # needs 4 blocks
    assert p.num_free_blocks == 3 and p.stats()["failed_allocations"] == 1
    p.allocate("ok", list(range(12)))
    assert p.num_free_blocks == 0


def test_prefix_sharing_hits_full_blocks_only_and_refcounts():
    p = PagedAllocator(num_blocks=16, block_size=4)
    sysp = list(range(100, 110))  # 2 full blocks + 2-token partial
    assert p.allocate("a", sysp + [1, 2]) == 0           # cold: nothing cached yet
    assert p.allocate("b", sysp + [3, 4]) == 8           # the 2 full prefix blocks hit; the mixed 3rd block does not
    ta, tb = p.block_table("a"), p.block_table("b")
    assert ta[:2] == tb[:2] and ta[2] != tb[2]
    assert p.blocks[ta[0]].ref_count == 2
    st = p.stats()
    assert st["cache_hits"] == 2 and st["cache_queries"] == 6 and st["sharing_ratio"] > 1
    p.free("a")
    assert p.blocks[ta[0]].ref_count == 1  # still referenced by b
    p.free("b")
    assert p.blocks[ta[0]].ref_count == 0 and p.num_evictable_blocks == 4  # 4 distinct full blocks are retained


def test_chained_hash_distinguishes_same_block_content_at_different_positions():
    p = PagedAllocator(num_blocks=16, block_size=4)
    p.allocate("a", [1, 2, 3, 4, 9, 9, 9, 9])
    assert p.allocate("b", [9, 9, 9, 9]) == 0  # [9,9,9,9] exists only as the *second* block of a


def test_lru_eviction_frees_least_recently_used_cached_block():
    p = PagedAllocator(num_blocks=4, block_size=2)
    p.allocate("a", [1, 2])
    p.allocate("b", [3, 4])
    p.free("a")
    p.free("b")  # both cached, LRU order: a first, then b
    assert p.num_free_blocks == 2 and p.num_evictable_blocks == 2
    p.allocate("c", [5, 6, 7, 8])               # consumes the 2 free blocks
    p.allocate("d", [9, 10])                    # must evict: a's block goes first
    assert p.stats()["evictions"] == 1
    assert p.allocate("e", [3, 4]) == 2         # b's block survived → hit
    with pytest.raises(OutOfMemory):
        p.allocate("f", [11, 12])


def test_cache_hit_revives_evictable_block():
    p = PagedAllocator(num_blocks=2, block_size=2)
    p.allocate("a", [1, 2])
    p.free("a")
    assert p.num_evictable_blocks == 1
    assert p.allocate("b", [1, 2]) == 2
    assert p.num_evictable_blocks == 0 and p.blocks[p.block_table("b")[0]].ref_count == 1


def test_copy_on_write_after_fork():
    p = PagedAllocator(num_blocks=8, block_size=4)
    p.allocate("parent", [1, 2, 3, 4, 5])        # block0 full, block1 partial [5]
    p.fork("parent", "child")
    assert p.block_table("parent") == p.block_table("child")
    p.append("child", 6)
    assert p.stats()["cow_copies"] == 1
    assert p.block_table("parent")[0] == p.block_table("child")[0]      # full block still shared
    assert p.block_table("parent")[1] != p.block_table("child")[1]      # partial block copied
    assert p.tokens("parent") == [1, 2, 3, 4, 5] and p.tokens("child") == [1, 2, 3, 4, 5, 6]
    p.append("parent", 7)
    assert p.stats()["cow_copies"] == 1  # parent is now sole owner of its partial block → no copy
    p.free("parent")
    p.free("child")
    assert all(b.ref_count == 0 for b in p.blocks)


def test_fragmentation_bounded_by_block_size():
    p = PagedAllocator(num_blocks=1000, block_size=16)
    for i in range(50):
        p.allocate(i, list(range(1000 + i, 1000 + i + 17)))  # 17 tokens → 2 blocks, 15 wasted slots
    assert p.stats()["internal_fragmentation"] == pytest.approx(15 / 32)


# ---- capacity model ------------------------------------------------------------------------------
def test_capacity_model_gqa_vs_mha():
    gqa = max_kv_tokens(QWEN05, 48, utilisation=0.9)
    mha = max_kv_tokens(mha_equivalent(QWEN05), 48, utilisation=0.9)
    assert 6.5 < gqa / mha < 7.5  # 14 q heads / 2 kv heads, minus the slightly larger MHA weights
    assert max_concurrency(QWEN05, 48, 1024, 256) == gqa // 1280
    assert max_kv_tokens(QWEN05, 0.5) == 0  # weights do not fit → zero tokens, never negative
    rows = capacity_table(["Qwen/Qwen2.5-0.5B-Instruct"], 48, 0.9, 1024, 256)
    assert len(rows) == 2 and rows[0]["max_kv_tokens"] > rows[1]["max_kv_tokens"]


# ---- simulator ----------------------------------------------------------------------------------
def test_simulator_paged_beats_contiguous_under_memory_pressure():
    spec = WorkloadSpec(n_requests=30, prompt_len_mean=256, gen_len_mean=64, shared_prefix_ratio=0.0, seed=1)
    c = run_contiguous(spec, capacity_tokens=4096)
    p = run_paged(spec, capacity_tokens=4096, block_size=16)
    assert c.tokens_generated == p.tokens_generated == sum(r.gen_len for r in generate_requests(spec))
    assert p.mean_internal_fragmentation < c.mean_internal_fragmentation
    assert p.makespan_steps <= c.makespan_steps


def test_simulator_prefix_sharing_raises_hit_fraction():
    base = WorkloadSpec(n_requests=30, shared_prefix_len=512, shared_prefix_ratio=0.0, seed=2)
    shared = WorkloadSpec(n_requests=30, shared_prefix_len=512, shared_prefix_ratio=1.0, seed=2)
    r0 = run_paged(base, 65_536, 16)
    r1 = run_paged(shared, 65_536, 16)
    assert r0.allocator_stats["cache_hit_tokens"] == 0
    assert r1.allocator_stats["cache_hit_tokens"] >= 29 * 512
    assert 0 < r1.allocator_stats["hit_token_fraction"] <= 1.0
