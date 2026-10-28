"""L02 — educational KV-cache allocators: contiguous vs paged (block) allocation.

This is a *simulator*: it tracks which slots hold which tokens and never touches a GPU. It
reproduces the bookkeeping that vLLM's block manager does (block tables, reference counts,
hash-based prefix caching, LRU eviction, copy-on-write) so the trade-offs can be measured on a
laptop. The production engine additionally has to move real tensors, schedule kernels that
gather blocks (PagedAttention), and deal with multi-GPU placement — none of that is modelled.

Two allocators, same interface:

    ContiguousAllocator   reserves `max_seq_len` slots per sequence up front (what naive HF
                          generate / early FasterTransformer did). Simple, zero lookup cost,
                          but internal fragmentation = reserved - used, and no sharing.
    PagedAllocator        slots are grouped into fixed-size blocks; a sequence owns a *block
                          table* (logical block index → physical block id). Full blocks are
                          content-hashed (chained with the previous block's hash) so identical
                          prefixes share physical blocks. Blocks whose refcount drops to zero
                          stay cached and are evicted LRU when the free list runs dry.

Stats exposed by both: capacity, used tokens, allocated slots, utilisation, waste
(internal fragmentation), plus cache hits / misses / evictions / copy-on-write counts for the
paged allocator.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence


class OutOfMemory(Exception):
    """Raised when an allocation cannot be satisfied (the engine would queue/preempt)."""


# ---------------------------------------------------------------------------------------------
# Contiguous allocator
# ---------------------------------------------------------------------------------------------
@dataclass
class _ContigSeq:
    start: int
    reserved: int
    length: int


class ContiguousAllocator:
    """Reserve `max_seq_len` contiguous slots per sequence (first-fit over a free-range list)."""

    def __init__(self, capacity_tokens: int, max_seq_len: int) -> None:
        if max_seq_len <= 0 or capacity_tokens <= 0:
            raise ValueError("capacity and max_seq_len must be positive")
        self.capacity = capacity_tokens
        self.max_seq_len = max_seq_len
        self._free: list[tuple[int, int]] = [(0, capacity_tokens)]  # (start, length)
        self._seqs: dict[Any, _ContigSeq] = {}
        self.failed_allocations = 0

    def can_allocate(self, n_tokens: int) -> bool:
        need = max(n_tokens, self.max_seq_len)
        return any(length >= need for _, length in self._free)

    def allocate(self, seq_id: Any, token_ids: Sequence[int]) -> int:
        if seq_id in self._seqs:
            raise KeyError(f"sequence {seq_id} already allocated")
        n = len(token_ids)
        if n > self.max_seq_len:
            raise OutOfMemory(f"prompt of {n} tokens exceeds max_seq_len={self.max_seq_len}")
        need = self.max_seq_len
        for i, (start, length) in enumerate(self._free):
            if length >= need:
                if length == need:
                    self._free.pop(i)
                else:
                    self._free[i] = (start + need, length - need)
                self._seqs[seq_id] = _ContigSeq(start, need, n)
                return 0  # nothing is ever shared
        self.failed_allocations += 1
        raise OutOfMemory(f"no contiguous range of {need} slots")

    def append(self, seq_id: Any, token_id: int) -> None:
        seq = self._seqs[seq_id]
        if seq.length + 1 > seq.reserved:
            raise OutOfMemory("sequence exceeded its reservation")
        seq.length += 1

    def free(self, seq_id: Any) -> None:
        seq = self._seqs.pop(seq_id)
        self._free.append((seq.start, seq.reserved))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for start, length in self._free:  # coalesce neighbours
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self._free = merged

    @property
    def used_tokens(self) -> int:
        return sum(s.length for s in self._seqs.values())

    @property
    def allocated_slots(self) -> int:
        return sum(s.reserved for s in self._seqs.values())

    @property
    def free_slots(self) -> int:
        return self.capacity - self.allocated_slots

    @property
    def largest_free_range(self) -> int:
        return max((length for _, length in self._free), default=0)

    def stats(self) -> dict[str, Any]:
        alloc = self.allocated_slots
        return {
            "allocator": "contiguous",
            "capacity": self.capacity,
            "active_sequences": len(self._seqs),
            "used_tokens": self.used_tokens,
            "allocated_slots": alloc,
            "free_slots": self.free_slots,
            "largest_free_range": self.largest_free_range,
            "utilisation": self.used_tokens / self.capacity,
            "internal_fragmentation": (alloc - self.used_tokens) / alloc if alloc else 0.0,
            "external_fragmentation": 1 - self.largest_free_range / self.free_slots if self.free_slots else 0.0,
            "failed_allocations": self.failed_allocations,
        }


# ---------------------------------------------------------------------------------------------
# Paged allocator
# ---------------------------------------------------------------------------------------------
@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    token_ids: list[int] = field(default_factory=list)
    content_hash: int | None = None  # set once the block is full (and therefore immutable)

    def is_full(self, block_size: int) -> bool:
        return len(self.token_ids) == block_size


@dataclass
class _PagedSeq:
    token_ids: list[int]
    block_table: list[int]  # logical block index -> physical block id


def _chain_hash(prev_hash: int | None, tokens: Sequence[int]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(prev_hash).encode())
    h.update(",".join(map(str, tokens)).encode())
    return int.from_bytes(h.digest(), "big")


class PagedAllocator:
    """Block-based allocation with optional automatic prefix caching.

    Memory model: `num_blocks` physical blocks of `block_size` token slots. A sequence's logical
    blocks map to arbitrary physical blocks via its block table, so no contiguous range is ever
    needed and the only waste is the unfilled tail of the last block (< block_size tokens).
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() gives block 0 first
        self._cached: dict[int, int] = {}                        # content hash -> block id
        self._evictable: OrderedDict[int, None] = OrderedDict()  # LRU of cached blocks with ref 0
        self._seqs: dict[Any, _PagedSeq] = {}
        self.cache_queries = 0
        self.cache_hits = 0
        self.cache_hit_tokens = 0
        self.allocated_prompt_tokens = 0  # includes re-allocations after preemption
        self.evictions = 0
        self.cow_copies = 0
        self.failed_allocations = 0

    # -- capacity --
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_evictable_blocks(self) -> int:
        return len(self._evictable)

    @property
    def num_available_blocks(self) -> int:
        """Free plus evictable: what a new sequence can actually obtain."""
        return self.num_free_blocks + self.num_evictable_blocks

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    def can_allocate(self, n_tokens: int) -> bool:
        """Conservative: ignores prefix hits (the engine checks exactly; we keep admission simple)."""
        return self.blocks_needed(n_tokens) <= self.num_available_blocks

    # -- internals --
    def _get_free_block(self) -> Block:
        if self._free:
            return self.blocks[self._free.pop()]
        if self._evictable:
            victim_id, _ = self._evictable.popitem(last=False)  # least recently used
            victim = self.blocks[victim_id]
            assert victim.ref_count == 0
            if victim.content_hash is not None and self._cached.get(victim.content_hash) == victim_id:
                del self._cached[victim.content_hash]
            victim.token_ids = []
            victim.content_hash = None
            self.evictions += 1
            return victim
        raise OutOfMemory("no free or evictable blocks")

    def _release_block(self, block: Block) -> None:
        block.ref_count -= 1
        if block.ref_count > 0:
            return
        if self.enable_prefix_caching and block.content_hash is not None:
            self._evictable[block.block_id] = None  # keep contents, allow reuse until evicted
        else:
            block.token_ids = []
            block.content_hash = None
            self._free.append(block.block_id)

    def _discard_block(self, block: Block) -> None:
        """Undo a fresh allocation (rollback): unregister its hash and return it to the free list."""
        if block.content_hash is not None and self._cached.get(block.content_hash) == block.block_id:
            del self._cached[block.content_hash]
        block.ref_count = 0
        block.token_ids = []
        block.content_hash = None
        self._free.append(block.block_id)

    def _touch(self, block: Block) -> None:
        """A cache hit on an evictable block revives it (ref 0 -> 1) and removes it from the LRU."""
        if block.ref_count == 0:
            self._evictable.pop(block.block_id, None)
        block.ref_count += 1

    def _seal(self, block: Block, prev_hash: int | None) -> int | None:
        """Called when a block becomes full: compute its chained hash and register it."""
        if not self.enable_prefix_caching:
            return None
        h = _chain_hash(prev_hash, block.token_ids)
        block.content_hash = h
        self._cached.setdefault(h, block.block_id)
        return h

    # -- interface --
    def allocate(self, seq_id: Any, token_ids: Sequence[int]) -> int:
        """Allocate blocks for a prompt. Returns the number of tokens served from the prefix cache."""
        if seq_id in self._seqs:
            raise KeyError(f"sequence {seq_id} already allocated")
        tokens = list(token_ids)
        table: list[int] = []
        acquired: list[tuple[Block, bool]] = []  # (block, was_cache_hit)
        prev_hash: int | None = None
        cached_tokens = 0
        try:
            for start in range(0, len(tokens), self.block_size):
                chunk = tokens[start : start + self.block_size]
                full = len(chunk) == self.block_size
                if full and self.enable_prefix_caching:
                    self.cache_queries += 1
                    h = _chain_hash(prev_hash, chunk)
                    hit_id = self._cached.get(h)
                    if hit_id is not None and self.blocks[hit_id].token_ids == chunk:
                        block = self.blocks[hit_id]
                        self._touch(block)
                        acquired.append((block, True))
                        table.append(block.block_id)
                        self.cache_hits += 1
                        cached_tokens += len(chunk)
                        prev_hash = h
                        continue
                block = self._get_free_block()
                block.ref_count = 1
                block.token_ids = list(chunk)
                acquired.append((block, False))
                table.append(block.block_id)
                if full:
                    prev_hash = self._seal(block, prev_hash)
        except OutOfMemory:
            for b, was_hit in reversed(acquired):  # roll back
                if was_hit:
                    self._release_block(b)
                else:
                    self._discard_block(b)
            self.failed_allocations += 1
            raise
        self._seqs[seq_id] = _PagedSeq(tokens, table)
        self.cache_hit_tokens += cached_tokens
        self.allocated_prompt_tokens += len(tokens)
        return cached_tokens

    def append(self, seq_id: Any, token_id: int) -> None:
        seq = self._seqs[seq_id]
        last = self.blocks[seq.block_table[-1]] if seq.block_table else None
        if last is None or last.is_full(self.block_size):
            block = self._get_free_block()
            block.ref_count = 1
            seq.block_table.append(block.block_id)
            last = block
        elif last.ref_count > 1:
            # Copy-on-write: another sequence shares this partial block, so take a private copy.
            new = self._get_free_block()
            new.ref_count = 1
            new.token_ids = list(last.token_ids)
            self._release_block(last)
            seq.block_table[-1] = new.block_id
            self.cow_copies += 1
            last = new
        last.token_ids.append(token_id)
        seq.token_ids.append(token_id)
        if last.is_full(self.block_size):
            prev = self.blocks[seq.block_table[-2]].content_hash if len(seq.block_table) > 1 else None
            self._seal(last, prev)

    def fork(self, seq_id: Any, new_seq_id: Any) -> None:
        """Parallel sampling / beam search: the child shares every block, including the partial
        last one. The next `append` on either sequence triggers copy-on-write for that block."""
        if new_seq_id in self._seqs:
            raise KeyError(f"sequence {new_seq_id} already allocated")
        seq = self._seqs[seq_id]
        for bid in seq.block_table:
            self.blocks[bid].ref_count += 1
        self._seqs[new_seq_id] = _PagedSeq(list(seq.token_ids), list(seq.block_table))

    def free(self, seq_id: Any) -> None:
        seq = self._seqs.pop(seq_id)
        for bid in seq.block_table:
            self._release_block(self.blocks[bid])

    def block_table(self, seq_id: Any) -> list[int]:
        return list(self._seqs[seq_id].block_table)

    def tokens(self, seq_id: Any) -> list[int]:
        """Reconstruct the sequence from physical blocks (used by tests to check mappings)."""
        return [t for bid in self._seqs[seq_id].block_table for t in self.blocks[bid].token_ids]

    # -- stats --
    @property
    def used_tokens(self) -> int:
        return sum(len(s.token_ids) for s in self._seqs.values())

    @property
    def allocated_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks - self.num_evictable_blocks

    @property
    def allocated_slots(self) -> int:
        return self.allocated_blocks * self.block_size

    def stats(self) -> dict[str, Any]:
        alloc = self.allocated_slots
        # physical tokens actually resident in referenced blocks (shared blocks counted once)
        resident = sum(len(b.token_ids) for b in self.blocks if b.ref_count > 0)
        return {
            "allocator": "paged",
            "block_size": self.block_size,
            "capacity": self.num_blocks * self.block_size,
            "active_sequences": len(self._seqs),
            "used_tokens": self.used_tokens,            # logical tokens across sequences
            "resident_tokens": resident,                # physical slots holding tokens
            "allocated_slots": alloc,
            "free_blocks": self.num_free_blocks,
            "evictable_blocks": self.num_evictable_blocks,
            "utilisation": resident / (self.num_blocks * self.block_size),
            "internal_fragmentation": (alloc - resident) / alloc if alloc else 0.0,
            "sharing_ratio": self.used_tokens / resident if resident else 1.0,
            "cache_queries": self.cache_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.cache_queries if self.cache_queries else 0.0,
            "cache_hit_tokens": self.cache_hit_tokens,
            "allocated_prompt_tokens": self.allocated_prompt_tokens,
            "hit_token_fraction": self.cache_hit_tokens / self.allocated_prompt_tokens if self.allocated_prompt_tokens else 0.0,
            "evictions": self.evictions,
            "cow_copies": self.cow_copies,
            "failed_allocations": self.failed_allocations,
        }
