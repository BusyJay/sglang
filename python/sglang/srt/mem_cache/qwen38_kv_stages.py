# SPDX-License-Identifier: Apache-2.0
"""Stage target KV backing and adaptive experts at existing result boundaries."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_stages(stages):
    if stages is None:
        return None
    if (
        not isinstance(stages, list)
        or len(stages) != 3
        or any(type(n) is not int or n <= 0 for n in stages)
        or not stages[0] < stages[1] < stages[2]
    ):
        raise ValueError(
            "kv_cache_stages must contain three increasing positive integers"
        )
    return tuple(stages)


def page_capacities(stages, page_size):
    """Keep user boundaries in tokens while backing their rounded-up pages."""
    capacities = tuple((n + page_size - 1) // page_size * page_size for n in stages)
    if len(set(capacities)) != len(stages):
        raise ValueError("KV stages collapse after page rounding")
    return capacities


class OneWayKVStages:
    """Advance before demand reaches a boundary, publishing only after return.

    ``used`` includes cached and in-flight physical pages, not just the running
    request's context. A failing return poisons the controller: serving must stop.
    """

    def __init__(self, capacities, return_stage):
        self.capacities = validate_stages(list(capacities))
        self.index = 0
        self.return_stage = return_stage
        self.poisoned = False

    @property
    def capacity(self):
        return self.capacities[self.index]

    def prepare(self, used, reserve):
        if self.poisoned:
            raise RuntimeError("KV stage return previously failed")
        if used < 0 or reserve < 0:
            raise ValueError("negative KV usage or reserve")
        while self.index < len(self.capacities) - 1 and used + reserve >= self.capacity:
            next_index = self.index + 1
            try:
                self.return_stage(next_index)
            except BaseException:
                self.poisoned = True
                raise
            self.index = next_index
        return self.capacity


def distribute_slots(current, budget, minimum, maximum):
    """Keep the current layer allocation's proportions under a changed budget."""
    if not len(current) * minimum <= budget <= len(current) * maximum:
        raise ValueError("expert budget cannot satisfy per-layer route capacity")
    weights = [max(n - minimum, 0) for n in current]
    layout = [minimum] * len(current)
    remaining = budget - sum(layout)
    while remaining:
        candidates = [i for i, n in enumerate(layout) if n < maximum]
        owner = max(
            candidates, key=lambda i: (weights[i] + 1) / (layout[i] - minimum + 1)
        )
        layout[owner] += 1
        remaining -= 1
    return layout


def prepare_expert_loan(runtime, pool):
    """Fit extra directly executable slots in the actual uncommitted KV bytes."""
    owner = pool._post_capture_owner
    stages = page_capacities(validate_stages(runtime.kv_cache_stages), pool.page_size)
    if pool.size != stages[-1]:
        raise ValueError("final KV stage must match the page-rounded pool capacity")
    spans = [owner.capacity_backed_bytes(n) for n in stages]
    runtime.expert_loan_budgets = runtime.staged_pool.configure_loans(
        [spans[-1] - span for span in spans]
    )
    runtime.initialize_staged_layout()
    logger.info(
        "Qwen38 adaptive KV stages=%s expert budgets=%s",
        stages,
        runtime.expert_loan_budgets,
    )


class StagedKVLoan:
    def __init__(self, allocator, pool, runtime):
        import torch

        self.allocator = allocator
        self.pool = pool
        self.runtime = runtime
        self.policy = OneWayKVStages(
            page_capacities(runtime.kv_cache_stages, allocator.page_size), self._return
        )
        self.reserve = 0
        self.probe = None
        # A long request leaves cached prefixes physically spread across the pool,
        # so the KV tail the expert pool loaned out can stay pinned by cold prefix
        # cache even after traffic drops two stages down. The scheduler wires a
        # bounded eviction hook so the tail can be reclaimed on that transition;
        # without a hook the behaviour is the previous silent no-op.
        configured = getattr(runtime, "kv_tail_evict_tokens", None)
        self.tail_evict_limit_tokens = (
            self.policy.capacities[-1] - self.policy.capacities[-2]
            if configured is None
            else int(configured)
        )
        self.evict_hook = None
        self.tail_evict_tokens = 0
        self.tail_evict_attempts = 0
        self.tail_evict_attempts_allowed = 3
        self.free_snapshot = torch.empty(
            allocator.num_pages + 1,
            dtype=torch.bool,
            device="cpu",
            pin_memory=allocator.device != "cpu",
        )
        self.free_mask = torch.empty(
            allocator.num_pages + 1, dtype=torch.bool, device=allocator.device
        )
        runtime.kv_loan = self
        runtime.kv_reborrow_probe = False
        allocator.backed_num_pages = self.policy.capacity // allocator.page_size
        allocator.free_pages = torch.arange(
            1,
            allocator.backed_num_pages + 1,
            dtype=torch.int64,
            device=allocator.device,
        )

    def _return(self, index):
        import torch

        a = self.allocator
        # Existing result completion covers tail invalidation; release before KV grows.
        if not self.runtime.staged_trim_enqueued:
            raise RuntimeError(
                "KV return attempted before an existing result completion boundary"
            )
        fence = getattr(self.runtime, "staged_trim_done", None)
        if fence is not None and not fence.query():
            raise RuntimeError("KV trim was not covered by result completion")
        if getattr(self.runtime, "hybrid", None) is not None or getattr(
            self.runtime, "preserve_cache_on_resize", False
        ):
            self.runtime.complete_staged_trim()
        self.runtime.staged_pool.release_stage(index)
        capacity = self.policy.capacities[index]
        # Establish all intermediate KV boundaries for later exact shrink.
        for stage in range(self.policy.index + 1, index + 1):
            self.pool._post_capture_owner.finalize(self.policy.capacities[stage])
        new_pages = capacity // a.page_size
        tail = torch.arange(
            a.backed_num_pages + 1, new_pages + 1, dtype=torch.int64, device=a.device
        )
        a.free_pages = torch.cat((a.free_pages, tail))
        a.backed_num_pages = new_pages
        self.runtime.pending_kv_stage = None
        self.runtime.staged_trim_enqueued = False
        # The eviction budget belongs to one long->short transition, not to the
        # process: a cycle that could not clear the tail must not silence the
        # hook for every later request that hands the same bytes back.
        self.tail_evict_attempts = 0
        logger.info(
            "Qwen38 returned expert loan: KV=%d expert slots=%d",
            capacity,
            self.runtime.expert_loan_budgets[index],
        )

    def used_tokens(self):
        a = self.allocator
        return (
            a.backed_num_pages - len(a.free_pages) - a.num_staged_pages
        ) * a.page_size

    def fits_allocation(self, reserve):
        a = self.allocator
        free = (len(a.free_pages) + a.num_staged_pages) * a.page_size
        return reserve <= free

    def prepare(self, reserve):
        # Actual allocation never waits or unmaps. Leave return to copy_done.
        if self.policy.poisoned:
            raise RuntimeError("KV stage return previously failed")
        # Allocation kernels independently check the physical free-page list.
        # Short host-prefix loads may fail and fall back to chunked recompute.

    def before_forward(self, reserve):
        if self.runtime.pending_kv_stage is not None:
            return
        # The scheduler reserves the next prefill chunk even during decode.
        # Historical large batches must not permanently prevent reborrowing.
        self.reserve = reserve
        used = self.used_tokens()
        # Current allocation is already charged. Return before the next batch.
        index = self.policy.index
        while index < 2 and used + reserve >= self.policy.capacities[index]:
            index += 1
        if index > self.policy.index:
            self.runtime.pending_kv_stage = index
        elif (
            self.policy.index > 0
            and used + 2 * reserve < self.policy.capacities[self.policy.index - 1]
        ):
            self.runtime.kv_reborrow_probe = True

    def reclaim_host_cached_tail(self, tree_cache, reserve):
        """Demote completed CPU-backed prefixes before matching the next batch.

        This only frees GPU cache rows. Physical backing and expert layout
        changes still wait for the existing forward/result snapshot boundary.
        """
        if self.policy.index == 0 or self.runtime.pending_kv_stage is not None:
            return
        demand = tree_cache.full_protected_size() + 2 * reserve
        target = next(
            (
                cap
                for cap in self.policy.capacities[: self.policy.index]
                if demand < cap
            ),
            None,
        )
        if target is not None:
            reclaimed = tree_cache.reclaim_device_tail(target)
            if reclaimed:
                import torch

                # Freed tail pages must not be the next allocation's first
                # choice, or a short request immediately pins the tail again.
                a = self.allocator
                a.free_pages, _ = torch.sort(a.get_all_free_pages())
                a.staged_pages = []
                a.num_staged_pages = 0

    def enqueue_reborrow_probe(self):
        import torch

        if not self.runtime.kv_reborrow_probe or self.probe is not None:
            return
        # Snapshot only pages already free before result processing. Later
        # frees are harmless; cached prefixes and in-flight pages remain false.
        self.free_mask.zero_()
        pages = self.allocator.get_all_free_pages()
        self.free_mask[pages] = True
        self.free_snapshot.copy_(self.free_mask, non_blocking=True)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.allocator.device))
        self.probe = done

    def _free_page_mask(self):
        """Fresh CPU page mask, used after eviction returned pages to the pool."""
        import torch

        a = self.allocator
        mask = torch.zeros(a.backed_num_pages + 1, dtype=torch.bool, device=a.device)
        mask[a.get_all_free_pages()] = True
        return mask.cpu()

    def _evict_tail_resident_prefixes(self, free, backed):
        """Evict cold prefix cache so the physical KV tail can be re-loaned.

        Only pages above the next lower stage are counted, which bounds the
        eviction by the bytes the shrink would reclaim and keeps a plain
        ``2 * reserve`` shortfall from evicting prefixes that are not in the way.
        """
        hook = getattr(self, "evict_hook", None)
        if hook is None or getattr(self, "tail_evict_limit_tokens", 0) <= 0:
            return False
        if getattr(self, "tail_evict_attempts", 0) >= getattr(
            self, "tail_evict_attempts_allowed", 0
        ):
            return False
        # The shrink also keeps a 2 * reserve margin below the next stage, so the
        # pages in the way start there rather than at the boundary itself. Nothing
        # above it means cold prefix cache is not what blocks the re-loan.
        first_page = (
            self.policy.capacities[self.policy.index - 1] - 2 * self.reserve
        ) // self.allocator.page_size
        blocked = sum(
            1 for page in range(max(first_page, 1), backed + 1) if not free[page]
        )
        if blocked == 0:
            return False
        tokens = min(blocked * self.allocator.page_size, self.tail_evict_limit_tokens)
        self.tail_evict_attempts = getattr(self, "tail_evict_attempts", 0) + 1
        try:
            hook(tokens)
        except BaseException:
            # Losing the loan is an optimisation miss, not a correctness failure;
            # only the KV return itself is allowed to poison serving.
            logger.exception("Qwen38 KV tail eviction failed; keeping the stage")
            return False
        self.tail_evict_tokens = getattr(self, "tail_evict_tokens", 0) + tokens
        logger.info(
            "Qwen38 evicted cold prefix cache for the KV tail: tokens=%d blocked_pages=%d",
            tokens,
            blocked,
        )
        return True

    def _reborrow(self):
        import torch

        if self.probe is None:
            return
        if not self.probe.query():
            raise RuntimeError(
                "KV free-page snapshot was not covered by result completion"
            )
        free = self.free_snapshot.tolist()
        backed = self.allocator.backed_num_pages
        self.probe = None
        self.runtime.kv_reborrow_probe = False
        target = self.policy.index
        # Shrinking unmaps the tail, so it needs every page above the next lower
        # stage free. Demand alone can look small while cold cached prefixes hold
        # those pages; evict a bounded amount and re-check instead of giving up.
        for _ in range(2):
            highest = max(
                (page for page in range(1, backed + 1) if not free[page]), default=0
            )
            demand = (
                max(highest * self.allocator.page_size, self.used_tokens())
                + 2 * self.reserve
            )
            target = next(
                (i for i, cap in enumerate(self.policy.capacities) if demand < cap),
                self.policy.index,
            )
            if target < self.policy.index:
                break
            # Without the scheduler's eviction hook the shrink is simply refused,
            # exactly as before the hook existed; dropping someone else's prefix
            # cache is never something the allocator decides on its own.
            if getattr(self, "evict_hook", None) is None:
                break
            if not self._evict_tail_resident_prefixes(free, backed):
                break
            free = self._free_page_mask().tolist()
        if target >= self.policy.index:
            return
        self.tail_evict_attempts = 0
        a = self.allocator
        capacity = self.policy.capacities[target]
        new_pages = capacity // a.page_size
        # The tail was wholly free in the snapshot and no next batch has
        # started. Merge later result frees, sort, remove the known tail count.
        all_free, _ = torch.sort(a.get_all_free_pages())
        a.free_pages = all_free[: len(all_free) - (backed - new_pages)]
        a.staged_pages = []
        a.num_staged_pages = 0
        self.pool._post_capture_owner.shrink_capacity(capacity)
        self.runtime.staged_pool.grow_stage(target)
        a.backed_num_pages = new_pages
        self.policy.index = target
        self.runtime.grow_staged_layout(target)
        logger.info(
            "Qwen38 reborrowed KV tail: KV=%d expert slots=%d",
            capacity,
            self.runtime.expert_loan_budgets[target],
        )

    def after_result(self):
        if self.runtime.kv_reborrow_probe:
            try:
                self._reborrow()
            except BaseException:
                self.policy.poisoned = True
                self.runtime.poisoned = True
                raise
        index = self.runtime.pending_kv_stage
        if index is None or not self.runtime.staged_trim_enqueued:
            return
        # Result processor already synchronized copy_done. New forwards only
        # use the retained prefixes: tail invalidation was enqueued before D2H.
        try:
            self._return(index)
        except BaseException:
            self.policy.poisoned = True
            raise
        self.policy.index = index


def attach_staged_loan(allocator, kvcache):
    runtime = getattr(kvcache, "staged_expert_runtime", None)
    if runtime is None:
        return None
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    if type(allocator) is not PagedTokenToKVPoolAllocator:
        raise ValueError("kv_cache_stages requires the standard paged allocator")
    return StagedKVLoan(allocator, kvcache.full_kv_pool, runtime)
