# SPDX-License-Identifier: Apache-2.0
"""Partition prefill routes without copying expert weights between pools."""

import torch


def partition_prefill_routes(
    ids: torch.Tensor,
    weights: torch.Tensor,
    cache_experts: dict[int, int],
    arena_experts: list[int],
    *,
    lookup: torch.Tensor | None = None,
    slot_limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Keep original weights and mark inactive routes with the -1 sentinel.

    Reusing an active expert as zero-weight padding creates duplicate expert
    IDs within a token. CUTLASS's routing permutation can then overwrite the
    active contribution. Both backends skip -1 routes without reading weights.
    No renormalization is performed.
    """
    if lookup is not None:
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_partition_routes

        cache_ids, arena_ids = cache_partition_routes(ids, lookup, slot_limit)
        # -1 routes are skipped by both backends. Keep original router weights;
        # constructing two zero-masked weight tensors adds no useful information.
        return (
            cache_ids,
            weights,
            arena_ids if arena_experts else None,
            weights if arena_experts else None,
        )
    if ids.is_cuda:
        raise ValueError("CUDA partition requires the runtime GPU lookup")
    if not cache_experts:
        raise ValueError("direct cache partition requires resident experts")
    max_expert = max(max(cache_experts), max(arena_experts, default=-1))
    lookup = torch.full((max_expert + 1,), -1, dtype=ids.dtype, device=ids.device)
    expert_ids = torch.tensor(list(cache_experts), dtype=ids.dtype, device=ids.device)
    slots = torch.tensor(
        list(cache_experts.values()), dtype=ids.dtype, device=ids.device
    )
    lookup[expert_ids] = slots
    valid = (ids >= 0) & (ids <= max_expert)
    mapped = lookup[ids.clamp(0, max_expert).long()]
    hit = valid & (mapped >= 0)
    cache_ids = torch.where(hit, mapped, -1).contiguous()
    cache_weights = torch.where(hit, weights, 0).contiguous()
    if not arena_experts:
        return cache_ids, cache_weights, None, None
    miss = valid & ~hit
    arena_ids = torch.where(miss, ids, -1).contiguous()
    arena_weights = torch.where(miss, weights, 0).contiguous()
    return cache_ids, cache_weights, arena_ids, arena_weights
