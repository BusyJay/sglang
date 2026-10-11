# SPDX-License-Identifier: Apache-2.0
"""JIT routing summaries and slot remapping for Qwen3.8 expert caches."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "qwen38_expert_cache",
        cuda_files=["moe/qwen38_expert_cache.cu"],
        cuda_wrappers=[
            ("route", "Qwen38CacheRoute::run"),
            ("plan", "Qwen38CachePlan::run"),
            ("scatter", "Qwen38CacheScatter::run"),
        ],
    )


def cache_route(topk_ids, lookup, remapped, counts):
    if topk_ids.dtype != torch.int32:
        raise ValueError("Qwen3.8 JIT routing requires int32 IDs")
    _module().route(topk_ids, lookup, remapped, counts)


def cache_scatter(cache, arena, keys, counts):
    """Scatter routed resident rows without allocating gathered expert weights."""
    _module().scatter(cache.view(torch.uint8).reshape(cache.shape[0], -1),
                      arena.view(torch.uint8).reshape(arena.shape[0], -1), keys, counts)


def cache_plan(
    counts,
    lookup,
    keys,
    frequency,
    ages,
    generation,
    clock,
    plan,
    decay=0.95,
    policy=0,
    protected_fraction=0.7,
):
    _module().plan(
        counts,
        lookup,
        keys,
        frequency,
        ages,
        generation,
        clock,
        plan,
        decay,
        policy,
        protected_fraction,
    )
