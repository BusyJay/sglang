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
            ("route_counts", "Qwen38CacheRoute::run_counts"),
            ("plan", "Qwen38CachePlan::run"),
            ("scatter", "Qwen38CacheScatter::run"),
            ("alpha_scatter", "Qwen38CacheAlphaScatter::run"),
            ("decode_plan", "Qwen38CacheDecodePlan::run"),
            ("decode_plan_mailbox", "Qwen38CacheDecodePlan::run_mailbox"),
            ("decode_plan_parallel", "Qwen38CacheDecodePlan::run_parallel"),
            (
                "decode_plan_parallel_early",
                "Qwen38CacheDecodePlan::run_parallel_early<false>",
            ),
            (
                "decode_plan_parallel_fc1",
                "Qwen38CacheDecodePlan::run_parallel_early<true>",
            ),
            ("decode_plan_layered", "Qwen38CacheDecodePlan::run_layered"),
            ("decode_plan_layered_lfu", "Qwen38CacheDecodePlan::run_layered_lfu"),
            ("packed_scatter", "Qwen38PackedScatter::run"),
            ("mailbox_register", "Qwen38MailboxRegister::run"),
            ("mailbox_unregister", "Qwen38MailboxUnregister::run"),
            ("mailbox_fill", "Qwen38MailboxFill::run"),
            ("mailbox_fill_grid", "Qwen38MailboxFill::run_config"),
            ("registered_copy", "Qwen38RegisteredCopy::run"),
            ("csf_headers", "sglang::Qwen38CsfHeaders::run"),
        ],
    )


def cache_route(topk_ids, lookup, remapped, counts):
    if topk_ids.dtype != torch.int32:
        raise ValueError("Qwen3.8 JIT routing requires int32 IDs")
    _module().route(topk_ids, lookup, remapped, counts)


def csf_patch_headers(heads, bank_delta, lower, upper):
    """Bind a tiered slab's immutable exception bank after a prefill upload."""
    _module().csf_headers(heads, bank_delta, lower, upper)


def cache_route_counts(topk_ids, counts):
    """Count experts and invalid routes without materializing remapped IDs."""
    _module().route_counts(topk_ids, counts)


def cache_alpha_scatter(g1, g2, slot_g1, slot_g2, plan):
    """Fill both slot alpha tables from the device miss plan in one launch."""
    _module().alpha_scatter(g1, g2, slot_g1, slot_g2, plan)


def cache_decode_plan(
    ids,
    remapped,
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
    """Fuse one-token histogram, policy and remap with the reference policy."""
    _module().decode_plan(
        ids,
        remapped,
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


def cache_scatter(cache, arena, keys, counts):
    """Scatter routed resident rows without allocating gathered expert weights."""
    _module().scatter(
        cache.view(torch.uint8).reshape(cache.shape[0], -1),
        arena.view(torch.uint8).reshape(arena.shape[0], -1),
        keys,
        counts,
    )


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


def packed_scatter(scratch, descriptors):
    """Internal byte scatter with immutable live-target address descriptors."""
    _module().packed_scatter(scratch, descriptors)


@cache_once
def _partition_module():
    return load_jit(
        "qwen38_partition",
        cuda_files=["moe/qwen38_partition.cuh"],
        cuda_wrappers=[("partition", "qwen38_partition")],
    )


def cache_partition_routes(ids, lookup, slot_limit, *, out=None):
    """Return cache-slot and arena-expert routes; inactive routes are -1."""
    cache, arena = (
        out if out is not None else (torch.empty_like(ids), torch.empty_like(ids))
    )
    _partition_module().partition(ids, lookup, cache, arena, slot_limit)
    return cache, arena


def cache_decode_plan_layered(*args):
    """Per-layer LFU plan with exact calibrated CPU bypass and mapped publication."""
    return _module().decode_plan_layered(*args)


def cache_decode_plan_layered_lfu(*args):
    """Layer-local LFU admission with CPU-inclusive expert heat and a count gate."""
    return _module().decode_plan_layered_lfu(*args)
