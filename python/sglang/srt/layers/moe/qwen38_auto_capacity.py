# SPDX-License-Identifier: Apache-2.0
"""Resolve expert capacity from physical free memory before graph capture."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def auto_kv_stages(context_length, chunk_size, page_size=64):
    """Keep the established 80K/160K/220K proportions and a full chunk reserve."""
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("automatic KV stages require a positive context length")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("automatic KV stages require a positive prefill chunk")
    final = (context_length + page_size - 1) // page_size * page_size
    first = max(
        (context_length * 4 + 11 * page_size - 1) // (11 * page_size) * page_size,
        (2 * chunk_size // page_size + 1) * page_size,
    )
    middle = max(
        first + page_size,
        (context_length * 8 + 11 * page_size - 1) // (11 * page_size) * page_size,
    )
    if middle >= final:
        raise ValueError(
            "context is too small for three KV stages at this prefill chunk"
        )
    return [first, middle, final]


def forecast_loan_budgets(base, maximum, loans, physical_bytes):
    """Include VMM page rounding when finding each stage's executable slots."""
    base_bytes = physical_bytes(base)
    budgets = []
    for loan in loans:
        lo, hi = base, maximum
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if physical_bytes(mid) - base_bytes <= loan:
                lo = mid
            else:
                hi = mid - 1
        budgets.append(lo)
    return budgets


def plan_auto_capacity(base, maximum, current, loans, physical_bytes, free, reserve):
    """Maximize the base budget while reserving actual INITIAL-stage headroom.

    Counting just the base delta is insufficient: rounding can change how much
    of the KV loan is consumed after rebasing. Charge the complete backed stage.
    """
    if free < reserve:
        raise ValueError(
            f"automatic expert cache needs {reserve} free bytes for runtime headroom; "
            f"only {free} remain even at the minimum cache. Reduce KV context."
        )
    ceiling = physical_bytes(current) + free - reserve
    lo, hi = base, maximum
    while lo < hi:
        mid = (lo + hi + 1) // 2
        budgets = forecast_loan_budgets(mid, maximum, loans, physical_bytes)
        if physical_bytes(budgets[0]) <= ceiling:
            lo = mid
        else:
            hi = mid - 1
    return lo, forecast_loan_budgets(lo, maximum, loans, physical_bytes)


def maybe_resolve_auto_capacity(model_runner):
    """Run once after target/draft pools exist, before any captured addresses."""
    runtime = getattr(model_runner.model, "qwen38_expert_runtime", None)
    if runtime is None or not getattr(runtime, "auto_cache_budget", False):
        return
    if getattr(runtime, "auto_cache_plan", None) is not None:
        return
    if runtime.run_invocations or runtime.warm_decode_cache:
        raise RuntimeError(
            "automatic expert capacity must precede all expert execution"
        )
    if getattr(runtime, "hybrid", None) is not None:
        raise ValueError(
            "automatic expert capacity currently requires GPU-only experts"
        )

    import torch

    from sglang.srt.runtime_context import get_schedule

    torch.cuda.synchronize(runtime.device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(runtime.device)
    # This reserve includes graph capture, JIT loading, GDN temporaries and QSA
    # index expansion. Never turn an aggressive fraction into near-zero slack.
    reserve = max(int(total * (1 - get_schedule().mem_fraction_static)), 2 << 30)
    pool = runtime.staged_pool
    base, predicted = plan_auto_capacity(
        pool.base_budget,
        pool.maximum_budget,
        pool.budgets[pool.stage_index],
        pool.loan_bytes,
        pool.physical_bytes,
        free,
        reserve,
    )
    old = list(pool.budgets)
    pool.rebase_before_execution(base)
    if pool.budgets != predicted:
        raise AssertionError("automatic cache forecast differs from VMM allocation")
    runtime.expert_loan_budgets = list(pool.budgets)
    runtime.initialize_staged_layout()
    torch.cuda.synchronize(runtime.device)
    after, _ = torch.cuda.mem_get_info(runtime.device)
    runtime.auto_cache_plan = dict(
        base_slots=base,
        expert_loan_budgets=list(pool.budgets),
        provisional_budgets=old,
        free_before_bytes=free,
        free_after_bytes=after,
        reserve_bytes=reserve,
        physical_bytes=pool.physical_bytes(pool.budgets[0]),
    )
    logger.info("Qwen38 automatic expert capacity: %s", runtime.auto_cache_plan)
