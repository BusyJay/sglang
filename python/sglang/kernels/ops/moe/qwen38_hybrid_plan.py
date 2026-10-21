# SPDX-License-Identifier: Apache-2.0
"""Device LFU admission and calibrated CPU routing for small verify batches."""

import sysconfig

from sglang.kernels.jit.utils import cache_once, load_jit

MAX_ROUTES = 32
COST_COLUMNS = 6 + 2 * MAX_ROUTES


@cache_once
def _module():
    return load_jit(
        "qwen38_hybrid_plan",
        cpp_files=["moe/qwen38_mailbox.cc"],
        cuda_files=["moe/qwen38_hybrid_plan.cuh"],
        extra_include_paths=[sysconfig.get_path("include")],
        cpp_wrappers=[
            ("poll", "Qwen38MailboxPoll::run"),
            ("complete", "Qwen38MailboxComplete::run"),
        ],
        cuda_wrappers=[
            ("plan", "qwen38_hybrid_plan"),
            ("plan_indexed", "qwen38_hybrid_plan_indexed"),
            ("plan_stage", "qwen38_hybrid_plan_stage"),
            ("plan_stage_indexed", "qwen38_hybrid_plan_stage_indexed"),
            ("wait", "qwen38_hybrid_wait"),
            ("stage", "qwen38_hybrid_stage"),
            ("result", "qwen38_hybrid_result"),
            ("merge", "qwen38_hybrid_merge"),
        ],
    )


def hybrid_plan(
    ids,
    lookup,
    frequency,
    owners,
    keys,
    costs,
    cache_ids,
    cpu_ids,
    cache_plan,
    stats,
    layer,
    decay,
    admission,
):
    """Mutate device LFU and emit GPU/CPU routes and one registered-fill plan.

    Plans use the existing mailbox layout: status/count/unused/unused followed
    by (original expert, destination slot) pairs. Costs are prefiltered full
    branch samples, packed as tokens/GPU routes/CPU routes/CPU experts/rejected
    experts/CPU elapsed followed by two zero-padded route-count profiles.
    """
    _module().plan(
        ids,
        lookup,
        frequency,
        owners,
        keys,
        costs,
        cache_ids,
        cpu_ids,
        cache_plan,
        stats,
        layer,
        decay,
        admission,
    )


def hybrid_plan_stage(
    ids,
    lookup,
    frequency,
    owners,
    keys,
    costs,
    cache_ids,
    cpu_ids,
    cache_plan,
    stats,
    layer,
    decay,
    admission,
    x,
    weights,
    host_x,
    host_ids,
    host_weights,
    host_head,
    sequence,
):
    """Plan and publish a CPU request only if the device selected CPU routes."""
    _module().plan_stage(
        ids,
        lookup,
        frequency,
        owners,
        keys,
        costs,
        cache_ids,
        cpu_ids,
        cache_plan,
        stats,
        layer,
        decay,
        admission,
        x,
        weights,
        host_x,
        host_ids,
        host_weights,
        host_head,
        sequence,
    )


def hybrid_plan_indexed(*args):
    """Plan using the persistent layer/top heaps and free-slot ring."""
    _module().plan_indexed(*args)


def hybrid_plan_stage_indexed(*args):
    """Indexed planning and conditional CPU publication in one launch."""
    _module().plan_stage_indexed(*args)


def hybrid_wait(cpu_ids, completed, sequence, failed_address=0):
    """Introduce a result dependency only for device-selected CPU routes."""
    _module().wait(cpu_ids, completed, sequence, failed_address)


def hybrid_stage(
    x, ids, weights, host_x, host_ids, host_weights, host_head, layer, sequence
):
    """Publish only CPU-selected verify batches in the registered mapping."""
    _module().stage(
        x, ids, weights, host_x, host_ids, host_weights, host_head, layer, sequence
    )


def hybrid_result(cpu_ids, output, host_result):
    """Read the CPU contribution from mapped pages after the completion wait."""
    _module().result(cpu_ids, output, host_result)


def hybrid_merge(
    cpu_ids,
    gpu,
    completed,
    output,
    host_result,
    sequence,
    failed_address=0,
    completed_address=0,
):
    """Conditionally wait/read CPU mmap and return the merged BF16 result."""
    _module().merge(
        cpu_ids,
        gpu,
        completed,
        output,
        host_result,
        sequence,
        failed_address,
        completed_address,
    )
