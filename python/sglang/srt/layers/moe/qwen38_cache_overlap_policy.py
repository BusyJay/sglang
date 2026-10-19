# SPDX-License-Identifier: Apache-2.0
"""Conservative cache/arena selection from the existing CPU prefill plan.

The mixed-path bounds are calibrated for SM120 Qwen3.8 NVFP4, H=2560,
I=640, top-k=10. No per-token GPU status is read back for this policy.
"""


def select_prefill_cache_path(
    counts, resident, arena_residents, *, hidden, width, top_k, w4a4=False
):
    """Return arena, cache or split without changing cache ownership.

    Cache misses already resident in arena do not represent host transfer time
    available to hide computation. A completely populated arena also avoids the
    cache-to-arena copy and should keep its original single dispatch.
    """
    needed = {expert for expert, count in enumerate(counts[:-1]) if count}
    if arena_residents is None or needed <= arena_residents or not resident:
        return "arena"
    misses = needed - resident.keys()
    if not misses:
        return "cache"
    host_misses = len(misses - arena_residents)
    if not w4a4 or (hidden, width, top_k) != (2560, 640, 10):
        return "arena"
    hit_routes = sum(counts[expert] for expert in resident)
    routes = sum(counts[:-1])
    # Small-batch mixed gains were not repeatable across sweeps. Large batches
    # amortize dispatch when at least three real fills can hide cached work.
    if routes >= 10240 and hit_routes >= 1024 and host_misses >= 3:
        return "split"
    return "arena"
