# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt-frequency seeds for an optional decode-cache warmup."""


def select_warm_experts(counts, capacity):
    """Select positive-frequency experts; ties use expert ID, not GPU topk order.

    Counts must omit the trailing invalid-route counter. This is a heuristic,
    not a prediction that prompt-hot experts will also be decode-hot.
    """
    if capacity <= 0 or any(count < 0 for count in counts):
        raise ValueError("invalid warmup capacity or route counts")
    return sorted(
        (expert for expert, count in enumerate(counts) if count > 0),
        key=lambda expert: (-counts[expert], expert),
    )[:capacity]
