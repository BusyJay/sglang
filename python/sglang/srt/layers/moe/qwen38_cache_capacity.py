# SPDX-License-Identifier: Apache-2.0
"""Validate per-layer expert capacity without torch or CUDA dependencies."""


def normalize_layer_slots(default, overrides, num_layers, num_experts, minimum=1):
    if type(default) is not int or not minimum <= default <= num_experts:
        raise ValueError("invalid default expert-cache capacity")
    if overrides is None:
        return [default] * num_layers
    if not isinstance(overrides, list) or len(overrides) != num_layers:
        raise ValueError("layer_cache_slots must be a list covering every layer")
    if any(type(value) is not int or not minimum <= value <= num_experts for value in overrides):
        raise ValueError("invalid per-layer expert-cache capacity")
    return list(overrides)
