# SPDX-License-Identifier: Apache-2.0
"""Validate per-layer expert capacity without torch or CUDA dependencies."""


def normalize_layer_slots(default, overrides, num_layers, num_experts, minimum=1):
    if type(default) is not int or not minimum <= default <= num_experts:
        raise ValueError("invalid default expert-cache capacity")
    if overrides is None:
        return [default] * num_layers
    if not isinstance(overrides, list) or len(overrides) != num_layers:
        raise ValueError("layer_cache_slots must be a list covering every layer")
    if any(
        type(value) is not int or not minimum <= value <= num_experts
        for value in overrides
    ):
        raise ValueError("invalid per-layer expert-cache capacity")
    return list(overrides)


def resolve_cache_slots(config, num_layers, num_experts, minimum=10):
    """Resolve a total budget to an exact, deterministic initial layout."""
    total = config.get("cache_slots_total")
    if total is None:
        return normalize_layer_slots(
            config.get("cache_slots_per_layer", 56),
            config.get("layer_cache_slots"),
            num_layers,
            num_experts,
            minimum=minimum,
        )
    if "cache_slots_per_layer" in config or "layer_cache_slots" in config:
        raise ValueError(
            "cache_slots_total cannot be combined with per-layer capacity options"
        )
    if (
        type(total) is not int
        or not num_layers * minimum <= total <= num_layers * num_experts
    ):
        raise ValueError(
            f"cache_slots_total must be an integer in [{num_layers * minimum}, {num_layers * num_experts}]"
        )
    base, remainder = divmod(total, num_layers)
    return [base + int(layer < remainder) for layer in range(num_layers)]


def slots_from_byte_budget(config, manifest):
    """Size opaque expert slots from actual pack strides, including alphas."""
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

    budget = config.get("cache_budget_bytes")
    if budget is None:
        return config
    if type(budget) is not int or budget <= 0:
        raise ValueError("cache_budget_bytes must be a positive integer")
    if any(
        name in config
        for name in ("cache_slots_total", "cache_slots_per_layer", "layer_cache_slots")
    ):
        raise ValueError(
            "cache_budget_bytes cannot be combined with slot capacity options"
        )
    rows = {
        sum(records[n].nbytes for n in REQUIRED_TENSORS) // manifest.num_experts
        for records in manifest.tensors.values()
    }
    if len(rows) != 1:
        raise ValueError("byte budget requires uniform expert row sizes")
    total = min(budget // rows.pop(), manifest.num_layers * manifest.num_experts)
    return {**config, "cache_slots_total": total}
