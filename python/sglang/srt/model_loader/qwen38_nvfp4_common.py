# SPDX-License-Identifier: Apache-2.0
"""Dependency-light Qwen3.8 model constraints and checkpoint filters."""

from __future__ import annotations

import re
from typing import Iterable, Iterator

QWEN38_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
QWEN38_EXPECTED = {
    "num_hidden_layers": 48,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "hidden_size": 2560,
    "moe_intermediate_size": 640,
}


def is_routed_expert_weight(name: str) -> bool:
    return (
        bool(re.search(r"(?:^|\.)layers\.\d+\.mlp\.experts\.", name))
        and "mtp" not in name
    )


def should_skip_checkpoint_tensor(name: str, *, language_model_only: bool) -> bool:
    return is_routed_expert_weight(name) or (
        language_model_only and (name.startswith("visual.") or "model.visual." in name)
    )


def filtered_safetensors_weights_iterator(
    files: Iterable[str], *, language_model_only: bool = True
) -> Iterator[tuple[str, object]]:
    """Check names before get_tensor(), so skipped 63+ GiB experts are untouched."""
    from safetensors import safe_open

    for filename in files:
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if should_skip_checkpoint_tensor(
                    name, language_model_only=language_model_only
                ):
                    continue
                yield name, handle.get_tensor(name)


def validate_qwen38_model_config(hf_config) -> list[str]:
    architectures = list(getattr(hf_config, "architectures", []) or [])
    text_config = getattr(hf_config, "text_config", hf_config)
    errors: list[str] = []
    if QWEN38_ARCHITECTURE not in architectures:
        errors.append(
            f"architectures must contain {QWEN38_ARCHITECTURE!r}, got {architectures!r}"
        )
    for field, expected in QWEN38_EXPECTED.items():
        actual = getattr(text_config, field, None)
        if actual != expected:
            errors.append(f"{field} must be {expected}, got {actual!r}")
    return errors
