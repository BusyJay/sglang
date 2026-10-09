# SPDX-License-Identifier: Apache-2.0
"""Startup constraints for Qwen3.8 Flash-Next NVFP4 expert offload."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

logger = logging.getLogger(__name__)


def handle_qwen38_nvfp4_cache(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if cfg.load_format != "qwen38_nvfp4_cache":
        return

    errors = []
    for label, option, size in (
        ("tensor", "--tp-size", cfg.tp_size),
        ("data", "--dp-size", cfg.dp_size),
        ("expert", "--ep-size", cfg.ep_size),
        ("pipeline", "--pp-size", cfg.pp_size),
    ):
        if size != 1:
            errors.append(f"{label} parallelism ({option}) must be 1, got {size}")
    if cfg.enable_waterfill:
        errors.append("--enable-waterfill is incompatible with qwen38_nvfp4_cache")
    if cfg.enforce_shared_experts_fusion:
        errors.append(
            "--enforce-shared-experts-fusion is incompatible with " "qwen38_nvfp4_cache"
        )
    if cfg.moe_a2a_backend != "none":
        errors.append("qwen38_nvfp4_cache requires --moe-a2a-backend none")
    if cfg.cpu_offload_gb > 0 or cfg.offload_group_size > 0:
        errors.append("generic layer offload is incompatible with the PLE host table")
    if cfg.speculative_algorithm is not None:
        errors.append(
            "Qwen3.8 cached MTP execution is not validated; disable speculation"
        )
    if errors:
        raise ValueError(
            "Invalid qwen38_nvfp4_cache configuration:\n- " + "\n- ".join(errors)
        )

    declare_resolution(
        server_args,
        "handle_qwen38_nvfp4_cache",
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        language_model_only=True,
        ple_offload_embedding=True,
        max_running_requests=1,
        moe_runner_backend="flashinfer_cutlass",
    )
    logger.info(
        "qwen38_nvfp4_cache selected: language-model-only, eager execution, "
        "single-rank parallelism and FlashInfer CUTLASS are required"
    )
