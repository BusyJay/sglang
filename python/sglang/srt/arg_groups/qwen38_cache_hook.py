# SPDX-License-Identifier: Apache-2.0
"""Startup constraints for Qwen3.8 Flash-Next NVFP4 expert offload."""

from __future__ import annotations

import json
import logging
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

logger = logging.getLogger(__name__)


def resolve_qwen38_cuda_graph(server_args: Any) -> None:
    """Validate after JSON/convenience/legacy graph options are resolved."""
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    cfg = resolving_view(server_args)
    if cfg.load_format != "qwen38_nvfp4_cache":
        return
    graph = cfg.cuda_graph_config
    if graph.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "qwen38 offload requires --cuda-graph-backend-prefill=disabled"
        )
    if graph.decode.backend == Backend.DISABLED:
        return
    if graph.decode.backend != Backend.BREAKABLE:
        raise ValueError(
            "qwen38 offload decode graph requires backend=breakable, not full/tc_piecewise"
        )
    extra = cfg.model_loader_extra_config or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    if extra.get("runtime_mode", "sync") != "threaded":
        raise ValueError("qwen38 offload decode graph requires runtime_mode=threaded")
    if cfg.enable_torch_compile:
        raise ValueError(
            "qwen38 offload decode graph does not support --enable-torch-compile"
        )
    if graph.decode.bs not in (None, [1]) or graph.decode.max_bs not in (None, 1):
        raise ValueError("qwen38 offload decode graph supports only bs=[1], max_bs=1")
    declare_resolution(
        server_args,
        "resolve_qwen38_cuda_graph",
        cuda_graph_config=with_phase(graph, Phase.DECODE, bs=[1], max_bs=1),
    )
    logger.info(
        "qwen38 offload: segmented decode graph enabled "
        "(host transport and CUTLASS remain eager)"
    )


def handle_qwen38_nvfp4_cache(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if cfg.load_format != "qwen38_nvfp4_cache":
        return

    errors = []
    remote_vision = bool(cfg.language_only)
    if cfg.encoder_only:
        errors.append("encoder GPU must use the normal loader, not qwen38_nvfp4_cache")
    if remote_vision:
        if cfg.language_model_only:
            errors.append(
                "--language-only cannot be combined with --language-model-only"
            )
        if cfg.enable_adaptive_dispatch_to_encoder:
            errors.append(
                "Flash-Next offload EPD requires remote-only encoder dispatch"
            )
        if cfg.enable_multimodal is False:
            errors.append("Flash-Next offload EPD requires multimodal processing")
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
            "--enforce-shared-experts-fusion is incompatible with qwen38_nvfp4_cache"
        )
    if cfg.moe_a2a_backend != "none":
        errors.append("qwen38_nvfp4_cache requires --moe-a2a-backend none")
    if cfg.cpu_offload_gb > 0 or cfg.offload_group_size > 0:
        errors.append("generic layer offload is incompatible with the PLE host table")
    if cfg.speculative_algorithm is not None:
        _configure_mtp(server_args, cfg, errors, remote_vision)
    if errors:
        raise ValueError(
            "Invalid qwen38_nvfp4_cache configuration:\n- " + "\n- ".join(errors)
        )

    declare_resolution(
        server_args,
        "handle_qwen38_nvfp4_cache",
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        language_model_only=not remote_vision,
        ple_offload_embedding=True,
        max_running_requests=1,
        moe_runner_backend="flashinfer_cutlass",
    )
    logger.info(
        "qwen38_nvfp4_cache selected: %s, eager by default, "
        "single-rank parallelism and FlashInfer CUTLASS are required",
        "remote ViT (EPD), no local vision tower"
        if remote_vision
        else "language-model-only",
    )


def _configure_mtp(server_args, cfg, errors, remote_vision):
    """Opt-in, same-device NEXTN chain; target-only defaults stay unchanged."""
    if cfg.speculative_algorithm.upper() not in ("NEXTN", "EAGLE"):
        errors.append("qwen38 MTP supports only NEXTN (EAGLE alias)")
    if remote_vision:
        errors.append("qwen38 MTP currently requires pure text, without EPD")
    if cfg.speculative_draft_load_format != "qwen38_mtp":
        errors.append("qwen38 MTP requires --speculative-draft-load-format qwen38_mtp")
    if cfg.speculative_draft_model_path not in (None, cfg.model_path):
        errors.append("qwen38 MTP draft must use the target model path")
    if cfg.speculative_draft_model_revision not in (None, cfg.revision):
        errors.append("qwen38 MTP draft must use the target revision")
    extra = cfg.model_loader_extra_config or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    if extra.get("runtime_mode", "sync") != "threaded":
        errors.append("qwen38 MTP requires runtime_mode=threaded")
    for field in (
        "speculative_adaptive",
        "enable_multi_layer_eagle",
        "enable_lora",
        "enable_torch_compile",
    ):
        if getattr(cfg, field, False):
            errors.append(f"qwen38 MTP does not support {field}")
    if getattr(cfg, "speculative_token_map", None):
        errors.append("qwen38 MTP requires the full target vocabulary")
    if getattr(cfg, "disable_overlap_schedule", False):
        errors.append("qwen38 MTP requires the EAGLE v2 overlap scheduler")
    if getattr(cfg, "dtype", "auto") not in ("auto", "bfloat16"):
        errors.append("qwen38 MTP requires BF16 model dtype")
    if getattr(cfg, "mamba_ssm_dtype", None) not in (None, "bfloat16"):
        errors.append("qwen38 MTP requires BF16 GDN state")
    steps = cfg.speculative_num_steps if cfg.speculative_num_steps is not None else 1
    if steps not in (1, 2, 3):
        errors.append("qwen38 MTP supports speculative_num_steps=1/2/3")
    if cfg.speculative_eagle_topk not in (None, 1):
        errors.append("qwen38 MTP requires speculative_eagle_topk=1")
    if cfg.speculative_num_draft_tokens not in (None, steps + 1):
        errors.append("qwen38 MTP requires draft tokens = steps + 1 (2/3/4)")
    if cfg.speculative_moe_runner_backend not in (None, "auto", "triton"):
        errors.append("qwen38 MTP requires the Triton block-FP8 draft MoE runner")
    if extra.get("adaptive_cache", False):
        from sglang.srt.layers.moe.qwen38_cache_capacity import resolve_cache_slots

        try:
            resolve_cache_slots(extra, 48, 512, minimum=10 * (steps + 1))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        return
    declare_resolution(
        server_args,
        "handle_qwen38_nvfp4_cache",
        speculative_draft_model_path=cfg.model_path,
        speculative_draft_model_revision=cfg.revision,
        speculative_num_steps=steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=steps + 1,
        speculative_moe_runner_backend="triton",
        speculative_draft_kv_cache_dtype=getattr(
            cfg, "speculative_draft_kv_cache_dtype", None
        )
        or "bfloat16",
        mamba_ssm_dtype="bfloat16",
    )
