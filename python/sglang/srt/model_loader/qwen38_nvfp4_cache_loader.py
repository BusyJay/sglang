# SPDX-License-Identifier: Apache-2.0
"""Loader front-end for Qwen3.8 Flash-Next host-cached NVFP4 experts.

This module owns validation and checkpoint filtering. The SM120 CUDA execution
path uses a synchronous reference runtime: this loader never falls back to
allocating all 512 routed experts per layer on the GPU.
"""

from __future__ import annotations

import logging
import os
import time

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe.qwen38_cache_capacity import resolve_cache_slots
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
    Qwen38CachedModelOptConfig,
)
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    DefaultModelLoader,
    _get_quantization_config,
    _initialize_model,
)
from sglang.srt.model_loader.qwen38_nvfp4_common import (
    cached_checkpoint_weight_files,
    filtered_safetensors_weights_iterator,
    resolve_cached_checkpoint,
    validate_cached_checkpoint_revision,
    validate_qwen38_model_config,
)
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    Qwen38ExpertPack,
    load_manifest,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


def default_cpu_hybrid(config, methods):
    """Opt into shared CPU/GPU compute for a calibrated threaded A16 runtime.

    Default to GPU-only unless explicitly enabled with CPU cost calibration
    and a supported configuration, avoiding CPU capability initialization
    and compilation when calibration is absent.
    """
    return (
        envs.SGLANG_ENABLE_QWEN38_CPU_HYBRID.get()
        and envs.SGLANG_QWEN38_CPU_COST_CALIBRATION.get() is not None
        and config.get("runtime_mode", "sync") == "threaded"
        and config.get("adaptive_cache", False)
        and config.get("cache_policy", "lfu") == "lfu"
        and config.get("prefill_arena", True)
        and config.get("adaptive_cache_mode", "rebalance") == "rebalance"
        and bool(methods)
        and all(method.use_w4a16 for method in methods)
    )


def default_prefill_direct_cache(config):
    """Direct cache reads are profitable on the calibrated prefill path.

    The auto policy only splits where the target-GPU sweeps showed a net gain,
    and the same option carries the CUTLASS workspace bound. Both need the
    threaded runtime, and reading the cache requires cache reuse, so the
    default follows those two preconditions instead of forcing a startup
    failure on configurations that cannot use them.
    """
    return config.get("runtime_mode", "sync") == "threaded" and config.get(
        "prefill_reuse_cache", True
    )


class Qwen38NvFp4CacheModelLoader(BaseModelLoader):
    """Validated loader boundary for the Qwen3.8 offload implementation."""

    def __init__(self, load_config) -> None:
        super().__init__(load_config)
        config = dict(load_config.model_loader_extra_config or {})
        from sglang.srt.layers.moe.qwen38_packed_transport import validate_packed_config

        validate_packed_config(
            config.get("packed_decode", "direct"),
            config.get("packed_prefill", "direct"),
            config.get("packed_min_bytes", 4096),
            config.get("packed_max_bytes", 64 << 20),
            config.get("packed_min_copies", 2),
        )
        if (
            any(
                config.get(k, "direct") != "direct"
                for k in ("packed_decode", "packed_prefill")
            )
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("packed transfer requires threaded runtime")
        unexpected = set(config) - {
            "manifest_path",
            "scale_compression",
            "checkpoint_revision",
            "sglang_commit",
            "cache_slots_per_layer",
            "cache_slots_total",
            "cache_budget_bytes",
            "kv_cache_stages",
            "prefill_arena",
            "runtime_mode",
            "cpu_hybrid_mode",
            "layered_cpu_max_experts",
            "stats_log_interval",
            "staging_buffers",
            "staging_buffer_bytes",
            "warm_decode_cache",
            "layer_cache_slots",
            "route_trace_path",
            "route_trace_limit",
            "adaptive_cache",
            "preallocate_gdn_proj",
            "adaptive_cache_window",
            "adaptive_cache_interval",
            "adaptive_cache_mode",
            "cache_policy",
            "cache_protected_fraction",
            "prefill_reuse_cache",
            "prefill_direct_cache",
            "decode_early_cache",
            "cache_overlap_policy",
            "preserve_cache_on_resize",
            "prefill_seed_cache",
            "async_pack_prefetch",
            "kv_tail_evict_tokens",
            "resident_alphas",
            "prefill_copy_mode",
            "fused_decode_plan",
            "decode_hit_fast_path",
            "decode_device_gate",
            "packed_decode",
            "packed_prefill",
            "packed_min_bytes",
            "packed_max_bytes",
            "packed_min_copies",
        }
        if unexpected:
            raise ValueError(
                f"unexpected Qwen3.8 cache loader options: {sorted(unexpected)}"
            )
        manifest_path = config.get("manifest_path") or os.getenv(
            "SGLANG_QWEN38_EXPERT_PACK_MANIFEST"
        )
        if not manifest_path:
            raise ValueError(
                "qwen38_nvfp4_cache requires manifest_path or "
                "SGLANG_QWEN38_EXPERT_PACK_MANIFEST"
            )
        if not isinstance(config.get("prefill_arena", True), bool):
            raise ValueError("prefill_arena must be a JSON boolean")
        if not isinstance(config.get("warm_decode_cache", False), bool):
            raise ValueError("warm_decode_cache must be a JSON boolean")
        if not isinstance(config.get("adaptive_cache", False), bool):
            raise ValueError("adaptive_cache must be a JSON boolean")
        if not isinstance(config.get("async_pack_prefetch", True), bool):
            raise ValueError("async_pack_prefetch must be a JSON boolean")
        for option in ("prefill_reuse_cache", "prefill_seed_cache"):
            if type(config.get(option, True)) is not bool:
                raise ValueError(f"{option} must be a JSON boolean")
            if option in config and config.get("runtime_mode", "sync") != "threaded":
                raise ValueError(f"{option} requires threaded mode")
        if config.get("cache_overlap_policy", "auto") not in ("auto", "always"):
            raise ValueError("cache_overlap_policy must be auto or always")
        if (
            "cache_overlap_policy" in config
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("cache_overlap_policy requires threaded mode")
        if type(config.get("prefill_direct_cache", None)) not in (type(None), bool):
            raise ValueError("prefill_direct_cache must be a JSON boolean")
        for option in ("decode_early_cache", "preserve_cache_on_resize"):
            if type(config.get(option, False)) is not bool:
                raise ValueError(f"{option} must be a JSON boolean")
            if (
                config.get(option, False)
                and config.get("runtime_mode", "sync") != "threaded"
            ):
                raise ValueError(f"{option} requires threaded mode")
        if config.get("prefill_direct_cache", None) is True and (
            config.get("runtime_mode", "sync") != "threaded"
            or not config.get("prefill_reuse_cache", True)
        ):
            raise ValueError(
                "prefill_direct_cache requires threaded mode and prefill_reuse_cache"
            )
        for option in ("resident_alphas", "fused_decode_plan"):
            if type(config.get(option, True)) is not bool:
                raise ValueError(f"{option} must be a JSON boolean")
            if option in config and config.get("runtime_mode", "sync") != "threaded":
                raise ValueError(f"{option} requires threaded mode")
        if config.get("prefill_copy_mode", "auto") not in ("auto", "dense", "sparse"):
            raise ValueError("prefill_copy_mode must be auto, dense or sparse")
        tail_evict = config.get("kv_tail_evict_tokens")
        if tail_evict is not None and (
            type(tail_evict) is not int
            or isinstance(tail_evict, bool)
            or tail_evict < 0
        ):
            raise ValueError("kv_tail_evict_tokens must be a non-negative integer")
        if (
            "prefill_copy_mode" in config
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("prefill_copy_mode requires threaded mode")
        if (
            config.get("adaptive_cache", False)
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("adaptive_cache requires runtime_mode=threaded")
        if (
            type(config.get("adaptive_cache_window", 512)) is not int
            or config.get("adaptive_cache_window", 512) < 16
        ):
            raise ValueError("adaptive_cache_window must be an integer >= 16")
        if (
            type(config.get("adaptive_cache_interval", 4096)) is not int
            or config.get("adaptive_cache_interval", 4096) < 0
        ):
            raise ValueError("adaptive_cache_interval must be a non-negative integer")
        if config.get("adaptive_cache_mode", "rebalance") not in (
            "rebalance",
            "observe",
            "sample",
        ):
            raise ValueError("adaptive_cache_mode must be rebalance, observe or sample")
        if config.get("cache_policy", "lfu") not in ("lfu", "slru"):
            raise ValueError("cache_policy must be lfu or slru")
        fraction = config.get("cache_protected_fraction", 0.7)
        if type(fraction) not in (int, float) or not 0 <= fraction < 1:
            raise ValueError("cache_protected_fraction must be in [0, 1)")
        if (
            config.get("cache_policy", "lfu") != "lfu"
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("SLRU requires runtime_mode=threaded")
        if config.get("warm_decode_cache", False) and (
            config.get("runtime_mode", "sync") not in ("device", "threaded")
            or not config.get("prefill_arena", True)
        ):
            raise ValueError(
                "decode warmup requires device/threaded mode and prefill arena"
            )
        if config.get("runtime_mode", "sync") not in (
            "sync",
            "async",
            "device",
            "threaded",
        ):
            raise ValueError("runtime_mode must be sync, async, device or threaded")
        if type(config.get("preallocate_gdn_proj", False)) is not bool:
            raise ValueError("preallocate_gdn_proj must be a JSON boolean")
        if type(config.get("decode_hit_fast_path", True)) is not bool:
            raise ValueError("decode_hit_fast_path must be a JSON boolean")
        if type(config.get("decode_device_gate", True)) is not bool:
            raise ValueError("decode_device_gate must be a JSON boolean")
        from sglang.srt.mem_cache.qwen38_kv_stages import validate_stages

        stages = validate_stages(config.get("kv_cache_stages"))
        if stages and (
            not config.get("adaptive_cache", False)
            or config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError(
                "kv_cache_stages requires adaptive_cache and threaded runtime"
            )
        limit = config.get("layered_cpu_max_experts", 1)
        if type(limit) is not int or not 0 <= limit <= 30:
            raise ValueError("layered_cpu_max_experts must be an integer in [0, 30]")
        if (
            "layered_cpu_max_experts" in config
            and config.get("cpu_hybrid_mode") != "layered_lfu"
        ):
            raise ValueError("layered_cpu_max_experts requires layered_lfu mode")
        mode = config.get("cpu_hybrid_mode")
        if mode is not None and mode not in (
            "off",
            "global",
            "layered",
            "layered_zero",
            "layered_probe",
            "layered_lfu",
        ):
            raise ValueError(
                "cpu_hybrid_mode must be off/global/layered/layered_zero/layered_probe/layered_lfu"
            )
        if (
            mode not in (None, "off")
            and config.get("runtime_mode", "sync") != "threaded"
        ):
            raise ValueError("cpu_hybrid_mode requires threaded runtime")
        if mode in ("layered", "layered_zero", "layered_probe", "layered_lfu") and (
            config.get("decode_early_cache", False)
            or not config.get("fused_decode_plan", True)
            or not config.get("decode_device_gate", True)
        ):
            raise ValueError(
                "layered CPU hybrid requires fused device gate without early cache"
            )
        self.config = config
        self.manifest = load_manifest(manifest_path)
        mode = config.get("scale_compression", "native")
        if mode not in ("native", "csf"):
            raise ValueError("scale_compression must be native or csf")
        compressed = bool(getattr(self.manifest, "scale_encoding", {}))
        if compressed != (mode == "csf"):
            raise ValueError("scale_compression must match the expert pack encoding")
        self.manifest.validate_cutlass_layout()
        self.manifest.assert_runtime_compatible(
            checkpoint_revision=config.get("checkpoint_revision"),
            sglang_commit=config.get("sglang_commit"),
        )
        from sglang.srt.layers.moe.qwen38_cache_capacity import slots_from_byte_budget

        self.initial_layer_slots = resolve_cache_slots(
            slots_from_byte_budget(config, self.manifest),
            self.manifest.num_layers,
            self.manifest.num_experts,
            minimum=10,
        )

    def download_model(self, model_config) -> None:
        # Base loader API name only: this resolves existing cache, never downloads.
        self._prepare_checkpoint(model_config)

    def _prepare_checkpoint(self, model_config):
        checkpoint_dir = resolve_cached_checkpoint(
            model_config.model_path,
            revision=getattr(model_config, "revision", None),
            cache_dir=self.load_config.download_dir,
        )
        validate_cached_checkpoint_revision(
            checkpoint_dir, model_config.hf_config, self.manifest.checkpoint_revision
        )
        return cached_checkpoint_weight_files(checkpoint_dir)

    def load_model(self, *, model_config, device_config):
        errors = validate_qwen38_model_config(model_config.hf_config)
        if errors:
            raise ValueError(
                "Invalid Qwen3.8 Flash-Next configuration:\n- " + "\n- ".join(errors)
            )
        text_config = getattr(
            model_config.hf_config, "text_config", model_config.hf_config
        )
        manifest_shape = {
            "num_hidden_layers": self.manifest.num_layers,
            "num_experts": self.manifest.num_experts,
            "num_experts_per_tok": self.manifest.top_k,
            "hidden_size": self.manifest.hidden_size,
            "moe_intermediate_size": self.manifest.intermediate_size,
        }
        mismatches = [
            f"{field}: pack={actual}, model={getattr(text_config, field, None)}"
            for field, actual in manifest_shape.items()
            if getattr(text_config, field, None) != actual
        ]
        if mismatches:
            raise ValueError(
                "Qwen3.8 pack/model shape mismatch: " + "; ".join(mismatches)
            )
        hf_config = model_config.hf_config
        remote_vision = bool(getattr(hf_config, "language_only", False))
        if remote_vision:
            if not getattr(hf_config, "skip_local_encoder_init", False):
                raise ValueError(
                    "Flash-Next EPD offload must disable local ViT initialization"
                )
        elif not getattr(hf_config, "language_model_only", False):
            raise ValueError(
                "qwen38_nvfp4_cache requires --language-model-only or remote --language-only"
            )

        weight_files = self._prepare_checkpoint(model_config)

        delegate = _get_quantization_config(model_config, self.load_config)
        if delegate is None or delegate.get_name() != "modelopt_mixed":
            raise ValueError(
                "qwen38_nvfp4_cache requires a serialized modelopt_mixed checkpoint"
            )
        gdn_value_heads = None
        if self.config.get("preallocate_gdn_proj", False):
            gdn_value_heads = getattr(
                model_config.hf_text_config, "linear_num_value_heads", None
            )
            if type(gdn_value_heads) is not int or gdn_value_heads <= 0:
                raise ValueError("preallocate_gdn_proj requires linear_num_value_heads")
        quant_config = Qwen38CachedModelOptConfig(
            delegate, self.manifest, gdn_value_heads=gdn_value_heads
        )
        target_device = torch.device(device_config.device)
        if target_device.type != "cuda" or torch.cuda.get_device_capability(
            target_device
        ) != (12, 0):
            raise ValueError("qwen38_nvfp4_cache requires an SM120 CUDA device")
        from sglang.srt.layers.moe.qwen38_startup_memory import (
            get_startup_memory_probe,
            memory_mark,
        )

        probe = get_startup_memory_probe(target_device)
        startup_cache = os.getenv("SGLANG_QWEN38_STARTUP_CACHE", "auto")
        if startup_cache not in ("auto", "off"):
            raise ValueError("SGLANG_QWEN38_STARTUP_CACHE must be auto or off")
        pack = None
        if startup_cache == "off":
            memory_mark(probe, "loader.before_pack", metadata={"config": self.config})
            pack = Qwen38ExpertPack(self.manifest)
            memory_mark(probe, "loader.after_pack")
        try:
            if pack is not None and self.config.get("async_pack_prefetch", True):
                pack.start_prefetch()
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    memory_mark(probe, "loader.before_model_construct")
                    text_config = model_config.hf_text_config
                    mapped_embedding = os.getenv(
                        "SGLANG_QWEN38_HOST_TOKEN_EMBEDDING", "1"
                    ) == "1" and not getattr(text_config, "tie_word_embeddings", False)
                    if mapped_embedding:
                        from sglang.srt.models.qwen38_host_embedding import (
                            find_token_embedding,
                        )
                        from sglang.srt.runtime_context import get_parallel

                        parallel = get_parallel()
                        mapped_embedding = (
                            parallel.tp_size == 1 and parallel.attn_tp_size == 1
                        )
                    if mapped_embedding:
                        text_config._qwen38_mapped_token_embedding = (
                            find_token_embedding(weight_files)
                        )
                    construct_started = time.perf_counter()
                    try:
                        model = _initialize_model(
                            model_config, self.load_config, quant_config
                        )
                    finally:
                        if mapped_embedding:
                            del text_config._qwen38_mapped_token_embedding
                    memory_mark(probe, "loader.after_model_construct", model=model)
                    logger.info(
                        "Qwen38 startup model construction: seconds=%.3f mapped_token_embedding=%s source=%s",
                        time.perf_counter() - construct_started,
                        mapped_embedding,
                        __file__,
                    )
                if remote_vision and getattr(model, "visual", None) is not None:
                    raise ValueError(
                        "Flash-Next offload worker unexpectedly allocated a local ViT"
                    )
                if startup_cache == "auto":
                    from sglang.srt.model_loader.qwen38_nvfp4_pack.startup_cache import (
                        prepare_startup_cache,
                    )

                    source_mode = prepare_startup_cache(
                        model, self.manifest, weight_files
                    )
                    memory_mark(
                        probe, "loader.before_pack", metadata={"config": self.config}
                    )
                    pack = Qwen38ExpertPack(
                        self.manifest,
                        source_mode=source_mode,
                        allow_shared_fallback=True,
                    )
                    memory_mark(probe, "loader.after_pack")
                    if self.config.get("async_pack_prefetch", True):
                        pack.start_prefetch()
                lazy_ple = (
                    getattr(
                        model_config.hf_text_config, "ple_offload_backend", "pinned"
                    )
                    == "lazy"
                    or getattr(model_config.hf_text_config, "ple_embedding_dtype", None)
                    == "nvfp4"
                )
                checkpoint_audit = {} if probe is not None else None
                weights_started = time.perf_counter()
                weights = filtered_safetensors_weights_iterator(
                    weight_files,
                    language_model_only=True,
                    lazy_ple=lazy_ple,
                    audit=checkpoint_audit,
                )
                if probe is None:
                    DefaultModelLoader.load_weights_and_postprocess(
                        model, weights, target_device
                    )
                else:
                    memory_mark(probe, "loader.before_load_weights", model=model)
                    DefaultModelLoader.load_weights_only(
                        model, weights, target_device, startup_memory_probe=probe
                    )
                    # Close generator handles and drop the iterator before observing
                    # the boundary, as the combined loader normally does on return.
                    if hasattr(weights, "close"):
                        weights.close()
                    del weights
                    memory_mark(
                        probe,
                        "loader.after_load_weights",
                        model=model,
                        metadata={"checkpoint_filter_counts": checkpoint_audit},
                    )
                    DefaultModelLoader.postprocess_weights(
                        model, target_device, startup_memory_probe=probe
                    )
                    memory_mark(probe, "loader.after_postprocess", model=model)
                logger.info(
                    "Qwen38 startup checkpoint load and postprocess: seconds=%.3f",
                    time.perf_counter() - weights_started,
                )
        except BaseException:
            if pack is not None:
                pack.close()
            raise

        from sglang.srt.layers.moe.qwen38_cache_runtime import (
            Qwen38SynchronousExpertRuntime,
        )
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        try:
            runtime_class = Qwen38SynchronousExpertRuntime
            runtime_options = {}
            if self.config.get("runtime_mode", "sync") == "async":
                from sglang.srt.layers.moe.qwen38_cache_async import (
                    Qwen38AsyncExpertRuntime,
                )

                runtime_class = Qwen38AsyncExpertRuntime
                runtime_options["staging_buffers"] = int(
                    self.config.get("staging_buffers", 4)
                )
            if self.config.get("runtime_mode", "sync") == "device":
                from sglang.srt.layers.moe.qwen38_cache_device import (
                    Qwen38DeviceExpertRuntime,
                )

                runtime_class = Qwen38DeviceExpertRuntime
                runtime_options["staging_buffers"] = int(
                    self.config.get("staging_buffers", 4)
                )
            if self.config.get("runtime_mode", "sync") == "threaded":
                from sglang.srt.layers.moe.qwen38_cache_threaded import (
                    Qwen38ThreadedExpertRuntime,
                )

                runtime_class = Qwen38ThreadedExpertRuntime
                runtime_options["staging_buffers"] = int(
                    self.config.get("staging_buffers", 4)
                )
                runtime_options["adaptive_cache"] = self.config.get(
                    "adaptive_cache", False
                )
                methods = [
                    module.quant_method
                    for module in model.modules()
                    if isinstance(
                        getattr(module, "quant_method", None),
                        Qwen38CachedNvFp4MoeMethod,
                    )
                ]
                mode = self.config.get("cpu_hybrid_mode")
                if mode is None:
                    mode = (
                        "global" if default_cpu_hybrid(self.config, methods) else "off"
                    )
                if mode not in (
                    "off",
                    "global",
                    "layered",
                    "layered_zero",
                    "layered_probe",
                    "layered_lfu",
                ):
                    raise ValueError(
                        "cpu_hybrid_mode must be off/global/layered/layered_zero/layered_probe/layered_lfu"
                    )
                if mode == "layered":
                    calibration = envs.SGLANG_QWEN38_LAYERED_CPU_CALIBRATION.get()
                    if calibration is None or not calibration.get("samples"):
                        logger.info(
                            "Layered CPU calibration absent; selecting per-layer GPU mode"
                        )
                        mode = "off"
                if mode.startswith("layered") and (
                    not methods
                    or not all(method.use_w4a16 for method in methods)
                    or not self.config.get("adaptive_cache", False)
                    or self.config.get("cache_policy", "lfu") != "lfu"
                    or not self.config.get("prefill_arena", True)
                ):
                    raise ValueError(
                        "layered CPU hybrid requires W4A16, adaptive LFU and prefill arena"
                    )
                runtime_options["cpu_hybrid"] = mode == "global"
                runtime_options["layered_cpu_mode"] = (
                    mode if mode.startswith("layered") else None
                )
                runtime_options["layered_cpu_max_experts"] = self.config.get(
                    "layered_cpu_max_experts", 1
                )
                runtime_options["adaptive_cache_window"] = self.config.get(
                    "adaptive_cache_window", 512
                )
                for name, default in (
                    ("adaptive_cache_interval", 4096),
                    ("adaptive_cache_mode", "rebalance"),
                    ("cache_policy", "lfu"),
                    ("cache_protected_fraction", 0.7),
                    ("prefill_reuse_cache", True),
                    ("prefill_direct_cache", default_prefill_direct_cache(self.config)),
                    ("decode_early_cache", False),
                    ("cache_overlap_policy", "auto"),
                    ("preserve_cache_on_resize", True),
                    ("prefill_seed_cache", True),
                    ("resident_alphas", True),
                    ("prefill_copy_mode", "auto"),
                    ("fused_decode_plan", True),
                    ("decode_hit_fast_path", True),
                    ("decode_device_gate", True),
                    ("packed_decode", "direct"),
                    ("packed_prefill", "direct"),
                    ("packed_min_bytes", 4096),
                    ("packed_max_bytes", 64 << 20),
                    ("packed_min_copies", 2),
                ):
                    runtime_options[name] = self.config.get(name, default)
            if runtime_options:
                runtime_options["staging_buffer_bytes"] = int(
                    self.config.get("staging_buffer_bytes", 64 << 20)
                )
            if self.config.get("runtime_mode", "sync") in ("device", "threaded"):
                runtime_options["warm_decode_cache"] = self.config.get(
                    "warm_decode_cache", False
                )
            runtime = runtime_class(
                pack,
                target_device,
                slots_per_layer=self.initial_layer_slots[0],
                prefill_arena=self.config.get("prefill_arena", True),
                stats_log_interval=int(self.config.get("stats_log_interval", 0)),
                layer_cache_slots=self.initial_layer_slots,
                route_trace_path=self.config.get("route_trace_path"),
                route_trace_limit=self.config.get("route_trace_limit", 10000),
                **runtime_options,
            )
            memory_mark(probe, "loader.after_runtime_construct", model=model)
            from sglang.srt.runtime_context import get_spec

            if (
                self.config.get("adaptive_cache", False)
                and get_spec().speculative_algorithm is not None
            ):
                runtime.adaptive_min_slots = (
                    self.manifest.top_k * get_spec().speculative_num_draft_tokens
                )
                if min(runtime.layer_cache_slots) < runtime.adaptive_min_slots:
                    raise ValueError(
                        "adaptive MTP cache must fit the full verify route bound in every layer"
                    )
            runtime.kv_cache_stages = self.config.get("kv_cache_stages")
            # Bounds how much cold prefix cache the KV loan may evict to reclaim a
            # tail it loaned to the expert pool. Absent means one stage gap; 0
            # keeps the historical behaviour of never evicting for a re-loan.
            runtime.kv_tail_evict_tokens = self.config.get("kv_tail_evict_tokens")
            reserve_started = time.perf_counter()
            runtime.reserve()
            budget = runtime.memory_budget()
            logger.info(
                "Qwen38 startup runtime reserve: seconds=%.3f decode_cache_MiB=%.2f "
                "prefill_arena_MiB=%.2f borrowed_prefill_arena_MiB=%.2f",
                time.perf_counter() - reserve_started,
                budget["decode_cache_bytes"] / (1 << 20),
                budget["prefill_arena_bytes"] / (1 << 20),
                budget.get("borrowed_prefill_arena_bytes", 0) / (1 << 20),
            )
            memory_mark(
                probe,
                "loader.after_runtime_reserve",
                model=model,
                metadata={"runtime_budget": runtime.memory_budget()},
            )
            bound = set()
            for module in model.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, Qwen38CachedNvFp4MoeMethod):
                    method.bind_runtime(runtime)
                    bound.add(method.layer_id)
            if bound != set(range(self.manifest.num_layers)):
                raise ValueError(
                    f"cache methods do not cover every layer: {sorted(bound)}"
                )
        except Exception:
            if "runtime" in locals() and hasattr(runtime, "close"):
                runtime.close()
            elif "runtime" in locals() and hasattr(runtime, "synchronize"):
                runtime.synchronize()
            pack.close()
            raise
        model.qwen38_expert_pack = pack
        model.qwen38_expert_runtime = runtime
        model.qwen38_cache_config = {
            "scale_compression": self.config.get("scale_compression", "native"),
            "slots_per_layer": runtime.slots_per_layer,
            "layer_cache_slots": runtime.layer_cache_slots,
            "cache_slots_total": sum(runtime.layer_cache_slots),
            "prefill_arena": runtime.prefill_arena,
        }
        return model.eval()
