# SPDX-License-Identifier: Apache-2.0
"""Loader front-end for Qwen3.8 Flash-Next host-cached NVFP4 experts.

This module owns validation and checkpoint filtering. The SM120 CUDA execution
path uses a synchronous reference runtime: this loader never falls back to
allocating all 512 routed experts per layer on the GPU.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

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
    filtered_safetensors_weights_iterator,
    validate_qwen38_model_config,
)
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    Qwen38ExpertPack,
    load_manifest,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype


class Qwen38NvFp4CacheModelLoader(BaseModelLoader):
    """Validated loader boundary for the Qwen3.8 offload implementation."""

    def __init__(self, load_config) -> None:
        super().__init__(load_config)
        config = dict(load_config.model_loader_extra_config or {})
        unexpected = set(config) - {
            "manifest_path",
            "checkpoint_revision",
            "sglang_commit",
            "cache_slots_per_layer",
            "prefill_arena",
            "runtime_mode",
            "stats_log_interval",
            "staging_buffers",
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
        if config.get("runtime_mode", "sync") not in (
            "sync",
            "async",
            "device",
            "threaded",
        ):
            raise ValueError("runtime_mode must be sync, async, device or threaded")
        self.config = config
        self.manifest = load_manifest(manifest_path)
        self.manifest.validate_cutlass_layout()
        self.manifest.assert_runtime_compatible(
            checkpoint_revision=config.get("checkpoint_revision"),
            sglang_commit=config.get("sglang_commit"),
        )
        slots = int(config.get("cache_slots_per_layer", 56))
        if not 10 <= slots <= 512:
            raise ValueError("cache_slots_per_layer must be between top-k=10 and 512")

    def download_model(self, model_config) -> None:
        if not Path(model_config.model_path).is_dir():
            raise ValueError(
                "qwen38_nvfp4_cache currently requires a local checkpoint directory; "
                "prepare the runtime pack before starting the server"
            )

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
        if not getattr(model_config.hf_config, "language_model_only", False):
            raise ValueError("qwen38_nvfp4_cache requires --language-model-only")

        checkpoint_dir = Path(model_config.model_path)
        weight_files = sorted(glob.glob(str(checkpoint_dir / "*.safetensors")))
        if not weight_files:
            raise RuntimeError(
                f"no safetensors checkpoint files found in {checkpoint_dir}"
            )

        delegate = _get_quantization_config(model_config, self.load_config)
        if delegate is None or delegate.get_name() != "modelopt_mixed":
            raise ValueError(
                "qwen38_nvfp4_cache requires a serialized modelopt_mixed checkpoint"
            )
        quant_config = Qwen38CachedModelOptConfig(delegate, self.manifest)
        target_device = torch.device(device_config.device)
        if target_device.type != "cuda" or torch.cuda.get_device_capability(
            target_device
        ) != (12, 0):
            raise ValueError("qwen38_nvfp4_cache requires an SM120 CUDA device")
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
            weights = filtered_safetensors_weights_iterator(
                weight_files, language_model_only=True
            )
            DefaultModelLoader.load_weights_and_postprocess(
                model, weights, target_device
            )

        from sglang.srt.layers.moe.qwen38_cache_runtime import (
            Qwen38SynchronousExpertRuntime,
        )
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        pack = Qwen38ExpertPack(self.manifest)
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
            runtime = runtime_class(
                pack,
                target_device,
                slots_per_layer=int(self.config.get("cache_slots_per_layer", 56)),
                prefill_arena=self.config.get("prefill_arena", True),
                stats_log_interval=int(self.config.get("stats_log_interval", 0)),
                **runtime_options,
            )
            runtime.reserve()
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
            "slots_per_layer": runtime.slots_per_layer,
            "prefill_arena": runtime.prefill_arena,
        }
        return model.eval()
