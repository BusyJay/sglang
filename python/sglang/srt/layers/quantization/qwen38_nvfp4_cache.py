# SPDX-License-Identifier: Apache-2.0
"""Quantization boundary for host-cached Qwen3.8 NVFP4 routed experts."""

from __future__ import annotations

import re
from typing import Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)


def _capture_expert_output(_method, hidden_states, *_route_inputs):
    # Upstream segments have only been recorded, not replayed: their route IDs
    # are not valid yet. Do not enqueue a host transfer from capture-time data.
    # Warmup initializes CUTLASS; replay always runs the real expert boundary.
    return torch.zeros_like(hidden_states)


class Qwen38CachedNvFp4MoeMethod(FusedMoEMethodBase):
    """Creates no full-expert GPU parameters.

    The fixed-address cache buffers are installed after loading by the runtime.
    Keeping this method parameter-free is the critical loader invariant and is
    CPU-testable without a checkpoint or CUDA installation.
    """

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, manifest, prefix: str):
        self.manifest = manifest
        self.prefix = prefix
        self.layer_id: int | None = None
        self.moe_runner_config = None
        self.runtime = None
        self.runners = {}
        self.swiglu_params = {}

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **_extra_weight_attrs,
    ) -> None:
        if num_experts != self.manifest.num_experts:
            raise ValueError("Qwen3.8 routed-expert count does not match the pack")
        if hidden_size != self.manifest.hidden_size:
            raise ValueError("Qwen3.8 hidden size does not match the pack")
        if intermediate_size_per_partition != self.manifest.intermediate_size:
            raise ValueError("Qwen3.8 MoE intermediate size does not match the pack")
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Qwen3.8 cached NVFP4 requires BF16 or FP16 activations")
        if (
            getattr(layer, "moe_ep_size", 1) != 1
            or getattr(layer, "moe_tp_size", 1) != 1
        ):
            raise ValueError("Qwen3.8 cached NVFP4 v1 supports TP=EP=1 only")
        self.layer_id = int(getattr(layer, "layer_id"))
        if not 0 <= self.layer_id < self.manifest.num_layers:
            raise ValueError(f"invalid Qwen3.8 layer id {self.layer_id}")
        layer.register_buffer(
            "qwen38_expert_cache_marker",
            torch.empty(0, dtype=torch.uint8),
            persistent=False,
        )

    def create_moe_runner(self, layer, moe_runner_config) -> None:
        del layer
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def bind_runtime(self, runtime) -> None:
        self.runtime = runtime

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        if self.runtime is None:
            raise RuntimeError("Qwen3.8 expert runtime was not bound by the loader")
        if not dispatch_output.format.is_standard():
            raise ValueError("Qwen3.8 cache requires standard single-rank dispatch")
        if dispatch_output.hidden_states.shape[0] == 0:
            return StandardCombineInput(hidden_states=dispatch_output.hidden_states)
        if self.moe_runner_config.apply_router_weight_on_input:
            raise ValueError("Qwen3.8 CUTLASS cannot apply router weights on input")
        if dispatch_output.hidden_states_scale is not None:
            raise ValueError("Qwen3.8 cache requires unquantized BF16/FP16 dispatch")
        topk = dispatch_output.topk_output
        if not hasattr(topk, "topk_ids"):
            topk = topk.to_standard()

        hidden_states = self._run_experts(
            dispatch_output.hidden_states,
            topk.topk_weights,
            topk.topk_ids,
            topk.router_logits,
            dispatch_output.hidden_states_pre_quant,
        )
        return StandardCombineInput(hidden_states=hidden_states)

    @eager_on_graph(True, capture_stub=_capture_expert_output)
    def _run_experts(
        self, hidden_states, topk_weights, topk_ids, router_logits,
        hidden_states_pre_quant=None,
    ):
        """Replay host transport for every token, never capture a transfer job.

        Pass tensors directly through the break boundary: the breakable backend
        weak-refs tensor inputs and pins its output bridge. Capturing a closure
        over dispatch_output instead would retain every layer's intermediates.
        CUTLASS stays eager here so computed events cover its actual execution
        on every replay, including evictions and prefill-to-decode transitions.
        """
        from dataclasses import replace

        from sglang.srt.layers.moe.moe_runner import MoeRunner
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            materialize_swiglu_params_for_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        dispatch_output = StandardDispatchOutput(
            hidden_states,
            None,
            StandardTopKOutput(topk_weights, topk_ids, router_logits),
            hidden_states_pre_quant,
        )

        def execute(prepared):
            capacity = prepared.num_experts
            if capacity not in self.runners:
                config = replace(
                    self.moe_runner_config,
                    num_experts=capacity,
                    num_local_experts=capacity,
                )
                self.runners[capacity] = MoeRunner(
                    MoeRunnerBackend.FLASHINFER_CUTLASS, config
                )
                self.swiglu_params[capacity] = materialize_swiglu_params_for_cutlass(
                    config, capacity, self.runtime.device
                )
            tensors = prepared.tensors
            swiglu_alpha, swiglu_beta, swiglu_limit = self.swiglu_params[capacity]
            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="fp4",
                w13_weight=tensors["w13_weight"],
                w2_weight=tensors["w2_weight"],
                output_dtype=torch.bfloat16,
                quant_scales=[
                    tensors["w13_input_scale_quant"],
                    tensors["w13_blockscale_swizzled"],
                    tensors["g1_alphas"],
                    tensors["w2_input_scale_quant"],
                    tensors["w2_blockscale_swizzled"],
                    tensors["g2_alphas"],
                ],
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                apply_routed_scaling_factor=False,
            )
            remapped = StandardTopKOutput(
                topk_weights, prepared.topk_ids, router_logits
            )
            return self.runners[capacity].run(
                dispatch_output._replace(topk_output=remapped), quant_info
            )

        # Return a plain tensor, so eager_on_graph copies fresh output into the
        # fixed bridge consumed by the next GPU segment on every replay.
        return self.runtime.run(self.layer_id, topk_ids, execute).hidden_states


class Qwen38CachedModelOptConfig(QuantizationConfig):
    """Delegate dense/shared/PLE quantization and replace routed FusedMoE only."""

    supports_qwen38_mixed_precision = True

    def __init__(self, delegate: QuantizationConfig, manifest, gdn_value_heads=None):
        super().__init__()
        self.delegate = delegate
        self.manifest = manifest
        self.gdn_value_heads = gdn_value_heads
        self._pending_gdn_weights = {}
        self.packed_modules_mapping = delegate.packed_modules_mapping

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def get_name(self) -> str:
        return "qwen38_nvfp4_cache"

    def get_scaled_act_names(self):
        return self.delegate.get_scaled_act_names()

    def get_supported_act_dtypes(self):
        return self.delegate.get_supported_act_dtypes()

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        raise RuntimeError("Qwen38CachedModelOptConfig wraps an already parsed config")

    def apply_weight_name_mapper(self, mapper) -> None:
        self.delegate.apply_weight_name_mapper(mapper)
        self.packed_modules_mapping = self.delegate.packed_modules_mapping

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if (
            isinstance(layer, FusedMoE)
            and re.search(r"(?:^|\.)layers\.\d+\.mlp(?:\.experts)?$", prefix)
            and "mtp" not in prefix
        ):
            if self.delegate.resolve_quant_algo(prefix) != "NVFP4":
                raise ValueError(f"{prefix} must use serialized NVFP4 routed experts")
            return Qwen38CachedNvFp4MoeMethod(self.manifest, prefix)
        method = self.delegate.get_quant_method(layer, prefix)
        if self.gdn_value_heads is not None and re.search(
            r"(?:^|\.)layers\.\d+\.linear_attn\.in_proj_(?:qkvz|ba)$", prefix
        ):
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
            from sglang.srt.layers.quantization.qwen38_gdn_preallocated import (
                Qwen38PreallocatedGdnLinearMethod,
            )

            key, projection = prefix.rsplit(".", 1)
            other = "in_proj_ba" if projection == "in_proj_qkvz" else "in_proj_qkvz"
            if (
                type(method) is UnquantizedLinearMethod
                and layer.params_dtype == torch.bfloat16
            ):
                # BA consumes only storage already selected by its QKVZ
                # projection; quantized/mixed layouts keep the original path.
                pair_is_preallocated = (
                    projection == "in_proj_qkvz"
                    and self.delegate.resolve_quant_algo(key + "." + other) is None
                ) or (projection == "in_proj_ba" and key in self._pending_gdn_weights)
                if pair_is_preallocated:
                    return Qwen38PreallocatedGdnLinearMethod(
                        self._pending_gdn_weights, key, projection, self.gdn_value_heads
                    )
        return method
