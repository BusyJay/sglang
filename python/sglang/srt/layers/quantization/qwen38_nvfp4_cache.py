# SPDX-License-Identifier: Apache-2.0
"""Quantization boundary for host-cached Qwen3.8 NVFP4 routed experts."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)


def _capture_expert_output(_method, hidden_states, *_route_inputs, **_options):
    # Upstream segments have only been recorded, not replayed: their route IDs
    # are not valid yet. Do not enqueue a host transfer from capture-time data.
    # Warmup initializes CUTLASS; replay always runs the real expert boundary.
    shared_call = _options.get("shared_call")
    if shared_call is not None:
        # No route reads or transfers: record only the independent shared work.
        shared_call.capture(_options["shared_input"])
    return torch.zeros_like(hidden_states)


class Qwen38CachedNvFp4MoeMethod(FusedMoEMethodBase):
    """Creates no full-expert GPU parameters.

    The fixed-address cache buffers are installed after loading by the runtime.
    Keeping this method parameter-free is the critical loader invariant and is
    CPU-testable without a checkpoint or CUDA installation.
    """

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, manifest, prefix: str):
        # Keep CUTLASS as the default: the optional SIMT A16 implementation is
        # substantially slower on SM120, particularly for multi-token prefill.
        enabled = os.getenv("SGLANG_QWEN38_CACHE_W4A16", "0")
        if enabled not in ("0", "1"):
            raise ValueError("SGLANG_QWEN38_CACHE_W4A16 must be 0 or 1")
        self.use_w4a16 = enabled == "1"
        self.w4a16_backend = os.getenv("SGLANG_QWEN38_CACHE_W4A16_BACKEND", "triton")
        if self.w4a16_backend not in ("triton", "b12x"):
            raise ValueError("W4A16 backend must be triton or b12x")
        if self.w4a16_backend == "b12x" and not self.use_w4a16:
            raise ValueError("B12X requires SGLANG_QWEN38_CACHE_W4A16=1")
        self.b12x = None
        self.b12x_w4a4 = None
        prefill = os.getenv("SGLANG_QWEN38_CACHE_PREFILL_W4A16", enabled)
        if prefill not in ("0", "1"):
            raise ValueError("SGLANG_QWEN38_CACHE_PREFILL_W4A16 must be 0 or 1")
        self.prefill_use_w4a16 = prefill == "1"
        self.w4a16_chunk_tokens = int(
            os.getenv("SGLANG_QWEN38_CACHE_W4A16_CHUNK_TOKENS", "16")
        )
        if not 1 <= self.w4a16_chunk_tokens <= 64:
            raise ValueError("W4A16 chunk tokens must be in 1..64")
        if getattr(manifest, "scale_encoding", {}):
            if not self.use_w4a16:
                raise ValueError("CSF packs require W4A16 for decode/verify")
        self.manifest = manifest
        self.prefix = prefix
        self.layer_id: int | None = None
        self.moe_runner_config = None
        self.runtime = None
        self.runners = {}
        self.swiglu_buffers = None

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
        if getattr(runtime, "cpu_hybrid_enabled", False) or getattr(
            runtime, "layered_cpu_mode", None
        ):
            config = self.moe_runner_config
            if (
                config.activation != "silu"
                or not config.is_gated
                or config.apply_router_weight_on_input
                or config.no_combine
                or config.swiglu_limit is not None
                or config.gemm1_clamp_limit is not None
                or config.gemm1_alpha is not None
                or config.gemm1_beta is not None
                or config.num_fused_shared_experts not in (None, 0)
            ):
                raise ValueError(
                    "CPU hybrid requires ordinary SwiGLU with output route weighting"
                )
            if not self.use_w4a16:
                raise ValueError("CPU hybrid requires target W4A16")
            if getattr(runtime, "layered_cpu_mode", None):
                if runtime.layered_hybrid is None:
                    from sglang.srt.layers.moe.qwen38_layered_hybrid import (
                        LayeredCpuHybrid,
                    )

                    runtime.layered_hybrid = LayeredCpuHybrid(
                        runtime,
                        self.w4a16_backend,
                        (
                            "zero"
                            if runtime.layered_cpu_mode == "layered_zero"
                            else (
                                "lfu"
                                if runtime.layered_cpu_mode == "layered_lfu"
                                else (
                                    "probe"
                                    if runtime.layered_cpu_mode == "layered_probe"
                                    else "calibrated"
                                )
                            )
                        ),
                    )
                elif runtime.layered_hybrid.backend != self.w4a16_backend:
                    raise ValueError("all layered CPU layers must use one backend")
            elif getattr(runtime, "hybrid", None) is None:
                from sglang.srt.layers.moe.qwen38_hybrid import HybridCache

                runtime.hybrid = HybridCache(runtime, self.w4a16_backend)
        if (
            getattr(runtime, "hybrid", None) is not None
            and runtime.hybrid.gpu_backend != self.w4a16_backend
        ):
            raise ValueError("all global-cache layers must use the same GPU backend")
        from sglang.srt.layers.moe.qwen38_cache_buffers import GrowableSwigluBuffers

        self.swiglu_buffers = GrowableSwigluBuffers(
            self.moe_runner_config,
            runtime.device,
            (
                runtime.manifest.num_layers * runtime.manifest.num_experts
                if getattr(runtime, "hybrid", None) is not None
                else runtime.manifest.num_experts
            ),
        )
        # Seed at the current layer capacity before KV and graph allocation.
        self.swiglu_buffers.get(runtime.slots_for_layer(self.layer_id))
        if self.w4a16_backend == "b12x":
            from sglang.srt.layers.quantization.qwen38_cached_b12x import (
                CachedB12xW4A16,
            )

            self.b12x = CachedB12xW4A16(
                runtime.device,
                self.manifest.hidden_size,
                self.manifest.intermediate_size,
                (
                    self.manifest.num_layers * self.manifest.num_experts
                    if getattr(runtime, "hybrid", None) is not None
                    else self.manifest.num_experts
                ),
                enable_fp32=(
                    getattr(runtime, "hybrid", None) is not None
                    or runtime.layered_hybrid is not None
                ),
                scale_encoding=getattr(self.manifest, "scale_encoding", {}),
            )
            if getattr(runtime, "hybrid", None) is not None:
                if runtime.hybrid.gpu_backend != self.w4a16_backend:
                    raise ValueError(
                        "all global-cache layers must use the same GPU backend"
                    )
                runtime.hybrid.b12x[self.layer_id] = self.b12x
            if runtime.layered_hybrid is not None:
                runtime.layered_hybrid.b12x[self.layer_id] = self.b12x
            if self.layer_id == 0:
                logging.getLogger(__name__).info(
                    "Qwen38 B12X W4A16 enabled: BF16 decode/verify 1..3 tokens; "
                    "prefill uses its configured backend; unsupported decode/verify raises"
                )
        if self.manifest.scale_encoding and not self.prefill_use_w4a16:
            from sglang.srt.layers.quantization.qwen38_cached_b12x_w4a4 import (
                CachedB12xW4A4,
            )
            from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
                validate_runner_config,
            )

            validate_runner_config(self.moe_runner_config)
            # Serial expert layers share activation/routing scratch. Only
            # borrowed slot views vary, including after cache pool resizing.
            if getattr(runtime, "b12x_w4a4", None) is None:
                runtime.b12x_w4a4 = CachedB12xW4A4(
                    runtime.device,
                    self.manifest.hidden_size,
                    self.manifest.intermediate_size,
                    self.manifest.num_layers * self.manifest.num_experts,
                    self.manifest.top_k,
                    scale_encoding=self.manifest.scale_encoding,
                )
            self.b12x_w4a4 = runtime.b12x_w4a4
        if self.layer_id == 0:
            logging.getLogger(__name__).info(
                "Qwen38 cache activation mode: prefill=%s decode/verify=%s; W4A16 chunk=%d",
                (
                    "W4A16"
                    if self.prefill_use_w4a16
                    else (
                        "W4A4 B12X inline CSF"
                        if self.b12x_w4a4 is not None
                        else "W4A4 CUTLASS"
                    )
                ),
                "W4A16" if self.use_w4a16 else "W4A4 CUTLASS",
                self.w4a16_chunk_tokens,
            )

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.qwen38_cache_overlap import current_fill_overlap
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

        overlap = current_fill_overlap()
        overlap_options = {}
        if overlap is not None:
            call, shared_input = overlap
            overlap_options = dict(shared_call=call, shared_input=shared_input)
        hidden_states = self._run_experts(
            dispatch_output.hidden_states,
            topk.topk_weights,
            topk.topk_ids,
            topk.router_logits,
            dispatch_output.hidden_states_pre_quant,
            execution_phase=self._execution_phase(),
            **overlap_options,
        )
        return StandardCombineInput(hidden_states=hidden_states)

    def _execution_phase(self):
        from sglang.srt.runtime_context import get_forward

        return get_forward().qwen38_execution_phase

    def uses_w4a16(self, execution_phase):
        # Phase, not token count: a one-token prefill is still prefill, and a
        # multi-token verify/replay must retain the decode activation mode.
        if execution_phase == "prefill":
            return self.prefill_use_w4a16
        return self.use_w4a16

    def can_overlap_shared(self, block, hidden_states, phase):
        from sglang.srt.layers.moe.qwen38_cache_overlap import can_overlap_shared

        return can_overlap_shared(self, block, hidden_states, phase)

    def run_shared_overlap(self, block, hidden_states, phase):
        from sglang.srt.layers.moe.qwen38_cache_overlap import run_shared_overlap

        return run_shared_overlap(block, hidden_states, phase)

    @eager_on_graph(True, capture_stub=_capture_expert_output)
    def _run_experts(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        router_logits,
        hidden_states_pre_quant=None,
        execution_phase=None,
        shared_call=None,
        shared_input=None,
    ):
        """Replay host transport for every token, never capture a transfer job.

        Pass tensors directly through the break boundary: the breakable backend
        weak-refs tensor inputs and pins its output bridge. Capturing a closure
        over dispatch_output instead would retain every layer's intermediates.
        CUTLASS stays eager here so computed events cover its actual execution
        on every replay, including evictions and prefill-to-decode transitions.
        """
        from dataclasses import replace

        hybrid = getattr(self.runtime, "hybrid", None)
        if hybrid is not None and execution_phase in ("decode", "verify"):
            if (
                hidden_states.dtype != torch.bfloat16
                or not 1 <= hidden_states.shape[0] <= 3
            ):
                raise ValueError("CPU hybrid supports BF16 target decode/verify M=1..3")
            return hybrid.run(
                self.layer_id,
                hidden_states.contiguous(),
                topk_ids.int().contiguous(),
                topk_weights.float().contiguous(),
                (
                    (lambda: shared_call.run(shared_input))
                    if shared_call is not None
                    else None
                ),
            )
        layered = getattr(self.runtime, "layered_hybrid", None)
        if layered is not None and execution_phase in ("decode", "verify"):
            if (
                hidden_states.dtype != torch.bfloat16
                or not 1 <= hidden_states.shape[0] <= 3
            ):
                raise ValueError("layered CPU hybrid supports BF16 target M=1..3")
            if layered.supports(
                self.layer_id,
                hidden_states.shape[0],
                self.runtime.slots_for_layer(self.layer_id),
                shared_call is not None,
            ) and not self.runtime._use_prefill_arena(self.layer_id, topk_ids.numel()):
                return self.runtime.run_layered(
                    self.layer_id,
                    hidden_states.contiguous(),
                    topk_ids.int().contiguous(),
                    topk_weights.float().contiguous(),
                    (
                        (lambda: shared_call.run(shared_input))
                        if shared_call is not None
                        else None
                    ),
                    execution_phase=execution_phase,
                )
        from sglang.srt.layers.moe.moe_runner import MoeRunner
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
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

        def execute_single(prepared, route_weights, output_dtype=None):
            if self.uses_w4a16(execution_phase):
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )
                from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
                    cached_w4a16_moe,
                    validate_runner_config,
                )

                validate_runner_config(self.moe_runner_config)
                if self.b12x is not None and self.b12x.eligible(
                    hidden_states, prepared.topk_ids, execution_phase
                ):
                    result = self.b12x.run(
                        hidden_states,
                        prepared.topk_ids,
                        route_weights,
                        prepared.tensors,
                        output_dtype=output_dtype,
                    )
                    return StandardCombineInput(hidden_states=result)
                if self.w4a16_backend == "b12x" and execution_phase in (
                    "decode",
                    "verify",
                ):
                    raise RuntimeError(
                        "B12X decode/verify is unavailable for this input; refusing Triton fallback"
                    )
                result = cached_w4a16_moe(
                    hidden_states.contiguous(),
                    prepared.topk_ids.contiguous(),
                    route_weights.contiguous(),
                    prepared.tensors,
                    chunk_tokens=self.w4a16_chunk_tokens,
                    output_dtype=output_dtype,
                )
                return StandardCombineInput(hidden_states=result)
            if self.b12x_w4a4 is not None:
                from sglang.kernels.ops.moe.qwen38_b12x_w4a4 import cached_b12x_w4a4_moe
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )

                if execution_phase != "prefill" or output_dtype not in (
                    None,
                    torch.bfloat16,
                ):
                    raise ValueError("B12X W4A4 is the BF16 GPU prefill path")
                result = cached_b12x_w4a4_moe(
                    self.b12x_w4a4,
                    hidden_states,
                    prepared.topk_ids,
                    route_weights,
                    prepared.tensors,
                )
                return StandardCombineInput(hidden_states=result)
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
            tensors = prepared.tensors
            swiglu_alpha, swiglu_beta, swiglu_limit = self.swiglu_buffers.get(capacity)
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
            runner = self.runners[capacity]
            # Bound CUTLASS's routed-activation workspace independently of the
            # attention/GDN prefill chunk. Do not repeat expert planning/fills.
            # Pre-quantized inputs have layout-specific scale padding; retain
            # their existing path rather than slice that representation here.
            chunk = 2048
            if (
                execution_phase == "prefill"
                and hidden_states.shape[0] > chunk
                and hidden_states_pre_quant is None
            ):
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )

                output = torch.empty(
                    hidden_states.shape,
                    dtype=quant_info.output_dtype,
                    device=hidden_states.device,
                )
                for start in range(0, hidden_states.shape[0], chunk):
                    end = start + chunk
                    remapped = StandardTopKOutput(
                        route_weights[start:end],
                        prepared.topk_ids[start:end],
                        router_logits[start:end] if router_logits is not None else None,
                    )
                    part = runner.run(
                        dispatch_output._replace(
                            hidden_states=hidden_states[start:end], topk_output=remapped
                        ),
                        quant_info,
                    ).hidden_states
                    output[start:end].copy_(part)
                return StandardCombineInput(hidden_states=output)
            remapped = StandardTopKOutput(
                route_weights, prepared.topk_ids, router_logits
            )
            return runner.run(
                dispatch_output._replace(topk_output=remapped), quant_info
            )

        def execute(prepared):
            if getattr(prepared, "early_output", None) is not None:
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )

                # The planner excludes already computed hits from mixed plans.
                # All-hit/all-miss plans have no early routes and run here once.
                missing = execute_single(prepared, topk_weights).hidden_states
                return StandardCombineInput(
                    hidden_states=prepared.early_output + missing
                )
            if getattr(prepared, "cache_experts", None) is None:
                return execute_single(prepared, topk_weights)
            from sglang.srt.layers.moe.qwen38_prefill_routes import (
                partition_prefill_routes,
            )
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )

            cache_ids, cache_weights, arena_ids, arena_weights = (
                partition_prefill_routes(
                    prepared.topk_ids,
                    topk_weights,
                    prepared.cache_experts,
                    prepared.arena_experts,
                    lookup=prepared.cache_lookup,
                    slot_limit=prepared.cache_slot_limit,
                )
            )
            cache_prepared = replace(
                prepared,
                tensors=prepared.cache_tensors,
                topk_ids=cache_ids,
                num_experts=prepared.cache_tensors["w13_weight"].shape[0],
                arena=False,
                cache_tensors=None,
                cache_experts=None,
                arena_experts=None,
            )
            partial_dtype = (
                torch.float32
                if hybrid is not None and self.uses_w4a16(execution_phase)
                else None
            )
            cached = execute_single(
                cache_prepared, cache_weights, partial_dtype
            ).hidden_states
            if arena_ids is None:
                return StandardCombineInput(
                    hidden_states=cached.to(hidden_states.dtype)
                )
            # Some backends reuse an output buffer; preserve the first result
            # before dispatching the second contribution.
            cached = cached.clone()
            if prepared.wait_for_weights is not None:
                prepared.wait_for_weights()
            missing = execute_single(
                replace(
                    prepared,
                    topk_ids=arena_ids,
                    cache_tensors=None,
                    cache_experts=None,
                    arena_experts=None,
                ),
                arena_weights,
                partial_dtype,
            ).hidden_states
            return StandardCombineInput(
                hidden_states=(cached + missing).to(hidden_states.dtype)
            )

        # Return a plain tensor, so eager_on_graph copies fresh output into the
        # fixed bridge consumed by the next GPU segment on every replay.
        options = {}
        if getattr(self.runtime, "supports_execution_phase", False):
            # Python forward scopes have exited at replay. Keep the phase in
            # the eager break's captured arguments alongside its tensor inputs.
            options["execution_phase"] = execution_phase
            if execution_phase == "prefill":
                options["prefill_w4a4"] = not self.uses_w4a16(execution_phase)
        if shared_call is not None:
            options["during_fill"] = lambda: shared_call.run(shared_input)
        if (
            execution_phase == "decode"
            and getattr(self.runtime, "decode_early_cache", False)
            # Forced-miss sweeps have no reliable profitable decode interval
            # with these backends. Auto must take the original single call,
            # including avoiding empty early kernels, events and output merge.
            and getattr(self.runtime, "cache_overlap_policy", "always") == "always"
            and self.uses_w4a16(execution_phase)
            and not self.moe_runner_config.inplace
            and self.moe_runner_config.activation == "silu"
            and self.moe_runner_config.is_gated
            and not self.moe_runner_config.apply_router_weight_on_input
            and not self.moe_runner_config.no_combine
        ):

            def early_cache(ids, tensors):
                from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
                    cached_w4a16_moe,
                )

                # Only the baseline Triton path guarantees masked weight reads
                # for -1 IDs. Native B12X is still used after fill completion.
                return cached_w4a16_moe(
                    hidden_states.contiguous(),
                    ids.contiguous(),
                    topk_weights.contiguous(),
                    tensors,
                    chunk_tokens=self.w4a16_chunk_tokens,
                    split_k=0,
                    verify_mode="baseline",
                )

            options["early_cache"] = early_cache
        if hybrid is not None:
            if execution_phase != "prefill":
                raise ValueError("global hybrid requires target prefill/decode/verify")
            return hybrid.prefill(
                self.layer_id,
                topk_ids.int().contiguous(),
                execute,
                (
                    (lambda: shared_call.run(shared_input))
                    if shared_call is not None
                    else None
                ),
            )
        return self.runtime.run(
            self.layer_id, topk_ids, execute, **options
        ).hidden_states


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
        # Mark only native MXFP8 roles in this model adapter. The dense FP8
        # implementation verifies exact local shape and backend at execution.
        if envs.SGLANG_OPT_QWEN38_MXFP8_SMALL_M.get():
            from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

            roles = {
                "mlp.shared_expert.gate_up_proj": "shared_gate_up",
                "mlp.shared_expert.down_proj": "shared_down",
                "linear_attn.in_proj_qkvz": "gdn_qkvz",
                "linear_attn.in_proj_ba": "gdn_ba",
                "linear_attn.out_proj": "gdn_out",
                "self_attn.qkv_proj": "attention_qkv",
                "self_attn.o_proj": "attention_out",
            }
            if isinstance(method, Fp8LinearMethod) and method.use_mxfp8:
                for suffix, role in roles.items():
                    if prefix.endswith("." + suffix):
                        layer._qwen38_mxfp8_role = role
                        break
        if self.gdn_value_heads is not None and re.search(
            r"(?:^|\.)layers\.\d+\.linear_attn\.in_proj_(?:qkvz|ba)$", prefix
        ):
            from sglang.srt.layers.quantization.qwen38_gdn_preallocated import (
                Qwen38PreallocatedGdnLinearMethod,
            )
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

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
