# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text inference with native MLX KDA and indexed MLA caches.

Equations follow transformers.models.glm5_next.modeling_glm5_next. Unlike
the ordinary MLX attention adapter, MLA keeps compressed KV and indexer state;
these caches must stay together for the lifetime of a request.
"""

import copy
import logging
from dataclasses import dataclass
from functools import cache

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.models.kimi_linear import ShortConv1d, _group_expert_select
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)
from sglang.kernels.ops.attention.linear.mlx_kda import mlx_kda_decode
from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention
from sglang.kernels.ops.gemm.mlx_affine_rows import (
    mlx_affine_mixed_rows,
    mlx_affine_multi_prepare_strides,
    mlx_affine_multi_rows,
    mlx_affine_routed_swiglu,
    mlx_affine_rows,
    mlx_affine_swiglu_rows,
)
from sglang.kernels.ops.gemm.mlx_dense_rows import (
    mlx_dense_argmax,
    mlx_dense_argmax_rows,
    mlx_dense_rows,
    mlx_headed_dense_rows,
)
from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_dynamic_32
from sglang.kernels.ops.gemm.mlx_mixed_moe import (
    mlx_mixed_fp8_nvfp4_down,
    mlx_mixed_fp8_nvfp4_swiglu,
)
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
from sglang.kernels.ops.gemm.mlx_nvfp4_swiglu import mlx_nvfp4_swiglu
from sglang.kernels.ops.gemm.mlx_qmm_scatter import mlx_sorted_qmm_scatter
from sglang.kernels.ops.gemm.mlx_qmm_swiglu import (
    _expert_worklist,
    mlx_sorted_qmm_swiglu,
)
from sglang.kernels.ops.kvcache.mlx_latent import mlx_latent_gather_cast
from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm
from sglang.kernels.ops.layernorm.mlx_hc_boundary import mlx_hc_residual_rms_mix
from sglang.kernels.ops.layernorm.mlx_hc_mix import mlx_hc_rms_mix
from sglang.kernels.ops.moe.mlx_glm import (
    mlx_glm_down_combine,
    mlx_glm_router,
    mlx_glm_router_select,
)
from sglang.kernels.ops.moe.mlx_router_pairs import mlx_glm_router_select_pairs
from sglang.kernels.ops.quantization.mlx_fp8_cache import (
    mlx_fp8_cache_pack,
    mlx_fp8_cache_unpack,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        for name, value in self.text_config.items():
            setattr(self, name, value)
        if self.qk_rope_head_dim != 0:
            raise ValueError("GLM5-Next MLX requires NoPE (qk_rope_head_dim=0)")
        if any(t == "shared" for t in getattr(self, "indexer_types", [])):
            raise ValueError("GLM5-Next MLX does not support shared DSA indexers")


class NativeCache(ArraysCache):
    def __init__(self, size):
        super().__init__(size=size)
        self.offset = 0
        self.mlx_mtp = None


class MLACache(NativeCache):
    """Immutable latent prefix plus a bounded append block in normal state.

    Slot four holds at most 256 new rows. Retained snapshots share the sealed
    prefix; slice-update copying is bounded to the append block, including
    speculative rollback. Dense prefill can request a contiguous logical view.
    """

    block_size = 256

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if index != 0:
            return value
        if value is None:
            return None
        prefix = min(self.offset, value.shape[1])
        tail = self.cache[4] if len(self.cache) > 4 else None
        if self.offset <= prefix or tail is None:
            return value if prefix == value.shape[1] else value[:, :prefix]
        return mx.concatenate([value[:, :prefix], tail[:, : self.offset - prefix]], 1)

    def latent_segments(self):
        prefix = self.cache[0]
        count = min(self.offset, prefix.shape[1])
        tail = self.cache[4] if len(self.cache) > 4 else None
        if tail is None:
            tail = prefix[:, :0]
        return prefix[:, :count], tail[:, : max(0, self.offset - count)]

    def append_latent(self, value, *, materialize=True):
        if len(self.cache) == 4:
            self.cache.append(None)
        prefix, tail = self.cache[0], self.cache[4]
        if prefix is None:
            self.cache[0] = value
            return value
        if self.offset < prefix.shape[1]:
            prefix = prefix[:, : self.offset]
            tail = None
        dtype = mx.result_type(prefix.dtype, value.dtype)
        prefix, value = prefix.astype(dtype), value.astype(dtype)
        if tail is not None:
            tail = tail.astype(dtype)
        used = self.offset - prefix.shape[1]
        length = used + value.shape[1]
        if self.offset + value.shape[1] <= self.block_size:
            self.cache[0] = mx.concatenate([prefix, value], 1)
            self.cache[4] = None
            return self.cache[0]
        if length > self.block_size:
            parts = [prefix]
            if used:
                parts.append(tail[:, :used])
            parts.append(value)
            self.cache[0] = mx.concatenate(parts, 1)
            self.cache[4] = None
            return self.cache[0]
        if tail is None:
            tail = mx.zeros(
                (value.shape[0], self.block_size, value.shape[2]), value.dtype
            )
        tail[:, used:length] = value
        self.cache[0], self.cache[4] = prefix, tail
        return mx.concatenate([prefix, tail[:, :length]], 1) if materialize else None


class FP8MLACache(MLACache):
    """Packed E4M3 latent cache with /32 scales and existing prefix leases.

    Indexer pools stay BF16. Packed rows and scale bytes are snapshotted together;
    no quantization state lives outside the ordinary cache array protocol.
    """

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return (
            mlx_fp8_cache_unpack(value) if index == 0 and value is not None else value
        )

    def append_latent(self, value, *, materialize=True):
        if value.shape[-1] != 512:
            raise ValueError("FP8 MLA cache requires 512 latent channels")
        packed = mlx_fp8_cache_pack(value)
        out = super().append_latent(packed, materialize=materialize)
        return mlx_fp8_cache_unpack(out) if out is not None else None


def clamped_swiglu(gate, up, limit):
    if limit is not None:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


def _uses_fp16_affine_ffn(x, projections):
    # BF16 + FP16 affine metadata promotes QMM to FP32, including casts of
    # every expert's scales/biases in gather_qmm. Match the activation inside
    # the FFN only; keep BF16 in attention and the residual stream.
    return x.dtype == mx.bfloat16 and all(
        isinstance(proj, (nn.QuantizedLinear, QuantizedSwitchLinear))
        and proj.mode == "affine"
        and proj.scales.dtype == mx.float16
        and proj.biases is not None
        and proj.biases.dtype == mx.float16
        for proj in projections
    )


def _project_weight_rows(weight, x):
    out = mlx_dense_rows(x, weight)
    if out is None:
        out = mx.concatenate(
            [x[:, j : j + 1] @ weight.T for j in range(x.shape[1])], axis=1
        )
    return out


def _project_rows(projection, x):
    """Use singleton reductions with shared weights on covered row tiles."""
    if x.shape[1] > 17:
        return mx.concatenate(
            [
                _project_rows(projection, x[:, start : start + 17])
                for start in range(0, x.shape[1], 17)
            ],
            1,
        )
    from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

    if isinstance(projection, FP8Linear):
        return projection(x)
    out = None
    if type(projection) is nn.Linear and "bias" not in projection:
        out = mlx_dense_rows(x, projection.weight)
    elif (
        type(projection) is nn.QuantizedLinear
        and projection.mode == "affine"
        and "bias" not in projection
        and getattr(projection, "global_scale", None) is None
    ):
        out = mlx_affine_mixed_rows(
            x,
            projection.weight,
            projection.scales,
            projection.biases,
            bits=projection.bits,
            group_size=projection.group_size,
        )
    if out is None:
        out = mx.concatenate(
            [projection(x[:, j : j + 1]) for j in range(x.shape[1])], axis=1
        )
    return out


def _project_multi_rows(projections, x, *, prepared_strides=None):
    """Original E group eligibility; specialize prepared actual strides only."""
    if x.shape[1] <= 17 and all(_e_affine_projection(p) for p in projections):
        strides = prepared_strides
        if strides is not None and any(v is None for v in strides):
            strides = None
        result = mlx_affine_multi_rows(
            x, _e_parts(projections), prepared_strides=strides
        )
        if result is not None:
            return tuple(result)
    return tuple(_project_rows(p, x) for p in projections)


def _row_exact_affine_ffn(block, x, indices=None, *, gate_up_only=False, grouped=None):
    projections = (block.gate_proj, block.up_proj, block.down_proj)
    cls = QuantizedSwitchLinear if indices is not None else nn.QuantizedLinear
    if not all(
        type(p) is cls
        and p.mode == "affine"
        and "bias" not in p
        and getattr(p, "global_scale", None) is None
        for p in projections
    ):
        return None

    def project(value, projection):
        return mlx_affine_rows(
            value,
            projection.weight,
            projection.scales,
            projection.biases,
            bits=projection.bits,
            group_size=projection.group_size,
            indices=indices,
        )

    limit = block.activation.limit if indices is not None else block.limit
    if (projections[0].bits, projections[0].group_size) == (
        projections[1].bits,
        projections[1].group_size,
    ):
        fields = ("weight", "scales", "biases")
        operands = [tuple(p[k] for k in fields) for p in projections[:2]]
        if indices is None:
            act = mlx_affine_swiglu_rows(
                x,
                *operands,
                bits=projections[0].bits,
                group_size=projections[0].group_size,
                limit=limit,
            )
        else:
            act = mlx_affine_routed_swiglu(
                x,
                *operands,
                indices,
                bits=projections[0].bits,
                group_size=projections[0].group_size,
                limit=limit,
                grouped=grouped,
            )
        if act is not None:
            return act if gate_up_only else project(act, projections[2])
    gate, up = project(x, projections[0]), project(x, projections[1])
    if gate is None or up is None:
        return None
    act = _compiled_swiglu(gate, up, limit)
    return act if gate_up_only else project(act, projections[2])


class MLP(nn.Module):
    compile_decode = False
    compile_activation = False
    match_affine_dtype = False

    def __init__(self, args, intermediate_size=None):
        super().__init__()
        dim = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.down_proj = nn.Linear(dim, args.hidden_size, bias=False)
        self.limit = args.swiglu_limit

    def __call__(self, x, *, batch_fusion=False):
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

        if isinstance(self.gate_proj, FP8Linear):
            if (
                self.gate_proj.block_size == self.up_proj.block_size == 128
                and self.gate_proj.activation_mode
                == self.up_proj.activation_mode
                == "dynamic32"
                and self.gate_proj.weight.shape == self.up_proj.weight.shape
            ):
                from sglang.kernels.ops.gemm.mlx_block_fp8 import mlx_block_fp8_pair
                from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_dynamic_32

                quantized = mlx_fp8_dynamic_32(x)
                gate, up = mlx_block_fp8_pair(
                    x,
                    self.gate_proj.weight,
                    self.gate_proj.scales,
                    self.up_proj.weight,
                    self.up_proj.scales,
                    quantized,
                )
                if "bias" in self.gate_proj:
                    gate = gate + self.gate_proj.bias
                if "bias" in self.up_proj:
                    up = up + self.up_proj.bias
            else:
                gate, up = self.gate_proj(x), self.up_proj(x)
            return self.down_proj(clamped_swiglu(gate, up, self.limit))
        restore_dtype = self.match_affine_dtype and _uses_fp16_affine_ffn(
            x, (self.gate_proj, self.up_proj, self.down_proj)
        )
        dtype = x.dtype
        if restore_dtype:
            x = x.astype(mx.float16)
        if (
            batch_fusion
            and x.shape[1] > 1
            and all(
                type(p) is nn.Linear and "bias" not in p
                for p in (self.gate_proj, self.up_proj, self.down_proj)
            )
        ):
            gate = _project_rows(self.gate_proj, x)
            up = _project_rows(self.up_proj, x)
            act = (
                _compiled_swiglu(gate, up, self.limit)
                if self.compile_activation
                else clamped_swiglu(gate, up, self.limit)
            )
            out = _project_rows(self.down_proj, act)
            return out.astype(dtype) if restore_dtype else out
        if batch_fusion and x.shape[1] > 1:
            out = _row_exact_affine_ffn(self, x)
            if out is not None:
                return out.astype(dtype) if restore_dtype else out
        if (
            self.compile_decode
            and x.dtype != mx.float32
            and x.shape[0] == 1
            and (x.shape[1] == 1 or batch_fusion and 1 < x.shape[1] <= 17)
        ):
            out = _compiled_quantized_ffn(self, x)
            if out is not None:
                return out.astype(dtype) if restore_dtype else out
        out = self.down_proj(
            (_compiled_swiglu if self.compile_activation else clamped_swiglu)(
                self.gate_proj(x), self.up_proj(x), self.limit
            )
        )
        return out.astype(dtype) if restore_dtype else out


class ClampedSwiGLU(nn.Module):
    compile_activation = False

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        return (_compiled_swiglu if self.compile_activation else clamped_swiglu)(
            gate, up, self.limit
        )


class ClampedSwitchGLU(SwitchGLU):
    compile_decode = False
    fuse_gate_up = False
    match_affine_dtype = False

    def __init__(self, args):
        # Keep mlx-lm's sorting/grouped matmul path for larger token batches,
        # with GLM's clamped activation.
        super().__init__(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        self._fused_gate_up = None
        self._gate_up_sources = ()

    def _current_gate_up_sources(self):
        return tuple(
            value
            for p in (self.gate_proj, self.up_proj)
            for value in (p, p.weight, p.get("scales"), p.get("biases"))
        )

    def _current_gate_up_metadata(self):
        return tuple(
            (
                type(p),
                getattr(p, "bits", None),
                getattr(p, "group_size", None),
                getattr(p, "mode", None),
            )
            for p in (self.gate_proj, self.up_proj)
        )

    def prepare_for_inference(self):
        if not self.fuse_gate_up:
            return False
        if (
            self._fused_gate_up is not None
            and getattr(self, "_gate_up_metadata", None)
            == self._current_gate_up_metadata()
            and all(
                old is new
                for old, new in zip(
                    self._gate_up_sources, self._current_gate_up_sources()
                )
            )
        ):
            return True
        self._fused_gate_up = None
        gate, up = self.gate_proj, self.up_proj
        if not all(
            type(p) is QuantizedSwitchLinear
            and "bias" not in p
            and getattr(p, "global_scale", None) is None
            for p in (gate, up)
        ):
            return False
        if gate.mode != "affine" or (gate.bits, gate.group_size, gate.mode) != (
            up.bits,
            up.group_size,
            up.mode,
        ):
            return False
        fields = ("weight", "scales", "biases")
        if any(
            gate[f] is None
            or up[f] is None
            or gate[f].shape != up[f].shape
            or gate[f].dtype != up[f].dtype
            for f in fields
        ):
            return False
        fused = copy.copy(gate)
        for field in fields:
            fused[field] = mx.concatenate([gate[field], up[field]], axis=1)
        # Paired kernels read the original operands with independent strides.
        # Keep the joined native projection lazy for uncovered dispatch shapes;
        # preparation must not copy every expert's gate/up checkpoint storage.
        self._fused_gate_up = fused
        self._gate_up_sources = self._current_gate_up_sources()
        self._gate_up_metadata = self._current_gate_up_metadata()
        return True

    def __call__(self, x, indices):
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import NVFP4SwitchLinear

        if isinstance(self.gate_proj, NVFP4SwitchLinear):
            expanded = x[..., None, None, :]
            gate = self.gate_proj(expanded, indices)
            up = self.up_proj(expanded, indices)
            activated = self.activation(up, gate)
            return self.down_proj(activated, indices).squeeze(-2)
        if self._fused_gate_up is not None and (
            getattr(self, "_gate_up_metadata", None) != self._current_gate_up_metadata()
            or not all(
                old is new
                for old, new in zip(
                    self._gate_up_sources, self._current_gate_up_sources()
                )
            )
        ):
            self._fused_gate_up = None
            self._gate_up_sources = ()
        restore_dtype = self.match_affine_dtype and _uses_fp16_affine_ffn(
            x, (self.gate_proj, self.up_proj, self.down_proj)
        )
        dtype = x.dtype
        if restore_dtype:
            x = x.astype(mx.float16)
        gate, up = self.gate_proj, self.up_proj
        if (
            x.shape[1] > 1
            and isinstance(gate, QuantizedSwitchLinear)
            and isinstance(up, QuantizedSwitchLinear)
            and gate.mode == up.mode == "affine"
            and gate.bits == up.bits
            and gate.group_size == up.group_size
            and "bias" not in gate
            and "bias" not in up
            and getattr(gate, "global_scale", None) is None
            and getattr(up, "global_scale", None) is None
        ):
            act = mlx_affine_routed_swiglu(
                x,
                (gate.weight, gate.scales, gate.biases),
                (up.weight, up.scales, up.biases),
                indices,
                bits=gate.bits,
                group_size=gate.group_size,
                limit=self.activation.limit,
            )
            if act is not None:
                out = self.down_proj(
                    act[..., None, :], mx.stop_gradient(indices)
                ).squeeze(-2)
                return out.astype(dtype) if restore_dtype else out
        if (
            indices.size >= 64
            and self._fused_gate_up is not None
            and getattr(self, "_gate_up_metadata", None)
            == self._current_gate_up_metadata()
            and x.dtype != mx.float32
        ):
            # Sorted QMM shares input tiles and fuses both MMA projections and
            # rounded clamp/SwiGLU inside the shader. Down restores pick order.
            flat_ids = mx.stop_gradient(indices).reshape(-1)
            order = mx.argsort(flat_ids)
            inverse = mx.argsort(order)
            sorted_idx = flat_ids[order]
            sorted_x = x.reshape(-1, x.shape[-1])[order // indices.shape[-1]][:, None]
            tasks = None
            rows, experts = sorted_idx.size, gate.weight.shape[0]
            if rows // experts >= 4 and mx.device_info()["architecture"].startswith(
                "applegpu_g17"
            ):
                bm = 32 if rows // experts < 64 else 64
                tasks = _expert_worklist(sorted_idx, experts, rows, bm)
            activated = mlx_sorted_qmm_swiglu(
                sorted_x,
                sorted_idx,
                (gate.weight, gate.scales, gate.biases),
                (up.weight, up.scales, up.biases),
                bits=gate.bits,
                group_size=gate.group_size,
                limit=self.activation.limit,
                tasks=tasks,
            )
            if activated is None:
                joined = self._fused_gate_up(sorted_x, sorted_idx, sorted_indices=True)
                gate_values, up_values = mx.split(joined, 2, axis=-1)
                activated = _compiled_swiglu(
                    gate_values, up_values, self.activation.limit
                )
            output = None
            down = self.down_proj
            if (
                type(down) is QuantizedSwitchLinear
                and down.mode == "affine"
                and "bias" not in down
                and getattr(down, "global_scale", None) is None
            ):
                output = mlx_sorted_qmm_scatter(
                    activated,
                    sorted_idx,
                    order,
                    (down.weight, down.scales, down.biases),
                    bits=down.bits,
                    group_size=down.group_size,
                    tasks=tasks,
                )
            if output is None:
                output = self.down_proj(activated, sorted_idx, sorted_indices=True)
                output = output[inverse]
            output = mx.unflatten(output, 0, indices.shape).squeeze(-2)
            return output.astype(dtype) if restore_dtype else output
        if (
            self.compile_decode
            and x.dtype != mx.float32
            and x.shape[:2] == (1, 1)
            and indices.size < 64
        ):
            out = _compiled_quantized_ffn(self, x, indices)
            if out is not None:
                return out.astype(dtype) if restore_dtype else out
        if (
            x.shape[:2] != (1, 1)
            or indices.size >= 64
            or not all(
                isinstance(proj, QuantizedSwitchLinear)
                for proj in (self.up_proj, self.gate_proj, self.down_proj)
            )
        ):
            out = super().__call__(x, indices)
            return out.astype(dtype) if restore_dtype else out

        # Checkpoint each routed projection before building the next graph.
        # Full-weight GLM decode measurements favor these boundaries over
        # evaluating all three gather_qmm projections in the layer group.
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        if (
            self.fuse_gate_up
            and x.dtype != mx.float32
            and self._fused_gate_up is not None
            and getattr(self, "_gate_up_metadata", None)
            == self._current_gate_up_metadata()
            and all(
                old is new
                for old, new in zip(
                    self._gate_up_sources, self._current_gate_up_sources()
                )
            )
        ):
            gate_up = self._fused_gate_up(x, indices)
            mx.eval(gate_up)
            gate, up = mx.split(gate_up, 2, axis=-1)
        else:
            up = self.up_proj(x, indices)
            mx.eval(up)
            gate = self.gate_proj(x, indices)
            mx.eval(gate)
        out = self.down_proj(self.activation(up, gate), indices)
        mx.eval(out)
        out = out.squeeze(-2)
        return out.astype(dtype) if restore_dtype else out


def _compiled_quantized_ffn(block, x, indices=None, *, gate_up_only=False):
    projections = (block.gate_proj, block.up_proj, block.down_proj)
    cls = QuantizedSwitchLinear if indices is not None else nn.QuantizedLinear
    if not all(
        type(p) is cls
        and p.mode == "affine"
        and "bias" not in p
        and getattr(p, "global_scale", None) is None
        for p in projections
    ):
        return None
    limit = block.activation.limit if indices is not None else block.limit
    spec = tuple((p.group_size, p.bits) for p in projections) + (limit, gate_up_only)
    # Trace only scalar configuration. All weight arrays are explicit inputs,
    # avoiding retained checkpoint constants and stale weight-update traces.
    if getattr(block, "_compiled_ffn_spec", None) != spec:
        configs = spec[:3]
        routed = indices is not None

        def forward(x, arrays, indices):
            if routed:
                x = mx.expand_dims(x, (-2, -3))
                indices = mx.stop_gradient(indices)

            def project(value, i):
                weight, scales, biases = arrays[i]
                group_size, bits = configs[i]
                if routed:
                    return mx.gather_qmm(
                        value,
                        weight,
                        scales,
                        biases,
                        rhs_indices=indices,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                    )
                return mx.quantized_matmul(
                    value,
                    weight,
                    scales,
                    biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                )

            act = clamped_swiglu(project(x, 0), project(x, 1), limit)
            out = act if gate_up_only else project(act, 2)
            return out.squeeze(-2) if routed else out

        block._compiled_ffn = mx.compile(forward)
        block._compiled_ffn_spec = spec
    arrays = [(p.weight, p.scales, p.biases) for p in projections]
    return block._compiled_ffn(x, arrays, indices)


@mx.compile
def _compiled_swiglu(gate, up, limit):
    return clamped_swiglu(gate, up, limit)


@mx.compile
def _compiled_residual(residual, post, branch):
    return residual + post[..., None] * branch[..., None, :]


@mx.compile
def _compiled_kda_gate(out, gate, reference):
    return (out * mx.sigmoid(gate.astype(mx.float32))).astype(reference.dtype)


def _compiled_routed_gate_up(block, x, indices):
    """Cache only scalar tracing configuration; pass mutable weights explicitly."""
    gate, up = block.gate_proj, block.up_proj
    if (
        gate.bits not in (4, 5, 6, 8)
        or gate.group_size not in (32, 64, 128)
        or gate.biases is None
        or up.biases is None
    ):
        return None
    pack = 8 if gate.bits == 5 else 4 if gate.bits == 6 else 32 // gate.bits
    if (
        x.shape[-1] % (64 * pack)
        or gate.weight.shape[-2] % 16
        or gate.scales.dtype != x.dtype
        or up.scales.dtype != x.dtype
        or gate.biases.dtype != x.dtype
        or up.biases.dtype != x.dtype
    ):
        return None
    spec = (gate.bits, gate.group_size, block.activation.limit)
    if getattr(block, "_compiled_gate_up_spec", None) != spec:
        bits, group_size, limit = spec

        def forward(x, gate_arrays, up_arrays, indices):
            return mlx_affine_routed_swiglu(
                x,
                gate_arrays,
                up_arrays,
                indices,
                bits=bits,
                group_size=group_size,
                limit=limit,
            )

        block._compiled_gate_up = mx.compile(forward)
        block._compiled_gate_up_spec = spec
    return block._compiled_gate_up(
        x,
        tuple(gate[k] for k in ("weight", "scales", "biases")),
        tuple(up[k] for k in ("weight", "scales", "biases")),
        indices,
    )


class MoE(nn.Module):
    fuse_decode = False

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), mx.float32)
        self.switch_mlp = ClampedSwitchGLU(args)
        self.shared_experts = MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )
        self._router_weight = None
        self._router_sources = ()

    def _current_router_sources(self):
        return tuple(
            self.gate.get(field) for field in ("weight", "scales", "biases")
        ) + tuple(
            getattr(self.gate, field, None) for field in ("bits", "group_size", "mode")
        )

    def prepare_for_inference(self):
        """Materialize the small FP32 router once, with update invalidation."""
        sources = self._current_router_sources()
        if self._router_weight is not None and all(
            old is new for old, new in zip(self._router_sources, sources)
        ):
            return
        weight = self.gate.weight
        if isinstance(self.gate, nn.QuantizedLinear):
            weight = mx.dequantize(
                weight,
                self.gate.scales,
                self.gate.biases,
                group_size=self.gate.group_size,
                bits=self.gate.bits,
                mode=self.gate.mode,
            )
        self._router_weight = weight.astype(mx.float32)
        mx.eval(self._router_weight)
        self._router_sources = sources

    def __call__(self, x, *, batch_fusion=False, shared_stream=None, prefill_a8=False):
        # The router uses FP32, including the matrix multiplication.
        weight = self.gate.weight
        if self._router_weight is not None and all(
            old is new
            for old, new in zip(self._router_sources, self._current_router_sources())
        ):
            weight = self._router_weight
        else:
            self._router_weight = None
            self._router_sources = ()
            if isinstance(self.gate, nn.QuantizedLinear):
                weight = mx.dequantize(
                    weight,
                    self.gate.scales,
                    self.gate.biases,
                    group_size=self.gate.group_size,
                    bits=self.gate.bits,
                    mode=self.gate.mode,
                )
        a = self.args
        selected = None
        pair_worklist = None
        bounded = x.shape[0] == 1 and (
            x.shape[1] == 1 or batch_fusion and 1 < x.shape[1] <= 17
        )
        if (
            self.fuse_decode
            and bounded
            and a.n_group == 1
            and a.scoring_func == "sigmoid"
        ):
            selected = mlx_glm_router(
                x,
                weight,
                self.e_score_correction_bias,
                top_k=a.num_experts_per_tok,
                scaling=a.routed_scaling_factor,
                renormalize=a.norm_topk_prob,
            )
        if selected is None:
            from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import (
                NVFP4SwitchLinear,
            )

            if isinstance(self.switch_mlp.gate_proj, NVFP4SwitchLinear):
                # A fixed row tile keeps FP32 router logits identical for AR
                # and verification while sharing the small weight matrix.
                values = x.astype(mx.float32).reshape(-1, x.shape[-1])
                blocks = []
                for start in range(0, values.shape[0], 8):
                    tile = values[start : start + 8]
                    count = tile.shape[0]
                    if count < 8:
                        tile = mx.concatenate(
                            [tile, mx.zeros((8 - count, tile.shape[1]), mx.float32)], 0
                        )
                    blocks.append((tile @ weight.astype(mx.float32).T)[:count])
                scores = mx.concatenate(blocks, 0).reshape(
                    *x.shape[:-1], weight.shape[0]
                )
            else:
                scores = x.astype(mx.float32) @ weight.astype(mx.float32).T
            if (
                self.fuse_decode
                and bounded
                and a.n_group == 1
                and a.scoring_func == "sigmoid"
            ):
                if (
                    batch_fusion
                    and x.shape[1] > 1
                    and isinstance(self.switch_mlp.gate_proj, NVFP4SwitchLinear)
                ):
                    routed = mlx_glm_router_select_pairs(
                        scores,
                        self.e_score_correction_bias,
                        top_k=a.num_experts_per_tok,
                        scaling=a.routed_scaling_factor,
                        renormalize=a.norm_topk_prob,
                    )
                    if routed is not None:
                        selected, pair_worklist = routed[:2], routed[2:]
                if selected is None:
                    selected = mlx_glm_router_select(
                        scores,
                        self.e_score_correction_bias,
                        top_k=a.num_experts_per_tok,
                        scaling=a.routed_scaling_factor,
                        renormalize=a.norm_topk_prob,
                    )
            if selected is None:
                selected = _group_expert_select(
                    scores,
                    self.e_score_correction_bias,
                    a.num_experts_per_tok,
                    a.n_group,
                    a.topk_group,
                    a.routed_scaling_factor,
                    a.norm_topk_prob,
                    a.scoring_func,
                )
        indices, weights = selected
        if prefill_a8:
            from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import (
                NVFP4SwitchLinear,
            )

            if isinstance(self.switch_mlp.gate_proj, NVFP4SwitchLinear):
                from sglang.kernels.ops.gemm.mlx_nvfp4_prefill import (
                    mlx_nvfp4_prefill_a8,
                    mlx_nvfp4_prefill_gate_up_a8,
                    mlx_nvfp4_prefill_groups,
                )

                switch = self.switch_mlp
                indices = indices.astype(mx.uint32)
                groups = mlx_nvfp4_prefill_groups(
                    indices, switch.gate_proj.weight.shape[0]
                )

                def project(value, projection):
                    return mlx_nvfp4_prefill_a8(
                        value,
                        projection.weight,
                        projection.scales,
                        projection.weight_scale_2 * 2688,
                        indices,
                        groups,
                    )

                operands = lambda p: (p.weight, p.scales, p.weight_scale_2 * 2688)
                gate, up = mlx_nvfp4_prefill_gate_up_a8(
                    x,
                    operands(switch.gate_proj),
                    operands(switch.up_proj),
                    indices,
                    groups,
                )
                act = switch.activation(up, gate)
                routed = project(act, switch.down_proj).squeeze(-2)
                shared = self.shared_experts(x)
                return (routed * weights[..., None]).sum(-2).astype(x.dtype) + shared
        if pair_worklist is not None and shared_stream is None:
            from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

            sw, shared_block = self.switch_mlp, self.shared_experts
            projections = (shared_block.gate_proj, shared_block.up_proj)
            if all(
                isinstance(p, FP8Linear)
                and p.block_size == 128
                and p.activation_mode == "dynamic32"
                and "bias" not in p
                for p in projections
            ):

                def operands(p):
                    return p.weight, p.scales, p.weight_scale_2.reshape(-1) * 2688

                activation, shared_activation = mlx_mixed_fp8_nvfp4_swiglu(
                    x,
                    operands(sw.gate_proj),
                    operands(sw.up_proj),
                    indices,
                    pair_worklist,
                    (projections[0].weight, projections[0].scales),
                    (projections[1].weight, projections[1].scales),
                    mlx_fp8_dynamic_32(x),
                    limit=sw.activation.limit,
                )
                shared_down = shared_block.down_proj
                if (
                    getattr(shared_down, "block_size", None) == 128
                    and getattr(shared_down, "activation_mode", None) == "dynamic32"
                    and "bias" not in shared_down
                ):
                    routed, shared = mlx_mixed_fp8_nvfp4_down(
                        activation,
                        shared_activation,
                        operands(sw.down_proj),
                        indices,
                        pair_worklist,
                        (shared_down.weight, shared_down.scales),
                        mlx_fp8_dynamic_32(shared_activation),
                    )
                    routed = routed.squeeze(-2)
                else:
                    routed = mlx_nvfp4_gather_rows(
                        activation,
                        *operands(sw.down_proj),
                        indices,
                        worklist=pair_worklist,
                        output_rows=1,
                    ).squeeze(-2)
                    shared = shared_down(shared_activation)
                routed = (routed * weights[..., None]).sum(-2).astype(x.dtype)
                return routed + shared
        if shared_stream is None:
            shared = self.shared_experts(x, batch_fusion=batch_fusion)
        else:
            # MLX array dependencies join the side stream at combine. The
            # caller owns the stream on the serving thread and passes it in.
            with mx.stream(shared_stream):
                shared = self.shared_experts(x, batch_fusion=batch_fusion)
            mx.async_eval(shared)
        if self.fuse_decode and bounded:
            sw = self.switch_mlp
            projections = (sw.gate_proj, sw.up_proj, sw.down_proj)
            if all(
                type(p) is QuantizedSwitchLinear
                and p.mode == "affine"
                and "bias" not in p
                and getattr(p, "global_scale", None) is None
                for p in projections
            ):
                local = x
                if sw.match_affine_dtype and _uses_fp16_affine_ffn(x, projections):
                    local = x.astype(mx.float16)
                down = sw.down_proj
                if (
                    local.dtype in (mx.float16, mx.bfloat16)
                    and down.scales.dtype == local.dtype
                    and down.bits in (4, 5, 6, 8)
                    and down.group_size in (32, 64, 128)
                    and down.weight.shape[-1] * 32 // down.bits % 512 == 0
                    and down.weight.shape[-2] % 16 == 0
                ):
                    # Joint singleton gate/up reuse the input and affine sum
                    # on large expert rows; small tiles retain compiled QMV.
                    act = (
                        _row_exact_affine_ffn(sw, local, indices, gate_up_only=True)
                        if batch_fusion and x.shape[1] > 1
                        else None
                    )
                    if (
                        act is None
                        and x.shape[1] == 1
                        and sw.gate_proj.weight.shape[-2] >= 512
                        and sw.gate_proj.bits == sw.up_proj.bits
                        and sw.gate_proj.group_size == sw.up_proj.group_size
                    ):
                        act = _compiled_routed_gate_up(sw, local, indices)
                    if act is None:
                        act = _compiled_quantized_ffn(
                            sw, local, indices, gate_up_only=True
                        )
                    result = mlx_glm_down_combine(
                        act,
                        indices,
                        weights,
                        down.weight,
                        down.scales,
                        down.biases,
                        shared,
                        bits=down.bits,
                        group_size=down.group_size,
                    )
                    if result is not None:
                        return result
        if pair_worklist is not None:
            sw = self.switch_mlp

            def project(value, projection):
                return mlx_nvfp4_gather_rows(
                    value,
                    projection.weight,
                    projection.scales,
                    projection.weight_scale_2 * 2688,
                    indices,
                    worklist=pair_worklist,
                    output_rows=1,
                )

            activation = mlx_nvfp4_swiglu(
                x,
                sw.gate_proj.weight,
                sw.gate_proj.scales,
                sw.gate_proj.weight_scale_2 * 2688,
                sw.up_proj.weight,
                sw.up_proj.scales,
                sw.up_proj.weight_scale_2 * 2688,
                indices,
                limit=sw.activation.limit,
                worklist=pair_worklist,
            )
            routed = project(activation, sw.down_proj).squeeze(-2)
        else:
            routed = self.switch_mlp(x, indices)
        routed = (routed * weights[..., None]).sum(-2).astype(x.dtype)
        return routed + shared


class KDA(nn.Module):
    compile_gate = False
    fuse_decode = False

    def __init__(self, args):
        super().__init__()
        cfg = args.linear_attn_config
        self.num_heads, self.head_dim = cfg["num_heads"], cfg["head_dim"]
        self.lower_bound = cfg.get("gate_lower_bound")
        self.dim = self.num_heads * self.head_dim
        for name in ("q", "k", "v"):
            setattr(
                self, name + "_proj", nn.Linear(args.hidden_size, self.dim, bias=False)
            )
            setattr(
                self,
                name + "_conv",
                ShortConv1d(self.dim, cfg["short_conv_kernel_size"]),
            )
        self.f_a_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.dim, bias=False)
        self.g_a_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.dim, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,), mx.float32)
        self.dt_bias = mx.zeros((self.dim,), mx.float32)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.dim, args.hidden_size, bias=False)
        self._fused_qkv = None
        self._fused_qkv3 = None
        self._fused_qkv_sources = ()
        self._fused_input_names = ("q_proj", "k_proj", "v_proj")
        self._fused_input_splits = (self.dim, 2 * self.dim)

    def _qkv_sources(self):
        return tuple(
            value
            for p in (getattr(self, name) for name in self._fused_input_names)
            for value in (p, p.weight, p.get("scales"), p.get("biases"))
        )

    def _qkv_metadata(self):
        return tuple(
            (
                type(p),
                getattr(p, "bits", None),
                getattr(p, "group_size", None),
                getattr(p, "mode", None),
            )
            for p in (getattr(self, name) for name in self._fused_input_names)
        )

    def _fused_qkv_is_current(self):
        return (
            self._fused_qkv is not None
            and getattr(self, "_fused_qkv_metadata", None) == self._qkv_metadata()
            and all(
                old is new
                for old, new in zip(self._fused_qkv_sources, self._qkv_sources())
            )
        )

    def prepare_for_inference(self):
        """Fuse compatible KDA inputs once, including the gate projections.

        Canonical checkpoint parameters become views of the fused storage;
        saving a model still uses its original names and mixed-bit metadata.
        No second resident copy of all KDA projection weights is retained.
        """
        if self._fused_qkv_is_current():
            return True
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import (
            FP8KDAInputs,
            FP8Linear,
        )

        if all(
            isinstance(getattr(self, name + "_proj"), FP8Linear)
            for name in ("q", "k", "v")
        ):
            names = ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")
            self._fused_qkv = FP8KDAInputs(
                tuple(getattr(self, name) for name in names),
                packed=getattr(self, "_cached_fp8_qkv", None),
            )
            if hasattr(self, "_cached_fp8_qkv"):
                object.__delattr__(self, "_cached_fp8_qkv")
            self._fused_qkv3 = None
            self._fused_input_names = names
            self._fused_input_splits = (
                self.dim,
                2 * self.dim,
                3 * self.dim,
                3 * self.dim + self.head_dim,
                3 * self.dim + 2 * self.head_dim,
            )
            self._fused_qkv_sources = self._qkv_sources()
            self._fused_qkv_metadata = self._qkv_metadata()
            return True
        self._fused_qkv = None
        self._fused_qkv3 = None
        self._fused_qkv_sources = ()
        # Six projections consume the same input. Prefer one matmul when their
        # storage formats match; otherwise retain the compatible Q/K/V path.
        names = ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")
        if not self._prepare_input_fusion(names):
            return self._prepare_input_fusion(names[:3])
        return True

    def _prepare_input_fusion(self, names):
        projections = tuple(getattr(self, name) for name in names)
        first = projections[0]
        if type(first) not in (nn.Linear, nn.QuantizedLinear):
            return False
        if any(type(p) is not type(first) or "bias" in p for p in projections):
            return False
        fields = ("weight",)
        if isinstance(first, nn.QuantizedLinear):
            if first.mode != "affine" or any(
                (p.bits, p.group_size, p.mode)
                != (first.bits, first.group_size, first.mode)
                for p in projections
            ):
                return False
            fields += ("scales", "biases")
        if any(
            any(
                p[field].shape[1:] != first[field].shape[1:]
                or p[field].dtype != first[field].dtype
                for p in projections
            )
            for field in fields
        ):
            return False
        fused = copy.copy(first)
        for field in fields:
            fused[field] = mx.concatenate([p[field] for p in projections], axis=0)
        mx.eval([fused[field] for field in fields])
        splits, size = [], 0
        for projection in projections[:-1]:
            size += projection.weight.shape[0]
            splits.append(size)
        for field in fields:
            for projection, value in zip(
                projections, mx.split(fused[field], splits, axis=0)
            ):
                projection[field] = value
        self._fused_qkv = fused
        self._fused_qkv3 = None
        if len(names) == 6:
            # The wider FP32 matmul regresses on the full M5 checkpoint.
            # A three-way view shares storage and keeps the original dispatch.
            qkv3 = copy.copy(fused)
            for field in fields:
                qkv3[field] = fused[field][: 3 * self.dim]
            self._fused_qkv3 = qkv3
        self._fused_input_names = names
        self._fused_input_splits = tuple(splits)
        self._fused_qkv_sources = self._qkv_sources()
        self._fused_qkv_metadata = self._qkv_metadata()
        return True

    def project_verify_rows(self, x):
        """Share dense input weights across rows, keeping recurrence transactional."""
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8KDAInputs

        if (
            not self.fuse_decode
            or self.head_dim != 128
            or len(self._fused_input_names) != 6
            or not self._fused_qkv_is_current()
            or not (
                type(self._fused_qkv) is nn.Linear
                or isinstance(self._fused_qkv, FP8KDAInputs)
            )
        ):
            return None
        proj = (
            self._fused_qkv(x)
            if isinstance(self._fused_qkv, FP8KDAInputs)
            else mlx_dense_rows(x, self._fused_qkv.weight)
        )
        if proj is None:
            return None
        _, _, _, fa_end, ga_end = self._fused_input_splits
        a_input = proj[..., 3 * self.dim : fa_end]
        gate_input = proj[..., fa_end:ga_end]
        a = _project_rows(self.f_b_proj, a_input)
        gate = _project_rows(self.g_b_proj, gate_input)
        return proj, a, gate

    def verify_rows(self, x, cache, *, projected=None, project_output=True):
        projected = self.project_verify_rows(x) if projected is None else projected
        if projected is None:
            return None
        outputs, snapshots = [], []
        for start in range(0, x.shape[1], 8):
            count = min(8, x.shape[1] - start)
            proj, a, gate = (v[:, start : start + count] for v in projected)
            result = mlx_kda_decode(
                proj,
                tuple(
                    getattr(self, name + "_conv").conv.weight
                    for name in ("q", "k", "v")
                ),
                tuple(cache[i] for i in range(3)),
                cache[3],
                a,
                gate,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                heads=self.num_heads,
                norm_eps=self.o_norm.eps,
                lower_bound=self.lower_bound,
                snapshots=True,
            )
            if result is None:
                if start == 0:
                    # Projection coverage is wider than the current recurrent
                    # kernel (e.g. FP32 activations). No cache was changed.
                    return None
                raise ValueError(
                    "Prepared KDA verification operands do not match the fused recurrence"
                )
            out, conv, states = result
            outputs.append(out)
            for j in range(count):
                snapshot = type(cache)(4)
                snapshot.cache = [v[j : j + 1] for v in conv] + [states[j : j + 1]]
                snapshot.offset = cache.offset + j + 1
                snapshots.append(snapshot)
            cache.cache = [v.view(v.dtype) for v in snapshots[-1].cache]
            cache.offset += count
        out = mx.concatenate(outputs, 1)
        return (_project_rows(self.o_proj, out) if project_output else out), snapshots

    def decode_requests(self, rows, caches):
        """Share large projections; recurrence remains private to each request."""
        x = mx.concatenate(rows, axis=1)
        projected = self.project_verify_rows(x)
        if projected is None:
            return [self(row, cache) for row, cache in zip(rows, caches)]
        outputs, updates = [], []
        for j, cache in enumerate(caches):
            proj, a, gate = (v[:, j : j + 1] for v in projected)
            result = mlx_kda_decode(
                proj,
                tuple(
                    getattr(self, name + "_conv").conv.weight
                    for name in ("q", "k", "v")
                ),
                tuple(cache[i] for i in range(3)),
                cache[3],
                a,
                gate,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                heads=self.num_heads,
                norm_eps=self.o_norm.eps,
                lower_bound=self.lower_bound,
            )
            if result is None:
                # General dtype/geometry retains the same per-request equation.
                return [self(row, state) for row, state in zip(rows, caches)]
            out, conv, state = result
            updates.append((cache, conv, state))
            outputs.append(out)
        for cache, conv, state in updates:
            for i in range(3):
                cache[i] = conv[i]
            cache[3] = state
            cache.offset += 1
        projected_out = _project_rows(self.o_proj, mx.concatenate(outputs, 1))
        return [projected_out[:, j : j + 1] for j in range(len(rows))]

    def __call__(self, x, cache, *, projected=None):
        b, t, _ = x.shape
        if (
            self.fuse_decode
            and b == 1
            and 1 <= t <= 8
            and self.head_dim == 128
            and x.dtype in (mx.bfloat16, mx.float16)
            and len(self._fused_input_names) == 6
            and self._fused_qkv_is_current()
        ):
            if projected is None:
                proj = self._fused_qkv(x)
                _, _, _, fa_end, ga_end = self._fused_input_splits
                a = _project_rows(self.f_b_proj, proj[..., 3 * self.dim : fa_end])
                gate = _project_rows(self.g_b_proj, proj[..., fa_end:ga_end])
            else:
                proj, a, gate = projected
            result = mlx_kda_decode(
                proj,
                tuple(
                    getattr(self, name + "_conv").conv.weight
                    for name in ("q", "k", "v")
                ),
                tuple(cache[i] for i in range(3)),
                cache[3],
                a,
                gate,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                heads=self.num_heads,
                norm_eps=self.o_norm.eps,
                lower_bound=self.lower_bound,
            )
            if result is not None:
                out, conv, state = result
                for i in range(3):
                    cache[i] = conv[i]
                cache[3] = state
                cache.offset += t
                return self.o_proj(out)
        if self._fused_qkv_is_current():
            if x.dtype == mx.float32 and self._fused_qkv3 is not None:
                inputs = mx.split(
                    self._fused_qkv3(x), (self.dim, 2 * self.dim), axis=-1
                )
            else:
                inputs = mx.split(self._fused_qkv(x), self._fused_input_splits, axis=-1)
            projected = inputs[:3]
            if len(inputs) == 6:
                f_a, g_a, beta = inputs[3:]
            else:
                f_a, g_a, beta = self.f_a_proj(x), self.g_a_proj(x), self.b_proj(x)
        else:
            if self._fused_qkv is not None:
                self._fused_qkv = None
                self._fused_qkv3 = None
                self._fused_qkv_sources = ()
            projected = [self.q_proj(x), self.k_proj(x), self.v_proj(x)]
            f_a, g_a, beta = self.f_a_proj(x), self.g_a_proj(x), self.b_proj(x)
        projections = []
        for i, (name, value) in enumerate(zip(("q", "k", "v"), projected)):
            value, state = getattr(self, name + "_conv")(value, cache[i], None, None)
            cache[i] = state
            projections.append(value.reshape(b, t, self.num_heads, self.head_dim))
        q, k, v = projections
        q, k = normalize_qk(q, k, inv_scale=self.head_dim**-0.5, eps=1e-6)
        a = self.f_b_proj(f_a).reshape(b, t, self.num_heads, self.head_dim)
        out, cache[3] = gated_delta_update(
            q,
            k,
            v,
            a,
            beta,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=cache[3],
            lower_bound=self.lower_bound,
        )
        cache.offset += t
        gate = self.g_b_proj(g_a).reshape(out.shape)
        # GLM's gated norm multiplies the FP32 normalized result and gate
        # before casting back to the activation dtype.
        normed = mx.fast.rms_norm(
            out.astype(mx.float32),
            self.o_norm.weight.astype(mx.float32),
            self.o_norm.eps,
        )
        out = (
            _compiled_kda_gate(normed, gate, x)
            if self.compile_gate
            else (normed * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)
        )
        return self.o_proj(out.reshape(b, t, -1))


class Indexer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wq_b = nn.Linear(
            args.q_lora_rank, args.index_n_heads * args.index_head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, args.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(args.index_head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, args.index_n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros(
            (args.index_kpool, args.index_head_dim)
        )
        self.index_kpool_compress_gate = mx.zeros(
            (args.index_head_dim, args.hidden_size)
        )

    def _update_pool_cache(self, keys, gates, cache, offset):
        """Append completed pools once; retain raw keys/gates only for the tail.

        Slot 3 is a geometrically grown pooled-key buffer. Its valid prefix is
        determined by the sequence offset, so every state array remains in the
        cache's normal evaluation/serialization protocol. Three-slot caches
        with a full raw prefix can be upgraded on their first call.
        """
        a = self.args
        if len(cache.cache) == 3:
            cache.cache.append(None)
        pooled = cache[3]
        old_pools = offset // a.index_kpool if pooled is not None else 0
        length = offset + keys.shape[1]
        expected_tail = offset % a.index_kpool if pooled is not None else offset
        cached_length = cache[1].shape[1] if cache[1] is not None else 0
        if cached_length != expected_tail:
            raise ValueError("GLM indexer cache length does not match its offset")
        if cache[1] is not None:
            keys = mx.concatenate([cache[1], keys], axis=1)
            gates = mx.concatenate([cache[2], gates], axis=1)
        b, raw_length, dim = keys.shape
        added = raw_length // a.index_kpool
        count = old_pools + added
        if count != length // a.index_kpool:
            raise ValueError("GLM indexer cache length does not match its offset")
        if added:
            grouped = keys[:, : added * a.index_kpool].reshape(
                b, added, a.index_kpool, dim
            )
            logits = gates[:, : added * a.index_kpool].reshape(grouped.shape)
            probs = mx.softmax(
                logits.astype(mx.float32)
                + self.index_kpool_compress_ape.astype(mx.float32),
                axis=2,
            ).astype(keys.dtype)
            new_pooled = (probs * grouped).sum(axis=2)
            capacity = pooled.shape[1] if pooled is not None else 0
            if count > capacity:
                grown = mx.zeros((b, max(64, count, 2 * capacity), dim), keys.dtype)
                if old_pools:
                    grown[:, :old_pools] = pooled[:, :old_pools]
                pooled = grown
            pooled[:, old_pools:count] = new_pooled
            cache[3] = pooled
        tail_start = added * a.index_kpool
        if tail_start < raw_length:
            # Gather a compact tail: a slice could keep the whole prefill's
            # raw key/gate allocation alive through a small view.
            positions = mx.arange(tail_start, raw_length)
            cache[1] = mx.take(keys, positions, axis=1)
            cache[2] = mx.take(gates, positions, axis=1)
        else:
            cache[1], cache[2] = None, None
        return (
            pooled[:, :count]
            if pooled is not None
            else mx.zeros((b, 0, dim), keys.dtype)
        )

    def project_verify_rows(self, x, qr):
        return (
            self.k_norm(_project_rows(self.wk, x)),
            _project_rows(self.wq_b, qr),
            _project_rows(self.weights_proj, x),
            _project_weight_rows(self.index_kpool_compress_gate, x),
        )

    def __call__(self, x, qr, cache, offset, *, projected=None):
        """Return selected raw KV indices, or None for dense causal attention.

        Invalid complete pools and padded tail positions carry -1. Keeping
        indices instead of a full-history boolean mask lets attention gather
        only selected latent KV before either QK or PV multiplication.
        """
        a = self.args
        b, t, _ = x.shape
        if projected is None:
            keys = self.k_norm(self.wk(x))
            gates = x @ self.index_kpool_compress_gate.T
        else:
            keys, _, _, gates = projected
        pooled = self._update_pool_cache(keys, gates, cache, offset)
        length, pool = offset + t, a.index_kpool
        # Below the sparse budget every complete pool and its causal tail
        # are selected. Still retain indexer state for future sparse steps.
        if length <= a.index_topk and a.index_kpool_always_select_tail:
            return None
        q = (self.wq_b(qr) if projected is None else projected[1]).reshape(
            b,
            t,
            a.index_n_heads,
            a.index_head_dim,
        )
        scores = self._score_pools(
            q,
            weights=(self.weights_proj(x) if projected is None else projected[2]),
            pooled=pooled,
        )
        return self._select_scores(scores, offset=offset)

    def _score_pools(self, q, *, weights, pooled):
        a = self.args
        scores = mx.maximum(
            q.astype(mx.float32) @ pooled.astype(mx.float32).swapaxes(-1, -2)[:, None],
            0,
        )
        scores *= a.index_head_dim**-0.5
        weights = weights.astype(mx.float32) * a.index_n_heads**-0.5
        return (scores * weights[..., None]).sum(axis=2)

    def _select_scores(self, scores, *, offset):
        a = self.args
        b, t, n = scores.shape
        pool = a.index_kpool
        pool_ends = mx.arange(n) * pool + pool - 1
        valid = pool_ends[None, :] <= (offset + mx.arange(t))[:, None]
        scores = mx.where(valid[None], scores, -float("inf"))
        count = min(a.index_topk // pool, n)
        if count:
            selected = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
            indices = selected[..., None] * pool + mx.arange(pool)
            valid = indices[..., -1] <= (offset + mx.arange(t))[None, :, None]
            indices = mx.where(valid[..., None], indices, -1).reshape(b, t, -1)
        else:
            indices = mx.zeros((b, t, 0), dtype=mx.int32)
        if a.index_kpool_always_select_tail:
            visible_count = offset + mx.arange(t) + 1
            tail_count = visible_count % pool
            tail_start = visible_count - tail_count
            tail_offsets = mx.arange(pool - 1)
            tail = tail_start[:, None] + tail_offsets
            tail = mx.where(tail_offsets < tail_count[:, None], tail, -1)
            tail = mx.broadcast_to(tail, (b, t, pool - 1))
            indices = mx.concatenate([indices, tail], axis=-1)
        return indices

    def verify_rows(self, cache, offset, *, projected):
        """Score a block once, retaining each prefix's pool state and causal set."""
        a = self.args
        keys, queries, weights, gates = projected
        b, count, _ = keys.shape
        states = []
        pooled = None
        for j in range(count):
            pooled = self._update_pool_cache(
                keys[:, j : j + 1], gates[:, j : j + 1], cache, offset + j
            )
            states.append(
                [v.view(v.dtype) if v is not None else None for v in cache.cache[1:]]
            )
        if offset + count <= a.index_topk and a.index_kpool_always_select_tail:
            return [None] * count, states
        q = queries.reshape(b, count, a.index_n_heads, a.index_head_dim)
        scores = self._score_pools(q, weights=weights, pooled=pooled)
        selected = []
        for j in range(count):
            length = offset + j + 1
            if length <= a.index_topk and a.index_kpool_always_select_tail:
                selected.append(None)
            else:
                selected.append(
                    self._select_scores(
                        scores[:, j : j + 1, : length // a.index_kpool],
                        offset=offset + j,
                    )
                )
        return selected, states


def _f_visible_indices(indices, selected_lengths, visible_ends):
    # selected_lengths and causal ends are independent device metadata.
    # Respect original selected prefix before stable legal compaction.
    rows, width = indices.shape[1:]
    position = mx.arange(width, dtype=mx.int32)
    lengths = (
        selected_lengths
        if isinstance(selected_lengths, mx.array)
        else mx.array(selected_lengths, mx.uint32)
    ).reshape(rows)
    valid = (position[None, None, :] < lengths[None, :, None]) & (indices >= 0)
    valid &= indices.astype(mx.uint32) < visible_ends[None, :, None]
    order = mx.argsort(mx.where(valid, position, position + width), axis=-1)
    table = mx.take_along_axis(mx.where(valid, indices, -1), order, axis=-1)
    return table, mx.sum(valid, axis=-1).reshape(rows).astype(mx.uint32)


class MLA(nn.Module):
    cache_latent_projection = False

    def _latent_matrices(self, activation_dtype):
        """Retain the operands at their original matmul promotion boundary.

        Dense BF16 KV-B weights also promote for FP32 queries. Pack/cast
        each sliced operand separately, matching the uncached operation's
        layout rather than changing it by casting the interleaved parent.
        """
        projection = self.kv_b_proj
        if "decoded_weight" in projection:
            weight = projection.decoded_weight
            a = self.args
            wk, wv = mx.split(
                weight.reshape(a.num_attention_heads, -1, a.kv_lora_rank),
                [a.qk_nope_head_dim],
                axis=1,
            )
            return wk, wv.swapaxes(-1, -2)
        quantized = isinstance(projection, nn.QuantizedLinear)
        sources = (
            tuple(projection.get(k) for k in ("weight", "scales", "biases"))
            if quantized
            else (projection.weight,)
        )
        spec = (
            getattr(projection, "group_size", None),
            getattr(projection, "bits", None),
            getattr(projection, "mode", None),
            activation_dtype,
            self.args.num_attention_heads,
            self.args.kv_lora_rank,
            self.args.qk_nope_head_dim,
            self.args.v_head_dim,
        )
        saved = getattr(self, "_latent_projection", None)
        if (
            self.cache_latent_projection
            and saved is not None
            and saved[1] == spec
            and len(saved[0]) == len(sources)
            and all(a is b for a, b in zip(saved[0], sources))
        ):
            return saved[2]
        weight = (
            mx.dequantize(
                *sources,
                group_size=projection.group_size,
                bits=projection.bits,
                mode=projection.mode,
            )
            if quantized
            else projection.weight
        )
        a = self.args
        wk, wv = mx.split(
            weight.reshape(a.num_attention_heads, -1, a.kv_lora_rank),
            [a.qk_nope_head_dim],
            axis=1,
        )
        wv = wv.swapaxes(-1, -2)
        if self.cache_latent_projection:
            dtype = mx.result_type(weight.dtype, activation_dtype)
            matrices = (wk.astype(dtype), wv.astype(dtype))
            mx.eval(matrices)
            self._latent_projection = (sources, spec, matrices)
            return matrices
        self._latent_projection = None
        return wk, wv

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.q_a_proj = nn.Linear(
            args.hidden_size, args.q_lora_rank, bias=args.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank,
            args.num_attention_heads * args.qk_nope_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, args.kv_lora_rank, bias=args.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank,
            args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.indexer = Indexer(args)

    def prepare_e_projections(self):
        groups = (
            (self.q_a_proj, self.kv_a_proj_with_mqa, self.indexer.wk),
            (
                self.q_a_proj,
                self.kv_a_proj_with_mqa,
                self.indexer.wk,
                self.indexer.weights_proj,
            ),
            (self.q_b_proj, self.indexer.wq_b),
        )
        signature = tuple(
            (
                p,
                tuple(p[k] for k in ("weight", "scales", "biases") if k in p),
                getattr(p, "bits", None),
                getattr(p, "group_size", None),
                getattr(p, "mode", None),
                "bias" in p,
                getattr(p, "global_scale", None),
            )
            for group in groups
            for p in group
        )
        old = getattr(self, "_e_projection_sources", None)
        if (
            old is not None
            and len(old) == len(signature)
            and all(
                a[0] is b[0]
                and len(a[1]) == len(b[1])
                and all(x is y for x, y in zip(a[1], b[1]))
                and a[2:] == b[2:]
                for a, b in zip(old, signature)
            )
        ):
            return False
        tables = []
        for group in groups:
            ids = [i for i, p in enumerate(group) if _e_affine_projection(p)]
            table = [None] * len(group)
            if len(ids) == len(group) and len(ids) >= 2:
                values = mlx_affine_multi_prepare_strides(
                    _e_parts([group[i] for i in ids])
                )
                for i, value in zip(ids, values):
                    table[i] = value
            tables.append(tuple(table))
        # Object attributes are observed/copied with the model, not parameters.
        object.__setattr__(self, "_e_projection_sources", signature)
        object.__setattr__(self, "_e_projection_strides", tuple(tables))
        return True

    def project_verify_rows(self, x, *, selecting=True):
        indexer = self.indexer
        projections = [self.q_a_proj, self.kv_a_proj_with_mqa, indexer.wk]
        if selecting:
            projections.append(indexer.weights_proj)
        values = _project_multi_rows(
            projections,
            x,
            prepared_strides=(
                getattr(self, "_e_projection_strides", (None, None, None))[
                    1 if selecting else 0
                ]
            ),
        )
        qr = self.q_a_layernorm(values[0])
        kv = self.kv_a_layernorm(values[1])
        if selecting:
            q, iq = _project_multi_rows(
                (self.q_b_proj, indexer.wq_b),
                qr,
                prepared_strides=getattr(
                    self, "_e_projection_strides", (None, None, None)
                )[2],
            )
            iw = values[3]
        else:
            q, iq, iw = _project_rows(self.q_b_proj, qr), None, None
        index = (
            indexer.k_norm(values[2]),
            iq,
            iw,
            _project_weight_rows(indexer.index_kpool_compress_gate, x),
        )
        return qr, q, kv, index

    def _selection_needed(self, offset, rows):
        return not (
            offset + rows <= self.args.index_topk
            and self.args.index_kpool_always_select_tail
        )

    def verify_rows(
        self,
        x,
        cache,
        *,
        projected=None,
        project_output=True,
        attention_graph=None,
        graph_observation=None,
    ):
        """Batch projections and latent attention with explicit per-row visibility."""
        a = self.args
        b, count, _ = x.shape
        qr, q, new_kv, index_projection = (
            self.project_verify_rows(
                x, selecting=self._selection_needed(cache.offset, count)
            )
            if projected is None
            else projected
        )
        offset = cache.offset
        if isinstance(cache, MLACache):
            cache.append_latent(new_kv, materialize=False)
            kv = cache.cache[0]
        else:
            kv = (
                mx.concatenate([cache[0], new_kv], 1)
                if cache[0] is not None
                else new_kv
            )
        snapshots, selected = [], []
        index_rows, index_states = self.indexer.verify_rows(
            cache, offset, projected=index_projection
        )
        implicit = (
            b == 1
            and a.num_attention_heads == 64
            and a.kv_lora_rank == 512
            and all(v is None for v in index_rows)
        )
        selected_widths = []
        for j in range(count):
            length = offset + j + 1
            indices = index_rows[j]
            if indices is None:
                if not implicit:
                    indices = mx.arange(length)[None, None, :]
                width = length
            else:
                width = (
                    min(a.index_topk // a.index_kpool, length // a.index_kpool)
                    * a.index_kpool
                )
                if a.index_kpool_always_select_tail:
                    width += length % a.index_kpool
                indices = indices[..., :width]
            selected.append(indices)
            selected_widths.append(width if implicit else indices.shape[-1])
            snapshot = type(cache)(4)
            prefix = cache.cache[0] if isinstance(cache, MLACache) else kv[:, :length]
            snapshot.cache = [prefix] + index_states[j]
            snapshot.offset = length
            snapshots.append(snapshot)
        cache.cache = [
            v.view(v.dtype) if v is not None else None for v in snapshots[-1].cache
        ]
        cache.offset = offset + count
        width = max(selected_widths)
        if attention_graph is not None and not (
            a.num_attention_heads == 64 and a.kv_lora_rank == 512
        ):
            attention_graph = None
        if attention_graph is not None and width:
            width = max(32, 1 << (width - 1).bit_length())
        indices = (
            None
            if implicit
            else mx.concatenate(
                [
                    mx.pad(
                        v,
                        [(0, 0), (0, 0), (0, width - v.shape[-1])],
                        constant_values=-1,
                    )
                    for v in selected
                ],
                axis=1,
            ).astype(mx.int32)
        )
        prefix, tail = (
            cache.latent_segments() if isinstance(cache, MLACache) else (kv, None)
        )
        lengths = selected_widths
        visible_ends = mx.array(offset, mx.uint32) + mx.arange(
            1, count + 1, dtype=mx.uint32
        )
        if attention_graph is not None:
            out = attention_graph(
                q,
                prefix,
                tail,
                indices,
                mx.array(lengths, mx.uint32),
                visible_ends=visible_ends,
                indices_valid=True,
                implicit_width=width if implicit else None,
                observation=graph_observation,
            )
        else:
            out = self.attend_rows(
                q,
                prefix,
                tail,
                indices,
                lengths,
                visible_ends=visible_ends,
                indices_valid=True,
                implicit_width=width if implicit else None,
            )
        return (_project_rows(self.o_proj, out) if project_output else out), snapshots

    def attend_rows(
        self,
        q,
        prefix,
        tail,
        indices,
        lengths,
        *,
        split=None,
        visible_ends=None,
        indices_valid=False,
        implicit_width=None,
    ):
        """Stateless latent attention with runtime visible-length metadata."""
        a = self.args
        b, count, _ = q.shape
        covered = b == 1 and a.num_attention_heads == 64 and a.kv_lora_rank == 512
        if (
            indices is not None
            and covered
            and visible_ends is not None
            and not indices_valid
        ):
            indices, lengths = _f_visible_indices(
                indices.astype(mx.int32), lengths, visible_ends
            )
        width = implicit_width if indices is None else indices.shape[-1]
        q = q.reshape(b, count, a.num_attention_heads, a.qk_nope_head_dim).transpose(
            0, 2, 1, 3
        )
        wk, wv = self._latent_matrices(q.dtype)
        latent_q = mlx_headed_dense_rows(q, wk, transposed=True)
        if latent_q is None:
            latent_q = q @ wk
        if width == 0:
            latent_out = mx.zeros(latent_q.shape, latent_q.dtype)
        else:
            latent_out = mlx_ragged_latent_attention(
                latent_q,
                prefix,
                None if indices is None else indices.astype(mx.int32),
                lengths,
                implicit_width=implicit_width,
                scale=a.qk_nope_head_dim**-0.5,
                tail=tail,
                split=split,
            )
            if latent_out is None:
                if indices is None:
                    # The general SDPA path needs a materialized mask (e.g.
                    # FP32 queries with packed FP8 KV). Keep its existing math.
                    positions = mx.arange(width, dtype=mx.int32)
                    visible = (
                        lengths
                        if isinstance(lengths, mx.array)
                        else mx.array(lengths, mx.uint32)
                    )
                    indices = mx.where(
                        positions[None, None, :] < visible.reshape(1, count, 1),
                        positions,
                        -1,
                    )
                selected_kv = mlx_latent_gather_cast(
                    prefix,
                    indices.astype(mx.int32),
                    mx.result_type(
                        mx.bfloat16 if prefix.dtype == mx.uint8 else prefix.dtype,
                        latent_q.dtype,
                    ),
                    tail=tail,
                )
                if selected_kv is None:
                    kv = (
                        mx.concatenate([prefix, tail], 1)
                        if tail is not None
                        else prefix
                    )
                    selected_kv = mx.take_along_axis(
                        kv[:, None], mx.maximum(indices, 0)[..., None], axis=2
                    ).reshape(b * count, 1, width, a.kv_lora_rank)
                selected_q = latent_q.transpose(0, 2, 1, 3).reshape(
                    b * count,
                    a.num_attention_heads,
                    1,
                    a.kv_lora_rank,
                )
                latent_out = mx.fast.scaled_dot_product_attention(
                    selected_q,
                    selected_kv,
                    selected_kv,
                    scale=a.qk_nope_head_dim**-0.5,
                    mask=(indices >= 0).reshape(b * count, 1, 1, width),
                )
                # A row with no completed pool and tail selection disabled has an
                # empty legal set. SDPA's fully masked row must match serial zero.
                latent_out = mx.where(
                    mx.any(indices >= 0, axis=-1).reshape(b * count, 1, 1, 1),
                    latent_out,
                    mx.zeros_like(latent_out),
                )
                latent_out = latent_out.reshape(
                    b, count, a.num_attention_heads, a.kv_lora_rank
                ).transpose(
                    0,
                    2,
                    1,
                    3,
                )
        # FP32 promotion packs each sliced operand row-major in MLX. With
        # the original BF16 slice, Wv retains its transposed GEMV layout.
        transposed = latent_out.dtype == mx.float32
        operand = wv if transposed else wv.swapaxes(-1, -2)
        out = mlx_headed_dense_rows(latent_out, operand, transposed=transposed)
        if out is None:
            out = latent_out @ wv
        out = out.transpose(0, 2, 1, 3).reshape(b, count, -1)
        return out

    def decode_requests(self, rows, caches):
        if self.args.kv_lora_rank < 256:
            # Tiny reference fixtures use an expanded SDPA reduction rather
            # than the covered GLM latent kernel. Preserve that equation.
            return [self(row, cache) for row, cache in zip(rows, caches)]
        x = mx.concatenate(rows, axis=1)
        qr, q, kv, index = self.project_verify_rows(
            x, selecting=any(self._selection_needed(c.offset, 1) for c in caches)
        )
        outputs = []
        for j, cache in enumerate(caches):
            branch, _ = self.verify_rows(
                rows[j],
                cache,
                project_output=False,
                projected=(
                    qr[:, j : j + 1],
                    q[:, j : j + 1],
                    kv[:, j : j + 1],
                    tuple(v[:, j : j + 1] if v is not None else None for v in index),
                ),
            )
            outputs.append(branch)
        out = _project_rows(self.o_proj, mx.concatenate(outputs, 1))
        return [out[:, j : j + 1] for j in range(len(rows))]

    def __call__(self, x, cache):
        a = self.args
        b, t, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = (
            self.q_b_proj(qr)
            .reshape(b, t, a.num_attention_heads, a.qk_nope_head_dim)
            .transpose(0, 2, 1, 3)
        )
        kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        if isinstance(cache, MLACache):
            cache.append_latent(kv, materialize=False)
        elif cache[0] is not None:
            kv = mx.concatenate([cache[0], kv], axis=1)
        if not isinstance(cache, MLACache):
            cache[0] = kv
        indices = self.indexer(x, qr, cache, cache.offset)
        cache.offset += t
        if isinstance(cache, MLACache) and indices is None:
            kv = cache[0]
        scale = a.qk_nope_head_dim**-0.5
        if indices is None and t > 1:
            # Dense prefill reuses the expanded K/V across all queries. Its
            # smaller per-head dimension is faster than latent-space SDPA
            # here; decode and sparse attention still avoid historical expansion.
            expanded = (
                self.kv_b_proj(kv)
                .reshape(
                    b,
                    -1,
                    a.num_attention_heads,
                    a.qk_nope_head_dim + a.v_head_dim,
                )
                .transpose(0, 2, 1, 3)
            )
            k, v = mx.split(expanded, [a.qk_nope_head_dim], axis=-1)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal"
            )
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))
        # W_k and W_v map the shared latent to a head's K/V. Associate the
        # products as (Q W_k) C^T and (P C) W_v^T, rather than expanding every
        # historical latent C into all heads on each step.
        wk, wv = self._latent_matrices(q.dtype)
        latent_q = q @ wk
        if indices is not None and t == 1:
            # Every selected complete pool is visible to the final query.
            # Only the fixed-width tail has padding. Trim it before gathering
            # so decode needs neither dummy KV rows nor an array mask. At pool
            # boundaries this also restores an aligned attention key length.
            pool = a.index_kpool
            width = min(a.index_topk // pool, cache.offset // pool) * pool
            if a.index_kpool_always_select_tail:
                width += cache.offset % pool
            indices = indices[..., :width]
        if indices is None:
            latent_kv = kv[:, None]
            latent_out = mx.fast.scaled_dot_product_attention(
                latent_q,
                latent_kv,
                latent_kv,
                scale=scale,
                mask="causal" if t > 1 else None,
            )
        elif indices.shape[-1] == 0:
            latent_out = mx.zeros(latent_q.shape, dtype=latent_q.dtype)
        else:
            # One query per flattened batch row. Every head shares the same
            # selected compressed KV; do not tile it across heads. QK/PV now
            # scale with the selected budget, not the full history length.
            width = indices.shape[-1]
            prefix, tail = (
                cache.latent_segments() if isinstance(cache, MLACache) else (kv, None)
            )
            selected_kv = (
                mlx_latent_gather_cast(
                    prefix,
                    indices,
                    mx.result_type(
                        mx.bfloat16 if prefix.dtype == mx.uint8 else prefix.dtype,
                        latent_q.dtype,
                    ),
                    tail=tail,
                )
                if a.kv_lora_rank >= 256
                else None
            )
            if selected_kv is None:
                if isinstance(cache, MLACache):
                    kv = cache[0]
                selected_kv = mx.take_along_axis(
                    kv[:, None],
                    mx.maximum(indices, 0)[..., None],
                    axis=2,
                ).reshape(b * t, 1, width, a.kv_lora_rank)
            selected_q = latent_q.transpose(0, 2, 1, 3).reshape(
                b * t,
                a.num_attention_heads,
                1,
                a.kv_lora_rank,
            )
            latent_out = (
                mx.fast.scaled_dot_product_attention(
                    selected_q,
                    selected_kv,
                    selected_kv,
                    scale=scale,
                    mask=(
                        None if t == 1 else (indices >= 0).reshape(b * t, 1, 1, width)
                    ),
                )
                .reshape(b, t, a.num_attention_heads, a.kv_lora_rank)
                .transpose(0, 2, 1, 3)
            )
        out = latent_out @ wv
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))


def hyper_connection(x, fn, base, scale, args):
    return _hyper_connection(
        x,
        fn,
        base,
        scale,
        args.hc_mult,
        args.rms_norm_eps,
        args.hc_eps,
        args.hc_sinkhorn_iters,
    )


@mx.compile
def _hyper_connection(x, fn, base, scale, h, norm_eps, eps, sinkhorn_iters):
    # Compile the repeated small reductions/elementwise Sinkhorn operations
    # together. Shape-specialized compilation is required by mx.split here.
    dtype = x.dtype
    flat = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    weights = mx.fast.rms_norm(flat, None, norm_eps) @ fn.astype(mx.float32).T
    pre, post, comb = mx.split(weights, [h, 2 * h], axis=-1)
    pb, qb, cb = mx.split(base.astype(mx.float32), [h, 2 * h])
    pre = mx.sigmoid(pre * scale[0] + pb) + eps
    post = 2 * mx.sigmoid(post * scale[1] + qb)
    comb = mx.softmax((comb * scale[2] + cb).reshape(*x.shape[:2], h, h), axis=-1) + eps
    comb /= comb.sum(-2, keepdims=True) + eps
    for _ in range(sinkhorn_iters - 1):
        comb /= comb.sum(-1, keepdims=True) + eps
        comb /= comb.sum(-2, keepdims=True) + eps
    collapsed = (pre[..., None] * x).sum(-2).astype(dtype)
    return collapsed, post.astype(dtype), comb.astype(dtype).swapaxes(-1, -2) @ x


class DecoderLayer(nn.Module):
    compile_residual = False
    preserve_residual_dtype = False
    fuse_hc = False

    def __init__(self, args, index):
        super().__init__()
        self.args = args
        self.is_linear = args.layer_types[index] == "linear_attention"
        self.self_attn = KDA(args) if self.is_linear else MLA(args)
        self.mlp = MoE(args) if args.mlp_layer_types[index] == "sparse" else MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if args.mhc:
            h = args.hc_mult
            for name in ("attn", "ffn"):
                setattr(
                    self,
                    f"hc_{name}_fn",
                    mx.zeros(((2 + h) * h, h * args.hidden_size), mx.float32),
                )
                setattr(self, f"hc_{name}_base", mx.zeros(((2 + h) * h,), mx.float32))
                setattr(self, f"hc_{name}_scale", mx.ones((3,), mx.float32))

    def _branch_inputs(self, x, name, norm, *, mix=None):
        if not self.args.mhc:
            return norm(x), None, x
        if (
            self.fuse_hc
            and x.shape[0] == 1
            and 1 <= x.shape[1] <= 17
            and x.shape[2] == 4
        ):
            if mix is None:
                mix = mlx_hc_rms_mix(
                    x,
                    getattr(self, f"hc_{name}_fn"),
                    eps=self.args.rms_norm_eps,
                )
            if mix is None:
                flat = x.reshape(1, x.shape[1], -1).astype(mx.float32)
                weight = getattr(self, f"hc_{name}_fn").astype(mx.float32).T
                mix = mx.concatenate(
                    [
                        mx.fast.rms_norm(
                            flat[:, j : j + 1], None, self.args.rms_norm_eps
                        )
                        @ weight
                        for j in range(x.shape[1])
                    ],
                    axis=1,
                )
            fused = mlx_hc_collapse_norm(
                x,
                mix,
                getattr(self, f"hc_{name}_base"),
                getattr(self, f"hc_{name}_scale"),
                norm.weight,
                eps=self.args.hc_eps,
                norm_eps=norm.eps,
                iters=self.args.hc_sinkhorn_iters,
            )
            if fused is not None:
                return fused
        parts = [
            hyper_connection(
                x[:, j : j + 1],
                getattr(self, f"hc_{name}_fn"),
                getattr(self, f"hc_{name}_base"),
                getattr(self, f"hc_{name}_scale"),
                self.args,
            )
            for j in range(x.shape[1])
        ]
        return (
            mx.concatenate([norm(v[0]) for v in parts], 1),
            mx.concatenate([v[1] for v in parts], 1),
            mx.concatenate([v[2] for v in parts], 1),
        )

    def _branch_transition(self, branch, post, residual, name, norm):
        """Complete one branch and feed the next mHC through explicit arrays."""
        if self.preserve_residual_dtype:
            branch = branch.astype(residual.dtype)
        fused = (
            mlx_hc_residual_rms_mix(
                residual,
                post,
                branch,
                getattr(self, f"hc_{name}_fn"),
                eps=self.args.rms_norm_eps,
            )
            if self.args.mhc and self.fuse_hc and post is not None
            else None
        )
        if fused is not None:
            value, mix = fused
            return self._branch_inputs(value, name, norm, mix=mix)
        value = self._branch_output(branch, post, residual)
        return self._branch_inputs(value, name, norm)

    def _branch_output(self, branch, post, residual):
        if self.preserve_residual_dtype:
            branch = branch.astype(residual.dtype)
        if post is None:
            return residual + branch
        if self.compile_residual or self.fuse_hc:
            return _compiled_residual(residual, post, branch)
        return residual + post[..., None] * branch[..., None, :]

    def _decode_with_next_inputs(self, x, cache, prepared, next_layer):
        """Carry an explicit mHC transition between layers inside one graph.

        The value remains available for capture and the final output. Prepared
        next-layer inputs are local graph values, never fields on a live module.
        """
        normed, post, residual = (
            self._branch_inputs(x, "attn", self.input_layernorm)
            if prepared is None
            else prepared
        )
        branch = self.self_attn(normed, cache)
        normed, post, residual = self._branch_transition(
            branch, post, residual, "ffn", self.post_attention_layernorm
        )
        branch = self.mlp(normed)
        if self.preserve_residual_dtype:
            branch = branch.astype(residual.dtype)
        if next_layer is None:
            return self._branch_output(branch, post, residual), None
        fused = mlx_hc_residual_rms_mix(
            residual,
            post,
            branch,
            next_layer.hc_attn_fn,
            eps=next_layer.args.rms_norm_eps,
        )
        if fused is None:
            value = self._branch_output(branch, post, residual)
            return value, next_layer._branch_inputs(
                value, "attn", next_layer.input_layernorm
            )
        value, mix = fused
        return value, next_layer._branch_inputs(
            value, "attn", next_layer.input_layernorm, mix=mix
        )

    def __call__(self, x, cache, *, shared_stream=None, prefill_a8=False):
        if (
            self.args.mhc
            and self.fuse_hc
            and x.shape[:3] == (1, 1, 4)
            and x.dtype in (mx.float16, mx.bfloat16)
        ):
            normed, post, residual = self._branch_inputs(
                x, "attn", self.input_layernorm
            )
            branch = self.self_attn(normed, cache)
            value, post, residual = self._branch_transition(
                branch, post, residual, "ffn", self.post_attention_layernorm
            )
            branch = (
                self.mlp(value, shared_stream=shared_stream, prefill_a8=prefill_a8)
                if isinstance(self.mlp, MoE)
                else self.mlp(value)
            )
            return self._branch_output(branch, post, residual)
        for name, norm, block in (
            ("attn", self.input_layernorm, lambda y: self.self_attn(y, cache)),
            (
                "ffn",
                self.post_attention_layernorm,
                lambda y: (
                    self.mlp(y, shared_stream=shared_stream, prefill_a8=prefill_a8)
                    if isinstance(self.mlp, MoE)
                    else self.mlp(y)
                ),
            ),
        ):
            if self.args.mhc:
                if self.fuse_hc and x.shape[:3] == (1, 1, 4):
                    mix = mlx_hc_rms_mix(
                        x,
                        getattr(self, f"hc_{name}_fn"),
                        eps=self.args.rms_norm_eps,
                    )
                    if mix is None:
                        flat = x.reshape(1, 1, -1).astype(mx.float32)
                        mix = mx.fast.rms_norm(flat, None, self.args.rms_norm_eps)
                        mix = mix @ getattr(self, f"hc_{name}_fn").astype(mx.float32).T
                    fused = mlx_hc_collapse_norm(
                        x,
                        mix,
                        getattr(self, f"hc_{name}_base"),
                        getattr(self, f"hc_{name}_scale"),
                        norm.weight,
                        eps=self.args.hc_eps,
                        norm_eps=norm.eps,
                        iters=self.args.hc_sinkhorn_iters,
                    )
                    if fused is not None:
                        normed, post, residual = fused
                        branch = block(normed)
                        if self.preserve_residual_dtype:
                            branch = branch.astype(residual.dtype)
                        x = _compiled_residual(residual, post, branch)
                        continue
                collapsed, post, residual = hyper_connection(
                    x,
                    getattr(self, f"hc_{name}_fn"),
                    getattr(self, f"hc_{name}_base"),
                    getattr(self, f"hc_{name}_scale"),
                    self.args,
                )
                branch = block(norm(collapsed))
                if self.preserve_residual_dtype:
                    branch = branch.astype(residual.dtype)
                x = (
                    _compiled_residual(residual, post, branch)
                    if self.compile_residual
                    else residual + post[..., None] * branch[..., None, :]
                )
            else:
                branch = block(norm(x))
                if self.preserve_residual_dtype:
                    branch = branch.astype(x.dtype)
                x = x + branch
        return x

    def decode_requests(self, rows, caches, *, shared_stream=None):
        """Layerwise request batch with separate ragged attention/cache owners."""
        prepared = [self._branch_inputs(x, "attn", self.input_layernorm) for x in rows]
        branches = self.self_attn.decode_requests([v[0] for v in prepared], caches)
        attended = [
            self._branch_output(branch, post, residual)
            for branch, (_, post, residual) in zip(branches, prepared)
        ]
        normed, post, residual = self._branch_inputs(
            mx.concatenate(attended, 1), "ffn", self.post_attention_layernorm
        )
        branch = (
            self.mlp(normed, batch_fusion=True, shared_stream=shared_stream)
            if isinstance(self.mlp, MoE)
            else self.mlp(normed, batch_fusion=True)
        )
        out = self._branch_output(branch, post, residual)
        return [out[:, j : j + 1] for j in range(len(rows))]

    def verify_requests(self, rows, caches, *, shared_stream=None):
        """Share verification projections without mixing request recurrences.

        Each chain owns its attention snapshots. Stateless FFN and both large
        attention projections share weights across all chains, including ragged
        chain lengths. Row slices preserve singleton reductions and routing order.
        """
        counts = [(x).shape[1] for x in rows]
        prepared = [
            [
                self._branch_inputs((x)[:, j : j + 1], "attn", self.input_layernorm)
                for j in range((x).shape[1])
            ]
            for x in rows
        ]
        normed = mx.concatenate([v[0] for chain in prepared for v in chain], 1)
        projected = self.self_attn.project_verify_rows(normed)
        attended, snapshots, branches = [], [], []
        cursor = 0
        for x, cache, count, inputs in zip(rows, caches, counts, prepared):
            local = None
            if projected is not None:

                def slice_rows(value):
                    if isinstance(value, tuple):
                        return tuple(slice_rows(v) for v in value)
                    return (
                        value[:, cursor : cursor + count] if value is not None else None
                    )

                local = tuple(slice_rows(v) for v in projected)
            result = self.self_attn.verify_rows(
                normed[:, cursor : cursor + count],
                cache,
                projected=local,
                project_output=False,
            )
            if result is None:
                chain, states = [], []
                for value, post, residual in inputs:
                    chain.append(
                        self._branch_output(
                            self.self_attn(value, cache), post, residual
                        )
                    )
                    state = type(cache)(4)
                    state.cache = [
                        v.view(v.dtype) if v is not None else None for v in cache.cache
                    ]
                    state.offset = cache.offset
                    states.append(state)
                attended.append(chain)
                branches.append(None)
                snapshots.append(states)
            else:
                branch, states = result
                branches.append(branch)
                attended.append(None)
                snapshots.append(states)
            cursor += count
        valid = [v for v in branches if v is not None]
        if valid:
            output = _project_rows(self.self_attn.o_proj, mx.concatenate(valid, 1))
            cursor = 0
            for i, (branch, inputs, count) in enumerate(
                zip(branches, prepared, counts)
            ):
                if branch is not None:
                    attended[i] = [
                        self._branch_output(
                            output[:, cursor + j : cursor + j + 1], post, residual
                        )
                        for j, (_, post, residual) in enumerate(inputs)
                    ]
                    cursor += count
        value, post, residual = self._branch_inputs(
            mx.concatenate([v for chain in attended for v in chain], 1),
            "ffn",
            self.post_attention_layernorm,
        )
        branches = []
        for start in range(0, (value).shape[1], 17):
            tile = value[:, start : start + 17]
            branches.append(
                self.mlp(tile, batch_fusion=True, shared_stream=shared_stream)
                if isinstance(self.mlp, MoE)
                else self.mlp(tile, batch_fusion=True)
            )
        branch = mx.concatenate(branches, 1)
        result = self._branch_output(branch, post, residual)
        outputs, cursor = [], 0
        for count in counts:
            outputs.append((result)[:, cursor : cursor + count])
            cursor += count
        return outputs, snapshots

    def verify_ffn_rows(self, x, cache, *, shared_stream=None):
        """Batch stateless FFN dispatch, retaining per-position attention state."""
        fixed = getattr(self, "_fixed_mla_layer", None)
        if fixed is not None and shared_stream is None:
            return fixed(x, cache)
        attended, snapshots = [], []
        # mHC rows are independent; submit their collapse/projection together.
        # Attention still returns an explicit cache snapshot for every position.
        normed_rows, attention_post, attention_residual = self._branch_inputs(
            x, "attn", self.input_layernorm
        )
        if (
            not self.is_linear
            and getattr(self, "_fixed_mla_attention", None) is not None
        ):
            batched = self.self_attn.verify_rows(
                normed_rows, cache, attention_graph=self._fixed_mla_attention
            )
        else:
            batched = self.self_attn.verify_rows(normed_rows, cache)
        if batched is not None:
            branches, snapshots = batched
            attended.append(
                self._branch_output(branches, attention_post, attention_residual)
            )
        else:
            for j in range(x.shape[1]):
                normed = normed_rows[:, j : j + 1]
                post = None if attention_post is None else attention_post[:, j : j + 1]
                residual = attention_residual[:, j : j + 1]
                branch = self.self_attn(normed, cache)
                attended.append(self._branch_output(branch, post, residual))
                snapshot = type(cache)(4)
                snapshot.cache = [
                    v.view(v.dtype) if v is not None else None for v in cache.cache
                ]
                snapshot.offset = cache.offset
                snapshots.append(snapshot)
        if batched is not None:
            normed, post, residual = self._branch_transition(
                branches,
                attention_post,
                attention_residual,
                "ffn",
                self.post_attention_layernorm,
            )
        else:
            normed, post, residual = self._branch_inputs(
                mx.concatenate(attended, 1), "ffn", self.post_attention_layernorm
            )
        branch = (
            self.mlp(normed, batch_fusion=True, shared_stream=shared_stream)
            if isinstance(self.mlp, MoE)
            else self.mlp(normed, batch_fusion=True)
        )
        return self._branch_output(branch, post, residual), snapshots


class MTPHead(DecoderLayer):
    """Checkpoint's plain MLA/MoE prediction layer (no hyper-connections).

    The head consumes final-normalized target hidden states paired with the
    following tokens. Its output normalization and the target LM head are
    shared across recursive draft steps, while its attention cache is private.
    """

    def __init__(self, args):
        args = copy.copy(args)
        args.mhc = False
        index = args.num_hidden_layers
        args.layer_types = list(args.layer_types) + ["deepseek_sparse_attention"]
        args.mlp_layer_types = list(args.mlp_layer_types) + ["sparse"]
        super().__init__(args, index)
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.shared_head = nn.Module()
        self.shared_head.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def consume_rows(self, hidden, embeddings, cache):
        """Advance consumed head history once with singleton-equivalent rows."""
        if hidden.shape[1] == 1:
            return self(hidden, embeddings, cache)
        if hidden.shape[1] > 8:
            # Keep each head tile below the SwitchGLU prefill sorting boundary;
            # changing to grouped QMM changes the singleton reduction tree.
            return mx.concatenate(
                [
                    self.consume_rows(
                        hidden[:, start : start + 8],
                        embeddings[:, start : start + 8],
                        cache,
                    )
                    for start in range(0, hidden.shape[1], 8)
                ],
                axis=1,
            )
        joined = mx.concatenate(
            [
                mx.concatenate(
                    [
                        self.enorm(embeddings[:, j : j + 1]),
                        self.hnorm(hidden[:, j : j + 1]),
                    ],
                    -1,
                )
                for j in range(hidden.shape[1])
            ],
            axis=1,
        )
        x = _project_rows(self.eh_proj, joined)
        if x.shape[1] == 1:
            return super().__call__(x, cache)
        return self.verify_ffn_rows(x, cache)[0]

    def consume_requests(self, hidden, embeddings, caches):
        """Consume ragged committed history and root together for private head caches."""
        blocks = []
        counts = []
        for values, tokens, state in zip(hidden, embeddings, caches):
            # Keep historical tiles at the existing eight-row expert boundary.
            start = 0
            while values.shape[1] - start > 8:
                self.consume_rows(
                    values[:, start : start + 8], tokens[:, start : start + 8], state
                )
                start += 8
            values, tokens = values[:, start:], tokens[:, start:]
            counts.append(values.shape[1])
            blocks.append(
                mx.concatenate(
                    [
                        mx.concatenate(
                            [
                                self.enorm(tokens[:, j : j + 1]),
                                self.hnorm(values[:, j : j + 1]),
                            ],
                            -1,
                        )
                        for j in range(values.shape[1])
                    ],
                    1,
                )
            )
        projected = _project_rows(self.eh_proj, mx.concatenate(blocks, 1))
        cursor = 0
        rows = []
        for count in counts:
            rows.append(projected[:, cursor : cursor + count])
            cursor += count
        outputs, _ = super().verify_requests(rows, caches)
        return [value[:, -1:] for value in outputs]

    def decode_requests(self, hidden, embeddings, caches, *, shared_stream=None):
        joined = mx.concatenate(
            [
                mx.concatenate([self.enorm(token), self.hnorm(value)], -1)
                for value, token in zip(hidden, embeddings)
            ],
            1,
        )
        x = _project_rows(self.eh_proj, joined)
        outputs, _ = super().verify_requests(
            [x[:, j : j + 1] for j in range(len(caches))],
            caches,
            shared_stream=shared_stream,
        )
        return outputs

    def __call__(self, hidden, embeddings, cache):
        if hidden.shape != embeddings.shape or hidden.ndim != 3:
            raise ValueError("MTP hidden states must match following-token embeddings")
        x = self.eh_proj(
            mx.concatenate([self.enorm(embeddings), self.hnorm(hidden)], -1)
        )
        return super().__call__(x, cache)


@cache
def _graph_scalar_defaults(module_type):
    fields = {}
    for cls in reversed(module_type.__mro__):
        for key, value in vars(cls).items():
            if not key.startswith("__") and isinstance(value, (bool, int, float, str)):
                fields[key] = value
    return tuple(fields)


def _graph_configuration(modules):
    """Observe captured Python metadata without evaluating device arrays.

    Array identities are keyed separately. Derived caches and graph owners do
    not affect the model equation. Shared args are visited once per call and
    class flag discovery is cached. Shared containers appear once in the flat
    signature, so hashing the key does not repeatedly walk shared model args.
    """
    excluded = {
        "_fixed_mla_layer",
        "_fixed_mla_attention",
        "_latent_projection",
        "_no_grad",
        "_training",
        "_gate_up_sources",
        "_router_sources",
        "_fused_qkv_sources",
        "_e_projection_sources",
    }
    records, pending, memo = [], [modules], {id(modules): 0}
    primitive = (bool, int, float, str, type(mx.float32))
    for value in pending:
        if isinstance(value, (list, tuple)):
            fields = enumerate(value)
        elif isinstance(value, (nn.Module, ModelArgs, dict)):
            fields = dict(value) if isinstance(value, dict) else {}
            if hasattr(value, "__dict__"):
                fields.update(vars(value))
            if isinstance(value, nn.Module):
                for key in _graph_scalar_defaults(type(value)):
                    fields.setdefault(key, getattr(value, key))
            fields = fields.items()
        else:
            records.append((type(value), id(value)))
            continue
        entries = []
        for key, child in fields:
            if key in excluded or isinstance(child, mx.array):
                continue
            if child is None or isinstance(child, primitive):
                entries.append((key, child))
            elif callable(child) and not isinstance(child, nn.Module):
                continue
            else:
                ident = id(child)
                index = memo.get(ident)
                if index is None:
                    index = len(pending)
                    memo[ident] = index
                    pending.append(child)
                entries.append((key, ("ref", index)))
        records.append((type(value), tuple(entries)))
    return tuple(records)


class _GraphObservation:
    """Reuse observation paths and parameter containers until live state changes.

    Check container membership, child identities, scalar values and class
    dictionaries on every transaction. Arrays remain explicit graph inputs;
    neither this plan nor graph clones copy their device storage.
    """

    _excluded = frozenset(
        {
            "_fixed_mla_layer",
            "_fixed_mla_attention",
            "_latent_projection",
            "_no_grad",
            "_training",
            "_gate_up_sources",
            "_router_sources",
            "_fused_qkv_sources",
            "_e_projection_sources",
        }
    )

    def __init__(self, modules):
        self.modules = modules
        self.nodes = []
        self.classes = {}
        pending, seen = [modules], {id(modules)}
        primitive = (bool, int, float, str, type(mx.float32))
        for value in pending:
            if isinstance(value, (list, tuple)):
                fields = tuple(enumerate(value))
                keys = None
            elif isinstance(value, (nn.Module, ModelArgs, dict)):
                fields = dict(value) if isinstance(value, dict) else {}
                fields.update(vars(value) if hasattr(value, "__dict__") else {})
                keys = (
                    tuple(value) if isinstance(value, dict) else (),
                    tuple(vars(value)) if hasattr(value, "__dict__") else (),
                )
                if isinstance(value, nn.Module):
                    for cls in type(value).__mro__:
                        self.classes.setdefault(cls, dict(vars(cls)))
                    # Discover current defaults on rebuild, including added flags.
                    defaults = {}
                    for cls in reversed(type(value).__mro__):
                        defaults.update(
                            {
                                k: v
                                for k, v in vars(cls).items()
                                if not k.startswith("__")
                                and isinstance(v, (bool, int, float, str))
                            }
                        )
                    for key in defaults:
                        fields.setdefault(key, getattr(value, key))
                fields = tuple(fields.items())
            else:
                continue
            observed = []
            for key, child in fields:
                if key in self._excluded:
                    continue
                scalar = child is None or isinstance(child, primitive)
                observed.append((key, scalar, child if scalar else id(child)))
                if (
                    not scalar
                    and not isinstance(child, mx.array)
                    and not (callable(child) and not isinstance(child, nn.Module))
                ):
                    if id(child) not in seen:
                        seen.add(id(child))
                        pending.append(child)
            length = len(value) if isinstance(value, (list, tuple)) else None
            self.nodes.append((value, keys, length, observed))
        # A plan identity is a configuration epoch. Structure and values are
        # checked before reusing it; changed state creates a fresh epoch.
        self.key = object()
        self.parameters = (
            [m.parameters() for m in modules]
            if isinstance(modules, (list, tuple))
            else modules.parameters()
        )
        self.sources = tuple(id(v) for _, v in tree_flatten(self.parameters))

    def current(self, modules):
        if modules is not self.modules:
            return False
        for cls, fields in self.classes.items():
            if vars(cls) != fields:
                return False
        for value, keys, length, observed in self.nodes:
            if keys is None:
                if len(value) != length:
                    return False
            elif keys != (
                tuple(value) if isinstance(value, dict) else (),
                tuple(vars(value)) if hasattr(value, "__dict__") else (),
            ):
                return False
            for key, scalar, previous in observed:
                child = (
                    value[key]
                    if keys is None or isinstance(value, dict) and key in value
                    else getattr(value, key)
                )
                if scalar:
                    if type(child) is not type(previous) or child != previous:
                        return False
                elif id(child) != previous:
                    return False
        return True


def _observe_graph(owner, modules):
    _prepare_e_modules(modules)
    # Run on live modules before parameters are collected or clones updated
    # with tracers. No operand identity checks or layout readbacks in capture.
    plan = getattr(owner, "_observation", None)
    if plan is None or not plan.current(modules):
        plan = owner._observation = _GraphObservation(modules)
    return (plan.key, _e_layout_key(modules)), plan.parameters, plan.sources


def _clone_graph_modules(modules, excluded=()):
    """Clone module containers while sharing array storage and compiled callables."""
    memo = {id(value): None for value in excluded if value is not None}
    visited = set()

    def visit(value):
        if id(value) in visited or id(value) in memo:
            return
        visited.add(id(value))
        if isinstance(value, (mx.array, type(mx.float32))) or (
            callable(value) and not isinstance(value, nn.Module)
        ):
            memo[id(value)] = value
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
            for child in vars(value).values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(modules)
    return copy.deepcopy(modules, memo)


class _FixedGraphOwner:
    def _remember_graph(self, spec, graph, configuration):
        # Retain common verification depths and capacity buckets. Model changes
        # invalidate all layouts; layout churn has a bounded storage cost.
        if getattr(self, "_graph_configuration_key", None) != configuration:
            self.graphs.clear()
            self._graph_configuration_key = configuration
        if len(self.graphs) >= 8:
            del self.graphs[next(iter(self.graphs))]
        self.graphs[spec] = graph

    def __deepcopy__(self, memo):
        # Compiled functions close over private module containers and are not
        # pickleable. A copied model must compile its own functions rather than
        # inherit closures referring to the source model's weights or caches.
        result = object.__new__(type(self))
        memo[id(self)] = result
        for name, value in vars(self).items():
            setattr(
                result,
                name,
                (
                    {}
                    if name == "graphs"
                    else None
                    if name == "_observation"
                    else copy.deepcopy(value, memo)
                ),
            )
        return result


class _FixedMLAAttention(_FixedGraphOwner):
    """Compile the latent subgraph with explicit indices, lengths and split.

    Indexer cache transitions stay at their existing ownership boundary. The
    KV segments use 256-row capacity buckets. Valid lengths and the logical
    segment split are device inputs, so advancing a token reuses the same graph.
    """

    def __init__(self, attention):
        self.attention = attention
        self.graphs = {}

    def __call__(
        self,
        q,
        prefix,
        tail,
        indices,
        lengths,
        *,
        observation=None,
        visible_ends=None,
        indices_valid=False,
        implicit_width=None,
    ):
        if tail is None:
            tail = prefix[:, :0]
        split = prefix.shape[1]
        if visible_ends is None:
            visible_ends = mx.full((q.shape[1],), split + tail.shape[1], mx.uint32)
        # Only the first short prefix grows per token. Long sealed prefixes
        # already have a stable shape; padding them would copy full history.
        if split < 256:
            prefix = mx.pad(prefix, [(0, 0), (0, 256 - split), (0, 0)])
        tail_capacity = max(256, ((tail.shape[1] + 255) // 256) * 256)
        tail = mx.pad(tail, [(0, 0), (0, tail_capacity - tail.shape[1]), (0, 0)])
        configuration, parameter, sources = (
            _observe_graph(self, self.attention) if observation is None else observation
        )
        spec = (
            (configuration, indices_valid, indices is None),
            q.shape,
            q.dtype,
            indices.shape if indices is not None else (1, q.shape[1], implicit_width),
            prefix.shape,
            tail.shape,
            prefix.dtype,
            sources,
        )
        graph = self.graphs.get(spec)
        if graph is None:
            attention = _clone_graph_modules(self.attention)
            attention.cache_latent_projection = False

            def forward(
                query, left, right, table, visible, causal_ends, split, weights
            ):
                attention.update(weights)
                return attention.attend_rows(
                    query,
                    left,
                    right,
                    table,
                    visible,
                    split=split,
                    visible_ends=causal_ends,
                    indices_valid=indices_valid,
                    implicit_width=implicit_width,
                )

            graph = mx.compile(forward)
            self._remember_graph(spec, graph, (spec[0], spec[7]))
        return graph(
            q,
            prefix,
            tail,
            indices,
            lengths,
            visible_ends,
            mx.array(split, mx.uint32),
            parameter,
        )


class _FixedMLALayer(_FixedGraphOwner):
    """Pure projection and FFN graphs surrounding owned index/cache updates."""

    def __init__(self, layer):
        self.layer = layer
        self.graphs = {}
        self.attention = _FixedMLAAttention(layer.self_attn)

    def __call__(self, x, cache):
        configuration, weights, sources = _observe_graph(self, self.layer)
        selecting = self.layer.self_attn._selection_needed(cache.offset, (x).shape[1])
        spec = (
            configuration,
            (x).shape,
            (x).dtype,
            self.layer.fuse_hc,
            self.layer.compile_residual,
            self.layer.preserve_residual_dtype,
            sources,
            selecting,
        )
        graphs = self.graphs.get(spec)
        if graphs is None:
            layer = _clone_graph_modules(
                self.layer,
                (
                    self,
                    getattr(self.layer, "_fixed_mla_attention", None),
                ),
            )
            layer._fixed_mla_layer = None
            layer._fixed_mla_attention = None

            def prepare(value, parameters):
                layer.update(parameters)
                normed, post, residual = layer._branch_inputs(
                    value, "attn", layer.input_layernorm
                )
                parts = post, residual
                return (
                    parts,
                    normed,
                    layer.self_attn.project_verify_rows(normed, selecting=selecting),
                )

            def finish(branch, parts, parameters, router_weight):
                layer.update(parameters)
                if isinstance(layer.mlp, MoE):
                    layer.mlp._router_weight = router_weight
                    layer.mlp._router_sources = (
                        layer.mlp._current_router_sources()
                        if router_weight is not None
                        else ()
                    )
                value, post, residual = layer._branch_transition(
                    branch, *parts, "ffn", layer.post_attention_layernorm
                )
                branch = (
                    layer.mlp(value, batch_fusion=True)
                    if isinstance(layer.mlp, MoE)
                    else layer.mlp(value, batch_fusion=True)
                )
                return layer._branch_output(branch, post, residual)

            graphs = mx.compile(prepare), mx.compile(finish)
            self._remember_graph(spec, graphs, (spec[0], spec[3:]))
        parts, normed, projected = graphs[0](x, weights)
        # A replaced attention module belongs to this transaction's layer.
        self.attention.attention = self.layer.self_attn
        branch, snapshots = self.layer.self_attn.verify_rows(
            normed,
            cache,
            projected=projected,
            attention_graph=self.attention,
            graph_observation=(configuration, weights["self_attn"], sources),
        )
        router_weight = None
        if isinstance(self.layer.mlp, MoE):
            block = self.layer.mlp
            if block._router_weight is not None and all(
                old is new
                for old, new in zip(
                    block._router_sources, block._current_router_sources()
                )
            ):
                router_weight = block._router_weight
        return graphs[1](branch, parts, weights, router_weight), snapshots


class _FixedLayerGroup(_FixedGraphOwner):
    """Stable graph identity, explicit weights, and explicit cache outputs.

    Cache objects created during tracing are private. The traced function
    returns their arrays; offsets advance in the caller using input length.
    Input layout is part of the key, so a seal/split retraces deliberately.
    """

    def __init__(self, layers):
        self.layers = layers
        self.graphs = {}

    def __call__(self, x, states, *, verification=False, layers=None, prefill_a8=False):
        if layers is not None and (
            len(layers) != len(self.layers)
            or any(a is not b for a, b in zip(layers, self.layers))
        ):
            self.layers = layers
        configuration, parameters, sources = _observe_graph(self, self.layers)
        flags = tuple(
            (
                layer.fuse_hc,
                layer.compile_residual,
                layer.preserve_residual_dtype,
                layer.self_attn.fuse_decode,
                layer.self_attn._fused_qkv_is_current(),
            )
            for layer in self.layers
        )
        spec = (
            configuration,
            x.shape,
            x.dtype,
            sources,
            flags,
            verification,
            prefill_a8,
            tuple(
                (
                    type(state),
                    tuple(
                        None if v is None else (v.shape, v.dtype) for v in state.cache
                    ),
                )
                for state in states
            ),
        )
        graph = self.graphs.get(spec)
        if graph is None:
            # Weights are explicit compiled inputs; cloned module dictionaries
            # are never exposed to the live serving model.
            # Clone module containers, sharing every original array storage.
            # A graph owner must never duplicate resident checkpoint weights.
            layers = _clone_graph_modules(self.layers)
            for original, layer in zip(self.layers, layers):
                if not original.self_attn._fused_qkv_is_current():
                    layer.self_attn._fused_qkv = None
                    layer.self_attn._fused_qkv3 = None
            cache_types = tuple(type(state) for state in states)

            def forward(value, weights, arrays, fused_weights):
                outputs, boundaries = [], []
                next_inputs = None
                # Next-layer mHC weights are explicit graph operands too.
                for layer, parameter in zip(layers, weights):
                    layer.update(parameter)
                for layer_index, (
                    layer,
                    parameter,
                    array,
                    cache_type,
                    prepared,
                ) in enumerate(
                    zip(layers, weights, arrays, cache_types, fused_weights)
                ):
                    fused, expert_weights, router_weight = prepared
                    layer.update(parameter)
                    attention = layer.self_attn
                    for projection, values in zip(
                        (attention._fused_qkv, attention._fused_qkv3), fused
                    ):
                        if projection is not None:
                            projection.update(values)
                    if attention._fused_qkv is not None:
                        attention._fused_qkv_sources = attention._qkv_sources()
                    if isinstance(layer.mlp, MoE):
                        layer.mlp._router_weight = router_weight
                        layer.mlp._router_sources = layer.mlp._current_router_sources()
                        switch = layer.mlp.switch_mlp
                        if (
                            expert_weights is not None
                            and switch._fused_gate_up is not None
                        ):
                            switch._fused_gate_up.update(expert_weights)
                            switch._gate_up_sources = switch._current_gate_up_sources()
                    local = cache_type(4)
                    local.cache = list(array)
                    local.offset = 0
                    if verification:
                        value, snapshots = layer.verify_ffn_rows(value, local)
                        boundaries.append([snapshot.cache for snapshot in snapshots])
                    elif (
                        type(layer) is DecoderLayer
                        and layer.args.mhc
                        and layer.fuse_hc
                        and value.shape[:3] == (1, 1, 4)
                        and value.dtype in (mx.float16, mx.bfloat16)
                    ):
                        following = (
                            layers[layer_index + 1]
                            if layer_index + 1 < len(layers)
                            else None
                        )
                        if following is not None and not (
                            type(following) is DecoderLayer
                            and following.fuse_hc
                            and following.args.mhc
                        ):
                            following = None
                        value, next_inputs = layer._decode_with_next_inputs(
                            value, local, next_inputs, following
                        )
                    else:
                        value = layer(value, local, prefill_a8=prefill_a8)
                        next_inputs = None
                    outputs.append(local.cache)
                return value, outputs, boundaries

            graph = mx.compile(forward)
            self._remember_graph(spec, graph, (spec[0], sources, flags))
        prepared = []
        for layer in self.layers:
            attention = tuple(
                p.parameters() if p is not None else None
                for p in (layer.self_attn._fused_qkv, layer.self_attn._fused_qkv3)
            )
            expert_weights, router = None, None
            if isinstance(layer.mlp, MoE):
                switch = layer.mlp.switch_mlp
                if (
                    switch._fused_gate_up is not None
                    and not switch.compile_decode
                    and getattr(switch, "_gate_up_metadata", None)
                    == switch._current_gate_up_metadata()
                    and all(
                        a is b
                        for a, b in zip(
                            switch._gate_up_sources, switch._current_gate_up_sources()
                        )
                    )
                ):
                    expert_weights = switch._fused_gate_up.parameters()
                if all(
                    a is b
                    for a, b in zip(
                        layer.mlp._router_sources, layer.mlp._current_router_sources()
                    )
                ):
                    router = layer.mlp._router_weight
            prepared.append((attention, expert_weights, router))
        result, arrays, boundaries = graph(
            x, parameters, [state.cache for state in states], prepared
        )
        snapshots = [[] for _ in range(x.shape[1])]
        if verification:
            for state, layer_boundaries in zip(states, boundaries):
                for j, values in enumerate(layer_boundaries):
                    snapshot = type(state)(4)
                    snapshot.cache = list(values)
                    snapshot.offset = state.offset + j + 1
                    snapshots[j].append(snapshot)
        for state, values in zip(states, arrays):
            state.cache = list(values)
            state.offset += x.shape[1]
        return (result, snapshots) if verification else result


class TextModel(nn.Module):
    # Submit one KDA/MLA period at a time. Async evaluation bounds the lazy
    # graph without making the host wait after each individual decoder layer.
    _eval_group_size = 4
    _decode_eval_group_size = 4
    _verification_eval_group_size = 4

    def __init__(self, args, enable_mtp=False):
        super().__init__()
        self.args = args
        self._shared_stream = None
        self._compile_groups = True
        self._fixed_groups = {}
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        if enable_mtp:
            self.layers.append(MTPHead(args))
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        for layer in self.layers[: args.num_hidden_layers]:
            if not layer.is_linear:
                layer._fixed_mla_attention = _FixedMLAAttention(layer.self_attn)
                layer._fixed_mla_layer = _FixedMLALayer(layer)

    def __call__(
        self, inputs, cache, *, capture_layer_ids=(), inputs_embeds=None, prefill=False
    ):
        prefill_a8 = (
            prefill and getattr(self, "_prefill_activation_dtype", "bf16") == "fp8"
        )
        capture_layer_ids = tuple(capture_layer_ids)
        captures = {}
        x = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        group_size = (
            self._decode_eval_group_size
            if inputs.shape[1] == 1
            else self._eval_group_size
        )
        if self._compile_groups and self._shared_stream is None:
            for start in range(0, self.args.num_hidden_layers, group_size):
                end = min(start + group_size, self.args.num_hidden_layers)
                cursor = start
                while cursor < end:
                    if not self.layers[cursor].is_linear:
                        layer = self.layers[cursor]
                        x = layer(x, cache[cursor], prefill_a8=prefill_a8)
                        if cursor in capture_layer_ids:
                            captures[cursor] = x.mean(-2) if self.args.mhc else x
                        cursor += 1
                        continue
                    stop = cursor + 1
                    while (
                        stop < end
                        and self.layers[stop].is_linear
                        and stop - 1 not in capture_layer_ids
                    ):
                        stop += 1
                    key = (cursor, stop)
                    group = self._fixed_groups.get(key)
                    if group is None:
                        group = _FixedLayerGroup(self.layers[cursor:stop])
                        self._fixed_groups[key] = group
                    x = group(
                        x,
                        cache[cursor:stop],
                        layers=self.layers[cursor:stop],
                        prefill_a8=prefill_a8,
                    )
                    if stop - 1 in capture_layer_ids:
                        captures[stop - 1] = x.mean(-2) if self.args.mhc else x
                    cursor = stop
                mx.async_eval(x, [c.cache for c in cache[start:end]])
        else:
            group_start = 0
            for i, (layer, state) in enumerate(
                zip(self.layers[: self.args.num_hidden_layers], cache)
            ):
                x = layer(
                    x,
                    cache=state,
                    shared_stream=self._shared_stream,
                    prefill_a8=prefill_a8,
                )
                if i in capture_layer_ids:
                    captures[i] = x.mean(-2) if self.args.mhc else x
                final_layer = i + 1 == self.args.num_hidden_layers
                if (i + 1) % group_size == 0 or final_layer:
                    mx.async_eval(x, [c.cache for c in cache[group_start : i + 1]])
                    group_start = i + 1
        if self.args.mhc:
            x = x.mean(axis=-2)
        hidden = self.norm(x)
        if capture_layer_ids:
            return hidden, mx.concatenate([captures[i] for i in capture_layer_ids], -1)
        return hidden


class Model(nn.Module):
    # Consumed by SGLang's runner: no ordinary q/k/v attention cache adapter.
    native_cache = True

    def __init__(self, args, *, enable_mtp=False):
        super().__init__()
        self.args = args
        if enable_mtp and getattr(args, "num_nextn_predict_layers", 0) != 1:
            raise ValueError("GLM MLX MTP requires exactly one prediction layer")
        self.model = TextModel(args, enable_mtp=enable_mtp)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers[: self.args.num_hidden_layers]

    @property
    def mtp(self):
        layers = self.model.layers
        return layers[-1] if len(layers) > self.args.num_hidden_layers else None

    def decode_requests(self, inputs, caches, *, token_only=False):
        """Ragged request contract: one token and independent layer caches each."""
        if not inputs or len(inputs) != len(caches):
            raise ValueError("One cache owner is required for each decode request")
        if any(v.shape != (1, 1) for v in inputs):
            raise ValueError("Request decode batches consume one token per request")
        if any(len(c) != len(self.layers) for c in caches):
            raise ValueError("Each request must own one cache per target layer")
        if len(inputs) > 17:
            return mx.concatenate(
                [
                    self.decode_requests(
                        inputs[i : i + 17], caches[i : i + 17], token_only=token_only
                    )
                    for i in range(0, len(inputs), 17)
                ],
                axis=0,
            )
        x = self.model.embed_tokens(mx.concatenate(inputs, axis=1))
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        rows = [x[:, j : j + 1] for j in range(len(inputs))]
        group_start = 0
        for i, layer in enumerate(self.layers):
            rows = layer.decode_requests(
                rows, [c[i] for c in caches], shared_stream=self.model._shared_stream
            )
            if (i + 1) % self.model._decode_eval_group_size == 0 or i + 1 == len(
                self.layers
            ):
                mx.async_eval(
                    rows,
                    [
                        [c.cache for c in states[group_start : i + 1]]
                        for states in caches
                    ],
                )
                group_start = i + 1
        hidden = mx.concatenate(
            [self.model.norm(v.mean(-2) if self.args.mhc else v) for v in rows], axis=1
        )
        if token_only:
            return self.target_tokens(hidden).swapaxes(0, 1)
        logits = _project_rows(self.lm_head, hidden)
        return logits.swapaxes(0, 1)

    def target_tokens(self, hidden):
        """Explicit greedy contract: no logit edits, processors or logprobs."""
        if hidden.shape[1] > 17:
            return mx.concatenate(
                [
                    self.target_tokens(hidden[:, start : start + 17])
                    for start in range(0, hidden.shape[1], 17)
                ],
                1,
            )
        tokens = (
            mlx_dense_argmax_rows(hidden, self.lm_head.weight)
            if type(self.lm_head) is nn.Linear and "bias" not in self.lm_head
            else None
        )
        return (
            tokens
            if tokens is not None
            else mx.argmax(_project_rows(self.lm_head, hidden), -1)
        )

    def forward_last_tokens(self, inputs, cache=None):
        hidden = self.model(inputs, cache if cache is not None else self.make_cache())
        return self.target_tokens(hidden[:, -1:])

    def forward_with_hidden(self, inputs, cache=None, *, token_only=False):
        hidden = self.model(inputs, cache if cache is not None else self.make_cache())
        output = self.target_tokens(hidden) if token_only else self.lm_head(hidden)
        return output, hidden

    def forward_with_capture(
        self, inputs, cache, capture_layer_ids, *, token_only=False
    ):
        """Capture completed layer outputs before final norm, with mHC contracted."""
        hidden, features = self.model(
            inputs, cache, capture_layer_ids=capture_layer_ids
        )
        logits = self.target_tokens(hidden) if token_only else self.lm_head(hidden)
        return logits, hidden, features

    def draft_token(self, hidden):
        normalized = self.mtp.shared_head.norm(hidden)
        token = None
        if type(self.lm_head) is nn.Linear and "bias" not in self.lm_head:
            token = mlx_dense_argmax(normalized, self.lm_head.weight)
        return mx.argmax(self.lm_head(normalized), axis=-1) if token is None else token

    def draft_tokens(self, hidden):
        normalized = mx.concatenate([self.mtp.shared_head.norm(v) for v in hidden], 1)
        token = (
            mlx_dense_argmax_rows(normalized, self.lm_head.weight)
            if type(self.lm_head) is nn.Linear and "bias" not in self.lm_head
            else None
        )
        if token is None:
            token = mx.argmax(_project_rows(self.lm_head, normalized), -1)
        return [token[:, j : j + 1] for j in range(len(hidden))]

    def forward_mtp(self, following_tokens, hidden, cache):
        if self.mtp is None:
            raise ValueError("Load the model with enable_mtp=True before drafting")
        out = self.mtp(hidden, self.model.embed_tokens(following_tokens), cache)
        return self.lm_head(self.mtp.shared_head.norm(out)), out

    def verify_mtp(self, root, draft, cache):
        """Two-row compatibility API; return the state after consuming root."""
        if root.shape != (1, 1) or draft.shape != (1, 1):
            raise ValueError("GLM MTP verification requires singleton token rows")
        logits, hidden, snapshots = self.verify_mtp_rows(
            mx.concatenate([root, draft], axis=1), cache
        )
        return logits, hidden, snapshots[0]

    def prefill_hidden(
        self, inputs, cache, *, inputs_embeds=None, capture_layer_ids=()
    ):
        return self.model(
            inputs,
            cache,
            prefill=True,
            inputs_embeds=inputs_embeds,
            capture_layer_ids=capture_layer_ids,
        )

    def configure_precision(
        self, *, kv_cache_dtype="bf16", prefill_activation_dtype="bf16"
    ):
        """Choose model-owned precision before caches or forward graphs exist."""
        if kv_cache_dtype not in (
            "auto",
            "bf16",
            "bfloat16",
            "fp8",
            "mxfp8",
            "fp8_e4m3",
        ):
            raise ValueError("GLM MLA cache dtype must be BF16 or FP8")
        if prefill_activation_dtype not in ("bf16", "fp8"):
            raise ValueError("GLM prefill activation dtype must be BF16 or FP8")
        self._kv_cache_dtype = (
            "fp8" if kv_cache_dtype in ("fp8", "mxfp8", "fp8_e4m3") else "bf16"
        )
        self.model._prefill_activation_dtype = prefill_activation_dtype

    def configure_mtp_verification(self, mode):
        if mode not in ("serial", "head", "ffn"):
            raise ValueError(f"Unknown GLM MTP verification mode: {mode}")
        self._mtp_verification_mode = mode

    def configure_fixed_groups(self, enabled):
        """Compile phase-sized pure layer groups for submission A/B."""
        self.model._compile_groups = bool(enabled)
        self.model._fixed_groups.clear()
        for layer in self.layers:
            if not layer.is_linear:
                layer._fixed_mla_attention = (
                    _FixedMLAAttention(layer.self_attn) if enabled else None
                )
                layer._fixed_mla_layer = _FixedMLALayer(layer) if enabled else None

    def configure_shared_stream(self, stream):
        """Bind a caller-owned MLX stream on the current serving thread."""
        self.model._shared_stream = stream

    def configure_eval_groups(self, *, decode=4, prefill=4, verification=4):
        """Bound submission graphs by phase, before serving any requests.

        Decode, prefill and MTP verification each default to four layers.
        Larger singleton groups can amortize submissions across KDA/MLA periods.
        Explicit arguments allow same-model submission A/B without global state.
        """
        if (
            type(decode) is not int
            or type(prefill) is not int
            or type(verification) is not int
            or min(decode, prefill, verification) < 1
        ):
            raise ValueError("MLX layer submission groups must be positive integers")
        self.model._decode_eval_group_size = decode
        self.model._eval_group_size = prefill
        self.model._verification_eval_group_size = verification
        self.model._fixed_groups.clear()

    def configure_latent_projection_cache(self, enabled):
        for layer in self.model.layers:
            if not layer.is_linear:
                layer.self_attn.cache_latent_projection = enabled
                layer.self_attn._latent_projection = None

    def verify_mtp_requests(
        self, inputs, caches, *, token_only=False, capture_layer_ids=()
    ):
        """Layerwise ragged MTP trunks with private per-depth cache owners."""
        if not inputs or len(inputs) != len(caches):
            raise ValueError("Each verification chain requires its own cache")
        if any(
            v.ndim != 2 or v.shape[0] != 1 or not 1 <= v.shape[1] <= 17 for v in inputs
        ):
            raise ValueError("Verification chains must contain 1-17 token rows")
        if any(len(c) != len(self.layers) for c in caches):
            raise ValueError("Each chain must own all target layer caches")
        if len(inputs) == 1:
            return [
                self.verify_mtp_rows(
                    inputs[0],
                    caches[0],
                    token_only=token_only,
                    capture_layer_ids=capture_layer_ids,
                )
            ]
        rows = [self.model.embed_tokens(v) for v in inputs]
        if self.args.mhc:
            rows = [mx.repeat(v[..., None, :], self.args.hc_mult, -2) for v in rows]
        snapshots = [[[] for _ in range(v.shape[1])] for v in inputs]
        captures = [{} for _ in inputs]
        start = 0
        for i, layer in enumerate(self.layers):
            rows, states = layer.verify_requests(
                rows, [c[i] for c in caches], shared_stream=self.model._shared_stream
            )
            if i in capture_layer_ids:
                for capture, value in zip(captures, rows):
                    capture[i] = value.mean(-2) if self.args.mhc else value
            for owner, chain in zip(snapshots, states):
                for destination, state in zip(owner, chain):
                    destination.append(state)
            if (i + 1) % self.model._verification_eval_group_size == 0 or i + 1 == len(
                self.layers
            ):
                mx.async_eval(
                    rows, [[s.cache for s in c[start : i + 1]] for c in caches]
                )
                start = i + 1
        hidden = [self.model.norm(v.mean(-2) if self.args.mhc else v) for v in rows]
        logits = (
            self.target_tokens(mx.concatenate(hidden, 1))
            if token_only
            else _project_rows(self.lm_head, mx.concatenate(hidden, 1))
        )
        result, cursor = [], 0
        for value, states, capture in zip(hidden, snapshots, captures):
            count = value.shape[1]
            entry = (logits[:, cursor : cursor + count], value, states)
            if capture_layer_ids:
                entry += (mx.concatenate([capture[i] for i in capture_layer_ids], -1),)
            result.append(entry)
            cursor += count
        return result

    def verify_mtp_rows(self, inputs, cache, *, token_only=False, capture_layer_ids=()):
        """Verify a draft chain with per-position attention state.

        Fused recurrence emits every row boundary; MLA batches explicit projected
        inputs with legal latent indices. Serving batches stateless
        FFN work with singleton QMV reductions on covered affine tiles; mHC mix
        and residual GEMVs retain their original reduction order. Immutable row
        snapshots publish exactly the consumed prefix without attention replay.
        Serial/head modes are diagnostic controls. Unsupported FFN tiles retain
        the general MLX path and require their own numerical validation.
        """
        if len(cache) != len(self.layers):
            raise ValueError("MTP verification requires one cache per target layer")
        if inputs.ndim != 2 or inputs.shape[0] != 1 or not 1 <= inputs.shape[1] <= 17:
            raise ValueError("GLM MTP verification requires one chain of 1-17 rows")
        captures = {}
        count = inputs.shape[1]
        mode = getattr(self, "_mtp_verification_mode", "serial")
        x = self.model.embed_tokens(inputs)
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        snapshots = [[] for _ in range(count)]
        group_start = 0
        i = 0
        while i < len(self.layers):
            layer, state = self.layers[i], cache[i]
            if (
                self.model._compile_groups
                and self.model._shared_stream is None
                and mode == "ffn"
                and count > 1
                and layer.is_linear
            ):
                end = min(
                    ((i // self.model._verification_eval_group_size) + 1)
                    * self.model._verification_eval_group_size,
                    len(self.layers),
                )
                stop = i + 1
                while (
                    stop < end
                    and self.layers[stop].is_linear
                    and stop - 1 not in capture_layer_ids
                ):
                    stop += 1
                key = ("verify", i, stop)
                group = self.model._fixed_groups.get(key)
                if group is None:
                    group = _FixedLayerGroup(self.layers[i:stop])
                    self.model._fixed_groups[key] = group
                x, states = group(
                    x, cache[i:stop], verification=True, layers=self.layers[i:stop]
                )
                for owner, values in zip(snapshots, states):
                    owner.extend(values)
                i = stop - 1
            elif mode == "ffn" and count > 1:
                x, layer_snapshots = layer.verify_ffn_rows(
                    x, state, shared_stream=self.model._shared_stream
                )
                for j, snapshot in enumerate(layer_snapshots):
                    snapshots[j].append(snapshot)
            else:
                rows = []
                for j in range(count):
                    rows.append(
                        layer(
                            x[:, j : j + 1],
                            state,
                            shared_stream=self.model._shared_stream,
                        )
                    )
                    snapshot = type(state)(4)
                    snapshot.cache = [
                        v.view(v.dtype) if v is not None else None for v in state.cache
                    ]
                    snapshot.offset = state.offset
                    snapshots[j].append(snapshot)
                x = mx.concatenate(rows, 1)
            if i in capture_layer_ids:
                captures[i] = x.mean(-2) if self.args.mhc else x
            if (i + 1) % self.model._verification_eval_group_size == 0 or i + 1 == len(
                cache
            ):
                mx.async_eval(x, [c.cache for c in cache[group_start : i + 1]])
                group_start = i + 1
            i += 1
        if self.args.mhc:
            x = x.mean(axis=-2)
        hidden_rows = [self.model.norm(x[:, j : j + 1]) for j in range(count)]
        hidden = mx.concatenate(hidden_rows, axis=1)
        if token_only:
            entry = (self.target_tokens(hidden), hidden, snapshots)
            if capture_layer_ids:
                entry += (mx.concatenate([captures[i] for i in capture_layer_ids], -1),)
            return entry
        logits = None
        if (
            mode != "serial"
            and count > 1
            and isinstance(self.lm_head, nn.QuantizedLinear)
        ):
            p = self.lm_head
            if (
                p.mode == "affine"
                and "bias" not in p
                and getattr(p, "global_scale", None) is None
            ):
                logits = mlx_affine_rows(
                    hidden,
                    p.weight,
                    p.scales,
                    p.biases,
                    bits=p.bits,
                    group_size=p.group_size,
                )
        if (
            logits is None
            and mode != "serial"
            and count > 1
            and type(self.lm_head) is nn.Linear
            and "bias" not in self.lm_head
        ):
            logits = mlx_dense_rows(hidden, self.lm_head.weight)
        if logits is None:
            from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

            logits = (
                self.lm_head(hidden)
                if isinstance(self.lm_head, FP8Linear)
                else mx.concatenate([self.lm_head(row) for row in hidden_rows], axis=1)
            )
        entry = (logits, hidden, snapshots)
        if capture_layer_ids:
            entry += (mx.concatenate([captures[i] for i in capture_layer_ids], -1),)
        return entry

    def make_cache(self):
        # KDA stores conv/recurrent state; MLA stores latent KV, raw indexer
        # tail keys/gates and the completed pooled-key buffer.
        return [
            (
                FP8MLACache(4)
                if getattr(self, "_kv_cache_dtype", "bf16") == "fp8"
                else MLACache(4)
            )
            if not layer.is_linear and self.args.kv_lora_rank >= 256
            else NativeCache(4)
            for layer in self.layers
        ]

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache or self.make_cache()))

    def forward_last_logits(self, inputs, cache=None):
        hidden = self.model(inputs, cache or self.make_cache())
        return self.lm_head(hidden[:, -1:, :])

    def prepare_for_inference(self):
        # Run after checkpoint loading and optional quantization, before serving.
        count = sum(
            layer.self_attn.prepare_for_inference()
            for layer in self.model.layers
            if layer.is_linear
        )
        logger.info("Fused GLM5-Next KDA input projections in %d layers", count)
        for layer in self.model.layers:
            if isinstance(layer.mlp, MoE):
                if getattr(self, "_optimization_profile", "router") != "base":
                    layer.mlp.prepare_for_inference()
                layer.mlp.switch_mlp.prepare_for_inference()
            # Materialize remaining projections per layer rather than submitting
            # the full checkpoint in one command buffer at runner startup.
            mx.eval(layer.parameters())
            _prepare_e_modules(layer)
            if isinstance(layer.mlp, MoE):
                switch = layer.mlp.switch_mlp
                if (
                    switch.gate_proj.weight.ndim == 3
                    and switch.gate_proj.weight.dtype == mx.uint8
                ):
                    from sglang.srt.hardware_backend.mlx.warmup import (
                        warmup_mlx_switch_glu,
                    )

                    warmup_mlx_switch_glu(switch, picks=self.args.num_experts_per_tok)
        mx.eval(self.model.embed_tokens.parameters(), self.model.norm.parameters())
        mx.eval(self.lm_head.parameters())
        if self.mtp is not None:
            mx.eval(self.mtp.parameters())
        if self.mtp is not None and type(self.lm_head) is nn.Linear:
            # Compile the greedy epilogue during preparation. Its first Metal
            # build belongs to model startup, not the first speculative cycle.
            token = mlx_dense_argmax(
                mx.zeros((1, 1, self.args.hidden_size), self.lm_head.weight.dtype),
                self.lm_head.weight,
            )
            if token is not None:
                mx.eval(token)

        self.warmup_for_inference()

    def warmup_for_inference(
        self, *, verification_counts=(), capture_layer_ids=(), token_only=False
    ):
        """Warm the loaded NVFP4 model's real graph owners before serving."""
        if not any(
            isinstance(layer.mlp, MoE)
            and layer.mlp.switch_mlp.gate_proj.weight.dtype == mx.uint8
            for layer in self.layers
        ):
            return False
        from sglang.srt.hardware_backend.mlx.warmup import warmup_mlx_model

        warmup_mlx_model(
            self,
            verification_counts=verification_counts,
            capture_layer_ids=capture_layer_ids,
            token_only=token_only,
        )
        return True

    def configure_optimizations(self, profile):
        """Select independent decode candidates before load-time preparation."""
        if profile not in (
            "base",
            "router",
            "pointwise",
            "compile_ffn",
            "gate_up",
            "all",
            "affine_fp16",
            "affine_all",
            "affine_gate_up",
            "affine_hc",
            "affine_kda",
            "affine_moe",
            "affine_fused",
        ):
            raise ValueError(f"Unknown GLM MLX optimization profile: {profile}")
        self._optimization_profile = profile
        self.model._fixed_groups.clear()
        affine = profile.startswith("affine_")
        fused = profile in ("affine_hc", "affine_kda", "affine_moe", "affine_fused")
        pointwise = profile in ("pointwise", "all", "affine_all") or fused
        for layer in self.model.layers:
            layer.fuse_hc = profile in ("affine_hc", "affine_fused")
            layer.compile_residual = pointwise
            # Real checkpoints also quantize MLA projections. FFN-only matching
            # lets the first MLA promote every later residual back to FP32.
            # Affine profiles keep BF16 residuals; base/router retain FP32 promotion.
            layer.preserve_residual_dtype = affine
            if layer.is_linear:
                layer.self_attn.compile_gate = pointwise
                layer.self_attn.fuse_decode = profile in ("affine_kda", "affine_fused")
            if isinstance(layer.mlp, MoE):
                layer.mlp.fuse_decode = profile in ("affine_moe", "affine_fused")
            mlps = (
                (layer.mlp.switch_mlp, layer.mlp.shared_experts)
                if isinstance(layer.mlp, MoE)
                else (layer.mlp,)
            )
            if isinstance(layer.mlp, MoE) and profile == "base":
                layer.mlp._router_weight = None
                layer.mlp._router_sources = ()
            for mlp in mlps:
                mlp.match_affine_dtype = affine
                mlp.compile_decode = (
                    profile in ("compile_ffn", "all", "affine_all") or fused
                )
                if isinstance(mlp, MLP):
                    mlp.compile_activation = pointwise
                else:
                    mlp.activation.compile_activation = pointwise
                    mlp.fuse_gate_up = profile in (
                        "gate_up",
                        "affine_gate_up",
                        "affine_fused",
                    )
                    if not mlp.fuse_gate_up:
                        mlp._fused_gate_up = None
                        mlp._gate_up_sources = ()

    def sanitize(self, weights):
        result = {}
        # Consume the input dictionary: mlx-lm retains it while sanitize runs.
        # Copying its values would keep every individual expert alive alongside
        # the stacked experts, doubling the ~190 GiB checkpoint at peak.
        for original_key in list(weights):
            key = original_key
            value = weights.pop(original_key)
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "model.", 1)
            ignored_prefixes = ("model.visual.", "visual.", "model.mtp.", "mtp.")
            if key.startswith(ignored_prefixes):
                continue
            if self.mtp is None and key.startswith(
                f"model.layers.{self.args.num_hidden_layers}."
            ):
                # The checkpoint's extra decoder is the MTP prediction head.
                continue
            if "_conv1d.weight" in key:
                key = key.replace("_conv1d.weight", "_conv.conv.weight")
                # Checkpoint Conv1d is [channels, 1, kernel]; MLX is [channels, kernel, 1].
                value = value.swapaxes(1, 2)
            if key.endswith(".A_log"):
                value = value.reshape(-1)
            result[key] = value
        for i, layer in enumerate(self.model.layers):
            if not isinstance(layer.mlp, MoE):
                continue
            p = f"model.layers.{i}.mlp"
            bias = f"{p}.gate.e_score_correction_bias"
            if bias in result:
                result[f"{p}.e_score_correction_bias"] = result.pop(bias)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "scales", "biases"):
                    if f"{p}.experts.0.{proj}.{suffix}" not in result:
                        continue
                    stacked = mx.stack(
                        [
                            result.pop(f"{p}.experts.{e}.{proj}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                    )
                    # Materialize one stack at a time: keeping both source
                    # shards and all stacked experts doubles resident weights.
                    mx.eval(stacked)
                    result[f"{p}.switch_mlp.{proj}.{suffix}"] = stacked
            logger.info(
                "Prepared GLM5-Next MLX experts for layer %d/%d",
                i + 1,
                len(self.model.layers),
            )
        return result


def _e_layout_key(modules):
    pending = list(modules) if isinstance(modules, (list, tuple)) else [modules]
    result = []
    for value in pending:
        if isinstance(value, MLA):
            result.append(getattr(value, "_e_projection_strides", None))
        elif hasattr(value, "self_attn"):
            pending.append(value.self_attn)
        elif hasattr(value, "layers"):
            pending.extend(value.layers)
    return tuple(result)


def _e_affine_projection(p):
    return (
        type(p) is nn.QuantizedLinear
        and p.mode == "affine"
        and "bias" not in p
        and getattr(p, "global_scale", None) is None
    )


def _e_parts(projections):
    return [(p.weight, p.scales, p.biases, p.bits, p.group_size) for p in projections]


def _prepare_e_modules(modules):
    # Live containers only, called before graph observation/compiled entry.
    pending = list(modules) if isinstance(modules, (list, tuple)) else [modules]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, MLA):
            value.prepare_e_projections()
        elif hasattr(value, "self_attn"):
            pending.append(value.self_attn)
        elif hasattr(value, "layers"):
            pending.extend(value.layers)
