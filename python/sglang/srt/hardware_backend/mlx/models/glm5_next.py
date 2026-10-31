# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text inference with native MLX KDA and indexed MLA caches.

Equations follow transformers.models.glm5_next.modeling_glm5_next. Unlike
the ordinary MLX attention adapter, MLA keeps compressed KV and indexer state;
these caches must stay together for the lifetime of a request.
"""

import copy
import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.models.kimi_linear import ShortConv1d, _group_expert_select
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from sglang.kernels.ops.attention.linear.mlx_kda import mlx_kda_decode
from sglang.kernels.ops.attention.mlx_ragged import mlx_ragged_latent_attention
from sglang.kernels.ops.gemm.mlx_affine_rows import (
    mlx_affine_mixed_rows,
    mlx_affine_routed_swiglu,
    mlx_affine_rows,
    mlx_affine_swiglu_rows,
)
from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows, mlx_headed_dense_rows
from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm
from sglang.kernels.ops.moe.mlx_glm import (
    mlx_glm_down_combine,
    mlx_glm_router,
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


def _row_exact_affine_ffn(block, x, indices=None, *, gate_up_only=False):
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
        restore_dtype = self.match_affine_dtype and _uses_fp16_affine_ffn(
            x, (self.gate_proj, self.up_proj, self.down_proj)
        )
        dtype = x.dtype
        if restore_dtype:
            x = x.astype(mx.float16)
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

    def prepare_for_inference(self):
        if not self.fuse_gate_up:
            return False
        if self._fused_gate_up is not None and all(
            old is new
            for old, new in zip(self._gate_up_sources, self._current_gate_up_sources())
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
        mx.eval([fused[f] for f in fields])
        for field in fields:
            gate[field], up[field] = mx.split(fused[field], 2, axis=1)
        self._fused_gate_up = fused
        self._gate_up_sources = self._current_gate_up_sources()
        return True

    def __call__(self, x, indices):
        if self._fused_gate_up is not None and not all(
            old is new
            for old, new in zip(self._gate_up_sources, self._current_gate_up_sources())
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

    def __call__(self, x, *, batch_fusion=False):
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
            scores = x.astype(mx.float32) @ weight.astype(mx.float32).T
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
                    # On M5 the fused multi-row epilogue helps verification,
                    # while native compiled singleton QMV remains faster in
                    # serving. Keep that measured one-row dispatch choice.
                    act = (
                        _row_exact_affine_ffn(sw, local, indices, gate_up_only=True)
                        if batch_fusion and x.shape[1] > 1
                        else None
                    )
                    if act is None:
                        act = _compiled_quantized_ffn(
                            sw, local, indices, gate_up_only=True
                        )
                    shared = self.shared_experts(x, batch_fusion=batch_fusion)
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
        routed = self.switch_mlp(x, indices)
        routed = (routed * weights[..., None]).sum(-2).astype(x.dtype)
        return routed + self.shared_experts(x, batch_fusion=batch_fusion)


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

    def _fused_qkv_is_current(self):
        return self._fused_qkv is not None and all(
            old is new for old, new in zip(self._fused_qkv_sources, self._qkv_sources())
        )

    def prepare_for_inference(self):
        """Fuse compatible KDA inputs once, including the gate projections.

        Canonical checkpoint parameters become views of the fused storage;
        saving a model still uses its original names and mixed-bit metadata.
        No second resident copy of all KDA projection weights is retained.
        """
        if self._fused_qkv_is_current():
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
        return True

    def project_verify_rows(self, x):
        """Share dense input weights across rows, keeping recurrence transactional."""
        if (
            not self.fuse_decode
            or self.head_dim != 128
            or len(self._fused_input_names) != 6
            or not self._fused_qkv_is_current()
            or type(self._fused_qkv) is not nn.Linear
        ):
            return None
        proj = mlx_dense_rows(x, self._fused_qkv.weight)
        if proj is None:
            return None
        _, _, _, fa_end, ga_end = self._fused_input_splits
        a_input = proj[..., 3 * self.dim : fa_end]
        gate_input = proj[..., fa_end:ga_end]
        a = mx.concatenate(
            [self.f_b_proj(a_input[:, j : j + 1]) for j in range(x.shape[1])], 1
        )
        gate = mx.concatenate(
            [self.g_b_proj(gate_input[:, j : j + 1]) for j in range(x.shape[1])], 1
        )
        return proj, a, gate

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
                a = self.f_b_proj(proj[..., 3 * self.dim : fa_end])
                gate = self.g_b_proj(proj[..., fa_end:ga_end])
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
        n = length // pool
        q = (self.wq_b(qr) if projected is None else projected[1]).reshape(
            b,
            t,
            a.index_n_heads,
            a.index_head_dim,
        )
        scores = mx.maximum(
            q.astype(mx.float32) @ pooled.astype(mx.float32).swapaxes(-1, -2)[:, None],
            0,
        )
        scores *= a.index_head_dim**-0.5
        weights = (self.weights_proj(x) if projected is None else projected[2]).astype(
            mx.float32
        ) * a.index_n_heads**-0.5
        scores = (scores * weights[..., None]).sum(axis=2)
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


class MLA(nn.Module):
    cache_latent_projection = False

    def _latent_matrices(self, activation_dtype):
        """Retain the operands at their original matmul promotion boundary.

        Dense BF16 KV-B weights also promote for FP32 queries. Pack/cast
        each sliced operand separately, matching the uncached operation's
        layout rather than changing it by casting the interleaved parent.
        """
        projection = self.kv_b_proj
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

    def project_verify_rows(self, x):
        qr = self.q_a_layernorm(_project_rows(self.q_a_proj, x))
        q = _project_rows(self.q_b_proj, qr)
        kv = self.kv_a_layernorm(_project_rows(self.kv_a_proj_with_mqa, x))
        return qr, q, kv, self.indexer.project_verify_rows(x, qr)

    def verify_rows(self, x, cache, *, projected=None):
        """Batch projections and latent attention with explicit per-row visibility."""
        a = self.args
        b, count, _ = x.shape
        qr, q, new_kv, index_projection = (
            self.project_verify_rows(x) if projected is None else projected
        )
        offset = cache.offset
        kv = mx.concatenate([cache[0], new_kv], 1) if cache[0] is not None else new_kv
        snapshots, selected = [], []
        for j in range(count):
            length = offset + j + 1
            indices = self.indexer(
                x[:, j : j + 1],
                qr[:, j : j + 1],
                cache,
                offset + j,
                projected=tuple(v[:, j : j + 1] for v in index_projection),
            )
            if indices is None:
                indices = mx.arange(length)[None, None, :]
            else:
                width = (
                    min(a.index_topk // a.index_kpool, length // a.index_kpool)
                    * a.index_kpool
                )
                if a.index_kpool_always_select_tail:
                    width += length % a.index_kpool
                indices = indices[..., :width]
            selected.append(indices)
            snapshot = NativeCache(4)
            snapshot.cache = [kv[:, :length]] + [
                v.view(v.dtype) if v is not None else None for v in cache.cache[1:]
            ]
            snapshot.offset = length
            snapshots.append(snapshot)
        cache.cache = [
            v.view(v.dtype) if v is not None else None for v in snapshots[-1].cache
        ]
        cache.offset = offset + count
        width = max(v.shape[-1] for v in selected)
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
            indices = mx.concatenate(
                [
                    mx.pad(
                        v,
                        [(0, 0), (0, 0), (0, width - v.shape[-1])],
                        constant_values=-1,
                    )
                    for v in selected
                ],
                axis=1,
            )
            latent_out = mlx_ragged_latent_attention(
                latent_q,
                kv,
                indices.astype(mx.int32),
                [v.shape[-1] for v in selected],
                scale=a.qk_nope_head_dim**-0.5,
            )
            if latent_out is None:
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
        return _project_rows(self.o_proj, out), snapshots

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
        if cache[0] is not None:
            kv = mx.concatenate([cache[0], kv], axis=1)
        cache[0] = kv
        indices = self.indexer(x, qr, cache, cache.offset)
        cache.offset += t
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
            width = min(a.index_topk // pool, kv.shape[1] // pool) * pool
            if a.index_kpool_always_select_tail:
                width += kv.shape[1] % pool
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

    def _branch_inputs(self, x, name, norm):
        if not self.args.mhc:
            return norm(x), None, x
        if (
            self.fuse_hc
            and x.shape[0] == 1
            and 1 <= x.shape[1] <= 17
            and x.shape[2] == 4
        ):
            flat = x.reshape(1, x.shape[1], -1).astype(mx.float32)
            weight = getattr(self, f"hc_{name}_fn").astype(mx.float32).T
            # Keep the trained FP32 mix's singleton GEMV rounding. Batch
            # the nonlinear HC work, not this cancellation-sensitive sum.
            mix = mx.concatenate(
                [
                    mx.fast.rms_norm(flat[:, j : j + 1], None, self.args.rms_norm_eps)
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
        collapsed, post, residual = hyper_connection(
            x,
            getattr(self, f"hc_{name}_fn"),
            getattr(self, f"hc_{name}_base"),
            getattr(self, f"hc_{name}_scale"),
            self.args,
        )
        return norm(collapsed), post, residual

    def _branch_output(self, branch, post, residual):
        if self.preserve_residual_dtype:
            branch = branch.astype(residual.dtype)
        if post is None:
            return residual + branch
        if self.compile_residual or self.fuse_hc:
            return _compiled_residual(residual, post, branch)
        return residual + post[..., None] * branch[..., None, :]

    def __call__(self, x, cache):
        for name, norm, block in (
            ("attn", self.input_layernorm, lambda y: self.self_attn(y, cache)),
            ("ffn", self.post_attention_layernorm, self.mlp),
        ):
            if self.args.mhc:
                if self.fuse_hc and x.shape[:3] == (1, 1, 4):
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

    def verify_ffn_rows(self, x, cache):
        """Batch stateless FFN dispatch, retaining per-position attention state."""
        attended, snapshots = [], []
        attention_inputs = [
            self._branch_inputs(x[:, j : j + 1], "attn", self.input_layernorm)
            for j in range(x.shape[1])
        ]
        normed_rows = mx.concatenate([v[0] for v in attention_inputs], 1)
        projected = (
            self.self_attn.project_verify_rows(normed_rows) if self.is_linear else None
        )
        batched = (
            None if self.is_linear else self.self_attn.verify_rows(normed_rows, cache)
        )
        if batched is not None:
            branches, snapshots = batched
            for j, (_, post, residual) in enumerate(attention_inputs):
                attended.append(
                    self._branch_output(branches[:, j : j + 1], post, residual)
                )
        else:
            for j, (normed, post, residual) in enumerate(attention_inputs):
                branch = (
                    self.self_attn(
                        normed,
                        cache,
                        projected=tuple(v[:, j : j + 1] for v in projected),
                    )
                    if projected is not None
                    else self.self_attn(normed, cache)
                )
                attended.append(self._branch_output(branch, post, residual))
                snapshot = NativeCache(4)
                snapshot.cache = [
                    v.view(v.dtype) if v is not None else None for v in cache.cache
                ]
                snapshot.offset = cache.offset
                snapshots.append(snapshot)
        normed, post, residual = self._branch_inputs(
            mx.concatenate(attended, axis=1), "ffn", self.post_attention_layernorm
        )
        branch = self.mlp(normed, batch_fusion=True)
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

    def __call__(self, hidden, embeddings, cache):
        if hidden.shape != embeddings.shape or hidden.ndim != 3:
            raise ValueError("MTP hidden states must match following-token embeddings")
        x = self.eh_proj(
            mx.concatenate([self.enorm(embeddings), self.hnorm(hidden)], -1)
        )
        return super().__call__(x, cache)


class TextModel(nn.Module):
    # Submit one KDA/MLA period at a time. Async evaluation bounds the lazy
    # graph without making the host wait after each individual decoder layer.
    _eval_group_size = 4

    def __init__(self, args, enable_mtp=False):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        if enable_mtp:
            self.layers.append(MTPHead(args))
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache):
        x = self.embed_tokens(inputs)
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        group_start = 0
        for i, (layer, state) in enumerate(
            zip(self.layers[: self.args.num_hidden_layers], cache)
        ):
            x = layer(x, cache=state)
            final_layer = i + 1 == self.args.num_hidden_layers
            if (i + 1) % self._eval_group_size == 0 or final_layer:
                mx.async_eval(x, [c.cache for c in cache[group_start : i + 1]])
                group_start = i + 1
        if self.args.mhc:
            x = x.mean(axis=-2)
        return self.norm(x)


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

    def forward_with_hidden(self, inputs, cache=None):
        hidden = self.model(inputs, cache if cache is not None else self.make_cache())
        return self.lm_head(hidden), hidden

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

    def configure_mtp_verification(self, mode):
        if mode not in ("serial", "head", "ffn"):
            raise ValueError(f"Unknown GLM MTP verification mode: {mode}")
        self._mtp_verification_mode = mode

    def configure_latent_projection_cache(self, enabled):
        for layer in self.model.layers:
            if not layer.is_linear:
                layer.self_attn.cache_latent_projection = enabled
                layer.self_attn._latent_projection = None

    def verify_mtp_rows(self, inputs, cache):
        """Verify a draft chain with per-position attention state.

        Attention and recurrence retain per-row state. Serving batches stateless
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
        count = inputs.shape[1]
        mode = getattr(self, "_mtp_verification_mode", "serial")
        x = self.model.embed_tokens(inputs)
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        snapshots = [[] for _ in range(count)]
        for i, (layer, state) in enumerate(zip(self.layers, cache)):
            if mode == "ffn" and count > 1:
                x, layer_snapshots = layer.verify_ffn_rows(x, state)
                for j, snapshot in enumerate(layer_snapshots):
                    snapshots[j].append(snapshot)
            else:
                rows = []
                for j in range(count):
                    rows.append(layer(x[:, j : j + 1], state))
                    snapshot = NativeCache(4)
                    snapshot.cache = [
                        v.view(v.dtype) if v is not None else None for v in state.cache
                    ]
                    snapshot.offset = state.offset
                    snapshots[j].append(snapshot)
                x = mx.concatenate(rows, axis=1)
            if (i + 1) % self.model._eval_group_size == 0 or i + 1 == len(cache):
                mx.async_eval(x, [c.cache for c in cache])
        if self.args.mhc:
            x = x.mean(axis=-2)
        hidden_rows = [self.model.norm(x[:, j : j + 1]) for j in range(count)]
        hidden = mx.concatenate(hidden_rows, axis=1)
        logits = None
        if (
            mode != "serial"
            and count > 1
            and isinstance(self.lm_head, nn.QuantizedLinear)
        ):
            p = self.lm_head
            if p.mode == "affine" and "bias" not in p and p.global_scale is None:
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
            logits = mx.concatenate([self.lm_head(row) for row in hidden_rows], axis=1)
        return logits, hidden, snapshots

    def make_cache(self):
        # KDA stores conv/recurrent state; MLA stores latent KV, raw indexer
        # tail keys/gates and the completed pooled-key buffer.
        return [NativeCache(4) for _ in self.layers]

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
                    mlp.fuse_gate_up = profile in ("gate_up", "affine_gate_up")
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
