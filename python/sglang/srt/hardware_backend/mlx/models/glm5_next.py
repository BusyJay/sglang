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
from mlx_lm.models.switch_layers import SwitchGLU

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


def clamped_swiglu(gate, up, limit):
    if limit is not None:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class MLP(nn.Module):
    def __init__(self, args, intermediate_size=None):
        super().__init__()
        dim = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.down_proj = nn.Linear(dim, args.hidden_size, bias=False)
        self.limit = args.swiglu_limit

    def __call__(self, x):
        return self.down_proj(
            clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        return clamped_swiglu(gate, up, self.limit)


class ClampedSwitchGLU(SwitchGLU):
    def __init__(self, args):
        # Keep mlx-lm's sorting/grouped matmul path for larger token batches;
        # the model-specific difference is only the clamped activation.
        super().__init__(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), mx.float32)
        self.switch_mlp = ClampedSwitchGLU(args)
        self.shared_experts = MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )

    def __call__(self, x):
        # The router uses FP32, including the matrix multiplication.
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
        scores = x.astype(mx.float32) @ weight.astype(mx.float32).T
        a = self.args
        indices, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            a.num_experts_per_tok,
            a.n_group,
            a.topk_group,
            a.routed_scaling_factor,
            a.norm_topk_prob,
            a.scoring_func,
        )
        routed = self.switch_mlp(x, indices)
        routed = (routed * weights[..., None]).sum(-2).astype(x.dtype)
        return routed + self.shared_experts(x)


class KDA(nn.Module):
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
        self._fused_qkv_sources = ()

    def _qkv_sources(self):
        return tuple(
            value
            for p in (self.q_proj, self.k_proj, self.v_proj)
            for value in (p, p.weight, p.get("scales"), p.get("biases"))
        )

    def _fused_qkv_is_current(self):
        return self._fused_qkv is not None and all(
            old is new for old, new in zip(self._fused_qkv_sources, self._qkv_sources())
        )

    def prepare_for_inference(self):
        """Fuse compatible Q/K/V once, following Glimmer's load-time fusion.

        Canonical checkpoint parameters become views of the fused storage;
        saving a model still uses its original names and mixed-bit metadata.
        No second resident copy of all KDA projection weights is retained.
        """
        if self._fused_qkv_is_current():
            return True
        self._fused_qkv = None
        self._fused_qkv_sources = ()
        projections = (self.q_proj, self.k_proj, self.v_proj)
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
                p[field].shape != first[field].shape
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
        for field in fields:
            for projection, value in zip(
                projections, mx.split(fused[field], 3, axis=0)
            ):
                projection[field] = value
        self._fused_qkv = fused
        self._fused_qkv_sources = self._qkv_sources()
        return True

    def __call__(self, x, cache):
        b, t, _ = x.shape
        if self._fused_qkv_is_current():
            projected = mx.split(self._fused_qkv(x), 3, axis=-1)
        else:
            if self._fused_qkv is not None:
                self._fused_qkv = None
                self._fused_qkv_sources = ()
            projected = [self.q_proj(x), self.k_proj(x), self.v_proj(x)]
        projections = []
        for i, (name, value) in enumerate(zip(("q", "k", "v"), projected)):
            value, state = getattr(self, name + "_conv")(value, cache[i], None, None)
            cache[i] = state
            projections.append(value.reshape(b, t, self.num_heads, self.head_dim))
        q, k, v = projections
        q, k = normalize_qk(q, k, inv_scale=self.head_dim**-0.5, eps=1e-6)
        a = self.f_b_proj(self.f_a_proj(x)).reshape(b, t, self.num_heads, self.head_dim)
        out, cache[3] = gated_delta_update(
            q,
            k,
            v,
            a,
            self.b_proj(x),
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=cache[3],
            lower_bound=self.lower_bound,
        )
        cache.offset += t
        gate = self.g_b_proj(self.g_a_proj(x)).reshape(out.shape)
        # GLM's gated norm multiplies the FP32 normalized result and gate
        # before casting back to the activation dtype.
        normed = mx.fast.rms_norm(
            out.astype(mx.float32),
            self.o_norm.weight.astype(mx.float32),
            self.o_norm.eps,
        )
        out = (normed * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)
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

    def __call__(self, x, qr, cache, offset):
        """Return selected raw KV indices, or None for dense causal attention.

        Invalid complete pools and padded tail positions carry -1. Keeping
        indices instead of a full-history boolean mask lets attention gather
        only selected latent KV before either QK or PV multiplication.
        """
        a = self.args
        b, t, _ = x.shape
        keys = self.k_norm(self.wk(x))
        gates = x @ self.index_kpool_compress_gate.T
        if cache[1] is not None:
            keys = mx.concatenate([cache[1], keys], axis=1)
            gates = mx.concatenate([cache[2], gates], axis=1)
        cache[1], cache[2] = keys, gates
        length, pool = keys.shape[1], a.index_kpool
        # Below the sparse budget every complete pool and its causal tail
        # are selected. Still retain indexer state for future sparse steps.
        if length <= a.index_topk and a.index_kpool_always_select_tail:
            return None
        n = length // pool
        grouped = keys[:, : n * pool].reshape(b, n, pool, a.index_head_dim)
        logits = gates[:, : n * pool].reshape(grouped.shape).astype(mx.float32)
        probs = mx.softmax(
            logits + self.index_kpool_compress_ape.astype(mx.float32), axis=2
        ).astype(keys.dtype)
        pooled = (probs * grouped).sum(axis=2)
        q = self.wq_b(qr).reshape(b, t, a.index_n_heads, a.index_head_dim)
        scores = mx.maximum(
            q.astype(mx.float32) @ pooled.astype(mx.float32).swapaxes(-1, -2)[:, None],
            0,
        )
        scores *= a.index_head_dim**-0.5
        weights = self.weights_proj(x).astype(mx.float32) * a.index_n_heads**-0.5
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
        weight = self.kv_b_proj.weight
        if isinstance(self.kv_b_proj, nn.QuantizedLinear):
            weight = mx.dequantize(
                weight,
                self.kv_b_proj.scales,
                self.kv_b_proj.biases,
                group_size=self.kv_b_proj.group_size,
                bits=self.kv_b_proj.bits,
                mode=self.kv_b_proj.mode,
            )
        wk, wv = mx.split(
            weight.reshape(a.num_attention_heads, -1, a.kv_lora_rank),
            [a.qk_nope_head_dim],
            axis=1,
        )
        latent_q = q @ wk
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
                    mask=(indices >= 0).reshape(b * t, 1, 1, width),
                )
                .reshape(b, t, a.num_attention_heads, a.kv_lora_rank)
                .transpose(0, 2, 1, 3)
            )
        out = latent_out @ wv.swapaxes(-1, -2)
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

    def __call__(self, x, cache):
        for name, norm, block in (
            ("attn", self.input_layernorm, lambda y: self.self_attn(y, cache)),
            ("ffn", self.post_attention_layernorm, self.mlp),
        ):
            if self.args.mhc:
                collapsed, post, residual = hyper_connection(
                    x,
                    getattr(self, f"hc_{name}_fn"),
                    getattr(self, f"hc_{name}_base"),
                    getattr(self, f"hc_{name}_scale"),
                    self.args,
                )
                x = residual + post[..., None] * block(norm(collapsed))[..., None, :]
            else:
                x = x + block(norm(x))
        return x


class TextModel(nn.Module):
    # Submit one KDA/MLA period at a time. Async evaluation bounds the lazy
    # graph without making the host wait after each individual decoder layer.
    _eval_group_size = 4

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache):
        x = self.embed_tokens(inputs)
        if self.args.mhc:
            x = mx.repeat(x[..., None, :], self.args.hc_mult, axis=-2)
        group_start = 0
        for i, (layer, state) in enumerate(zip(self.layers, cache)):
            x = layer(x, cache=state)
            if (i + 1) % self._eval_group_size == 0 or i + 1 == len(self.layers):
                mx.async_eval(x, [c.cache for c in cache[group_start : i + 1]])
                group_start = i + 1
        if self.args.mhc:
            x = x.mean(axis=-2)
        return self.norm(x)


class Model(nn.Module):
    # Consumed by SGLang's runner: no ordinary q/k/v attention cache adapter.
    native_cache = True

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [NativeCache(4 if layer.is_linear else 3) for layer in self.layers]

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache or self.make_cache()))

    def forward_last_logits(self, inputs, cache=None):
        hidden = self.model(inputs, cache or self.make_cache())
        return self.lm_head(hidden[:, -1:, :])

    def prepare_for_inference(self):
        # Run after checkpoint loading and optional quantization, before serving.
        count = sum(
            layer.self_attn.prepare_for_inference()
            for layer in self.layers
            if layer.is_linear
        )
        logger.info("Fused GLM5-Next KDA Q/K/V projections in %d layers", count)

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
            if key.startswith(("model.visual.", "visual.", "model.mtp.", "mtp.")):
                continue
            if key.startswith(f"model.layers.{self.args.num_hidden_layers}."):
                # The checkpoint's extra decoder is the MTP prediction head.
                continue
            if "_conv1d.weight" in key:
                key = key.replace("_conv1d.weight", "_conv.conv.weight")
                # Checkpoint Conv1d is [channels, 1, kernel]; MLX is [channels, kernel, 1].
                value = value.swapaxes(1, 2)
            if key.endswith(".A_log"):
                value = value.reshape(-1)
            result[key] = value
        for i, layer in enumerate(self.layers):
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
                len(self.layers),
            )
        return result
