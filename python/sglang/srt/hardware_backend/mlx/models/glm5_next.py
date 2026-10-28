# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text inference with native MLX KDA and indexed MLA caches.

Equations follow transformers.models.glm5_next.modeling_glm5_next. Unlike
the ordinary MLX attention adapter, MLA keeps compressed KV and indexer state;
these caches must stay together for the lifetime of a request.
"""

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


class ClampedSwitchGLU(SwitchGLU):
    def __init__(self, args):
        super().__init__(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.limit = args.swiglu_limit

    def __call__(self, x, indices):
        # SwitchLinear broadcasts the selected-expert axis over the inputs.
        x = mx.expand_dims(x, (-2, -3))
        gate = self.gate_proj(x, indices)
        up = self.up_proj(x, indices)
        return self.down_proj(clamped_swiglu(gate, up, self.limit), indices).squeeze(-2)


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

    def __call__(self, x, cache):
        b, t, _ = x.shape
        projections = []
        for i, name in enumerate(("q", "k", "v")):
            value, state = getattr(self, name + "_conv")(
                getattr(self, name + "_proj")(x), cache[i], None, None
            )
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
        a = self.args
        b, t, _ = x.shape
        keys = self.k_norm(self.wk(x))
        gates = x @ self.index_kpool_compress_gate.T
        if cache[1] is not None:
            keys = mx.concatenate([cache[1], keys], axis=1)
            gates = mx.concatenate([cache[2], gates], axis=1)
        cache[1], cache[2] = keys, gates
        length, pool = keys.shape[1], a.index_kpool
        visible = mx.arange(length)[None, :] <= (offset + mx.arange(t))[:, None]
        # Below the sparse budget every complete pool and its causal tail
        # are selected. Still retain indexer state for future sparse steps.
        if length <= a.index_topk:
            return mx.broadcast_to(visible, (b, t, length))
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
        selected = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
        pool_mask = mx.zeros((b, t, n + 1), mx.int32)
        pool_mask = mx.put_along_axis(
            pool_mask, selected, mx.array(1, mx.int32), axis=-1
        )
        raw_pool = mx.arange(length) // pool
        mask = mx.take(pool_mask, raw_pool, axis=-1).astype(mx.bool_)
        if a.index_kpool_always_select_tail:
            tail_start = ((offset + mx.arange(t) + 1) // pool) * pool
            mask |= mx.arange(length)[None, None, :] >= tail_start[None, :, None]
        return mask & visible[None]


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
        mask = self.indexer(x, qr, cache, cache.offset)
        cache.offset += t
        # Keep only the compressed latent in cache. Expansion is transient.
        expanded = (
            self.kv_b_proj(kv)
            .reshape(b, -1, a.num_attention_heads, a.qk_nope_head_dim + a.v_head_dim)
            .transpose(0, 2, 1, 3)
        )
        k, v = mx.split(expanded, [a.qk_nope_head_dim], axis=-1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=a.qk_nope_head_dim**-0.5, mask=mask[:, None]
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))


def hyper_connection(x, fn, base, scale, args):
    dtype = x.dtype
    flat = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    weights = mx.fast.rms_norm(flat, None, args.rms_norm_eps) @ fn.astype(mx.float32).T
    h = args.hc_mult
    pre, post, comb = mx.split(weights, [h, 2 * h], axis=-1)
    pb, qb, cb = mx.split(base.astype(mx.float32), [h, 2 * h])
    pre = mx.sigmoid(pre * scale[0] + pb) + args.hc_eps
    post = 2 * mx.sigmoid(post * scale[1] + qb)
    comb = (
        mx.softmax((comb * scale[2] + cb).reshape(*x.shape[:2], h, h), axis=-1)
        + args.hc_eps
    )
    comb /= comb.sum(-2, keepdims=True) + args.hc_eps
    for _ in range(args.hc_sinkhorn_iters - 1):
        comb /= comb.sum(-1, keepdims=True) + args.hc_eps
        comb /= comb.sum(-2, keepdims=True) + args.hc_eps
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
        for layer, state in zip(self.layers, cache):
            x = layer(x, cache=state)
            # Bound the lazy graph and release layer-local transient expanded KV.
            mx.eval(x, *[s for s in state.state if s is not None])
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
