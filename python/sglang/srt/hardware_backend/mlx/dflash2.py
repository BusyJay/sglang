# SPDX-License-Identifier: Apache-2.0
"""BF16 DFlash2 drafting with explicit request-owned context KV.

Captures are concatenated completed target-layer outputs, with mHC streams
averaged and without target final normalization. The caller feeds only committed
positions, in consecutive absolute order, exactly once. The newest anchor has
not yet been forwarded by the target: its position follows the consumed context.
States are plain trees suitable for the serving auxiliary-cache snapshot codec.
No proposal KV is persisted; only target captures enter the context cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from sglang.kernels.ops.sampling.mlx_topk import mlx_topk


def _config_dict(config) -> dict:
    return dict(config) if isinstance(config, dict) else dict(vars(config))


class GroupedDynamicConv(nn.Module):
    """Two causal, block-local kernels projected from one sublayer input."""

    def __init__(self, hidden_size: int, taps: int, group_size: int):
        super().__init__()
        self.taps = taps
        self.group_size = group_size
        self.groups = hidden_size // group_size
        self.base_kernel = mx.concatenate(
            [mx.ones((2, 1, hidden_size)), mx.zeros((2, taps - 1, hidden_size))],
            axis=1,
        )
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * taps * self.groups, bias=False
        )

    def convolve(self, x, delta, side):
        batch, length, hidden = x.shape
        grouped = x.reshape(batch, length, self.groups, self.group_size)
        coefficients = (
            self.base_kernel[side].reshape(
                1, 1, self.taps, self.groups, self.group_size
            )
            + delta[..., None]
        )
        out = coefficients[:, :, 0] * grouped
        for tap in range(1, min(self.taps, length)):
            shifted = mx.concatenate(
                [mx.zeros_like(grouped[:, :tap]), grouped[:, :-tap]], axis=1
            )
            out = out + coefficients[:, :, tap] * shifted
        return out.reshape(batch, length, hidden)

    def prepare(self, x):
        delta = self.kernel_projection(x).reshape(
            *x.shape[:-1], 2, self.taps, self.groups
        )
        return self.convolve(x, delta[:, :, 0], 0), delta[:, :, 1]

    def finish(self, x, delta):
        return self.convolve(x, delta, 1)


class DFlashAttention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        # Keep checkpoint names while batching their projections where possible.
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def project_kv(self, x):
        shape = (*x.shape[:-1], self.kv_heads, self.head_dim)
        return self.k_norm(self.k_proj(x).reshape(shape)), self.v_proj(x).reshape(shape)


class DFlashMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config["hidden_size"]
        eps = config["rms_norm_eps"]
        draft = config["dflash_config"]
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.self_attn = DFlashAttention(
            hidden,
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            eps,
        )
        self.mlp = DFlashMLP(hidden, config["intermediate_size"])
        self.attention_conv = GroupedDynamicConv(
            hidden, draft["conv_kernel_size"], draft["conv_group_size"]
        )
        self.mlp_conv = GroupedDynamicConv(
            hidden, draft["conv_kernel_size"], draft["conv_group_size"]
        )


class CandidateSelector(nn.Module):
    def __init__(self, hidden, vocab, rank, top_k):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = mx.zeros((vocab, rank))
        self.successor_codebook = mx.zeros((vocab, rank))
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)

    def select(self, hidden, logits, anchors):
        unary, ids = mlx_topk(logits, self.top_k)
        projected = self.hidden_projection(hidden)
        successors = self.successor_codebook[ids]
        predecessor = anchors.reshape(-1)
        tokens = []
        for slot in range(hidden.shape[1]):
            # Preserve BF16 multiplication before the dot-product reduction,
            # as in SGLang's _score_edges, rather than three FP32 products.
            pred = self.predecessor_codebook[predecessor] * projected[:, slot]
            edge = mx.matmul(pred[:, None, :], successors[:, slot].swapaxes(-1, -2))[
                :, 0
            ]
            pick = mx.argmax(unary[:, slot].astype(mx.float32) + edge, axis=-1)
            predecessor = mx.take_along_axis(ids[:, slot], pick[:, None], axis=-1)[:, 0]
            tokens.append(predecessor)
        return mx.stack(tokens, axis=1).astype(mx.int32)


class DFlash2Model(nn.Module):
    """DFlash2 backbone and greedy selector; target embedding/head stay external."""

    def __init__(self, config):
        super().__init__()
        config = _config_dict(config)
        draft = dict(config["dflash_config"])
        config["dflash_config"] = draft
        config.setdefault("rms_norm_eps", 1e-5)
        config.setdefault(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        )
        config.setdefault("num_key_value_heads", config["num_attention_heads"])
        self.config = config
        self.target_layer_ids = tuple(int(i) for i in draft["target_layer_ids"])
        self.block_size = int(draft["block_size"])
        self.window = int(config["sliding_window"])
        self.mask_token_id = int(draft["mask_token_id"])
        self.hidden_size = int(config["hidden_size"])
        self.vocab_size = int(config["vocab_size"])
        self.head_dim = int(config["head_dim"])
        rope = config.get("rope_parameters") or {}
        self.rope_theta = float(rope.get("rope_theta", config.get("rope_theta", 10000)))
        if config.get("is_causal", False) or config.get("attention_bias", False):
            raise ValueError(
                "This DFlash2 implementation requires noncausal, bias-free attention"
            )
        if config.get("hidden_act", "silu") != "silu" or config.get(
            "use_gemma_norm", False
        ):
            raise ValueError(
                "This DFlash2 implementation requires SiLU and ordinary RMSNorm"
            )
        if rope.get("rope_type", "default") != "default" or config.get("rope_scaling"):
            raise ValueError(
                "This DFlash2 checkpoint must use default, full-width NeoX RoPE"
            )
        if config.get("partial_rotary_factor", 1.0) != 1.0 or not config.get(
            "rope_is_neox_style", True
        ):
            raise ValueError("DFlash2 requires full-width NeoX RoPE")
        if self.block_size <= 1 or self.window <= 1:
            raise ValueError(
                "DFlash2 requires a root/proposal block and a context window"
            )
        if self.hidden_size % draft["conv_group_size"] or draft["conv_kernel_size"] < 1:
            raise ValueError("Invalid DFlash2 grouped convolution geometry")
        if config["num_attention_heads"] % config["num_key_value_heads"]:
            raise ValueError("Query heads must be divisible by KV heads")
        if (
            not 0 < draft["selector_top_k"] <= self.vocab_size
            or draft["selector_rank"] <= 0
        ):
            raise ValueError("Invalid DFlash2 selector geometry")
        if not self.target_layer_ids or len(set(self.target_layer_ids)) != len(
            self.target_layer_ids
        ):
            raise ValueError("DFlash2 capture IDs must be nonempty and unique")
        if any(
            i < 0 or i >= config["num_target_layers"] for i in self.target_layer_ids
        ):
            raise ValueError("DFlash2 capture ID outside target depth")
        types = config.get(
            "layer_types", ["sliding_attention"] * config["num_hidden_layers"]
        )
        if types != ["sliding_attention"] * config["num_hidden_layers"]:
            raise ValueError(
                "This DFlash2 implementation requires sliding_attention layers"
            )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * self.hidden_size, self.hidden_size, bias=False
        )
        self.hidden_norm = nn.RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.norm = nn.RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.layers = [DFlashLayer(config) for _ in range(config["num_hidden_layers"])]
        self.candidate_selector = CandidateSelector(
            self.hidden_size,
            self.vocab_size,
            draft["selector_rank"],
            draft["selector_top_k"],
        )
        self.set_dtype(mx.bfloat16)

    def make_state(self) -> dict:
        """Return a fresh cloneable state, with next_position kept on device."""
        return {
            "keys": (),
            "values": (),
            "positions": mx.array([], dtype=mx.int32),
            "next_position": mx.array(0, dtype=mx.int32),
        }

    def _rope_positions(self, x, positions):
        # Each token is a batch row of length one, allowing explicit positions
        # without reading offsets on the host or assuming contiguous chunks.
        batch, length, heads, dim = x.shape
        rotated = mx.fast.rope(
            x.reshape(batch * length, heads, 1, dim),
            dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=positions.reshape(-1).astype(mx.int32),
        )
        return rotated.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)

    def consume(self, features, positions, state):
        """Append committed captures [1,T,K*H] at consecutive int32 positions.

        Returned arrays never overwrite the supplied tree. Only the newest W-1
        features are projected when a cold prefill exceeds the draft window.
        """
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[-1] != self.fc.weight.shape[-1]
        ):
            raise ValueError("DFlash2 features must have shape [1,T,K*H]")
        if (
            positions.ndim != 1
            or positions.shape[0] != features.shape[1]
            or positions.dtype != mx.int32
        ):
            raise ValueError("DFlash2 positions must be int32 [T]")
        if not positions.size:
            return state
        limit = self.window - 1
        positions = positions[-limit:]
        context = self.hidden_norm(self.fc(features[:, -limit:].astype(mx.bfloat16)))
        old_keys, old_values = state["keys"], state["values"]
        if len(old_keys) not in (0, len(self.layers)) or len(old_keys) != len(
            old_values
        ):
            raise ValueError("Invalid DFlash2 per-layer cache state")
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            k, v = layer.self_attn.project_kv(context)
            k = self._rope_positions(k, positions[None])
            v = v.transpose(0, 2, 1, 3)
            if old_keys:
                k = mx.concatenate([old_keys[i], k], axis=2)
                v = mx.concatenate([old_values[i], v], axis=2)
            keys.append(k[:, :, -limit:])
            values.append(v[:, :, -limit:])
        return {
            "keys": tuple(keys),
            "values": tuple(values),
            "positions": mx.concatenate([state["positions"], positions])[-limit:],
            "next_position": positions[-1] + mx.array(1, dtype=mx.int32),
        }

    def draft(self, anchors, states, embedding, lm_head, num_drafts):
        """Return greedy proposals [B,num_drafts], without modifying any state.

        Every layer retains all trained root/mask KV and query outputs. The last
        FFN computes the requested prefix, with a six-row native matrix floor.
        Context KV receive neither layer normalization nor dynamic convolution.
        """
        if (
            anchors.ndim != 2
            or anchors.shape != (len(states), 1)
            or anchors.dtype != mx.int32
        ):
            raise ValueError("DFlash2 anchors must be int32 [B,1], matching states")
        if not isinstance(num_drafts, int) or not 0 <= num_drafts < self.block_size:
            raise ValueError("num_drafts must be in [0, block_size-1]")
        batch = len(states)
        if not batch or not num_drafts:
            return mx.zeros((batch, num_drafts), dtype=mx.int32)
        if any(len(s["keys"]) != len(self.layers) for s in states):
            raise ValueError("Consume target context before drafting")
        length = max(s["positions"].size for s in states)
        mask = mx.stack(
            [
                mx.concatenate(
                    [
                        mx.zeros((length - s["positions"].size,), dtype=mx.bool_),
                        mx.ones(
                            (s["positions"].size + self.block_size,), dtype=mx.bool_
                        ),
                    ]
                )
                for s in states
            ]
        )[:, None, None, :]
        offsets = mx.stack([s["next_position"] for s in states])
        positions = offsets[:, None] + mx.arange(self.block_size, dtype=mx.int32)[None]
        inputs = mx.concatenate(
            [
                anchors,
                mx.full(
                    (batch, self.block_size - 1), self.mask_token_id, dtype=mx.int32
                ),
            ],
            axis=1,
        )
        h = embedding(inputs).astype(mx.bfloat16)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            x, kernel = layer.attention_conv.prepare(layer.input_layernorm(h))
            # Keep the native eight-row attention. Only the final FFN can
            # discard suffix rows, since its convolution is causal. A six-row
            # floor retains the native matrix reduction used by the block.
            ffn_count = (
                min(self.block_size, max(6, num_drafts + 1))
                if i == len(self.layers) - 1
                else self.block_size
            )
            q = attn.q_norm(
                attn.q_proj(x).reshape(
                    batch, self.block_size, attn.heads, self.head_dim
                )
            )
            q = self._rope_positions(q, positions)
            k, v = attn.project_kv(x)
            k = self._rope_positions(k, positions)
            v = v.transpose(0, 2, 1, 3)
            caches = []
            for field in ("keys", "values"):
                caches.append(
                    mx.concatenate(
                        [
                            mx.pad(
                                s[field][i],
                                [
                                    (0, 0),
                                    (0, 0),
                                    (length - s["positions"].size, 0),
                                    (0, 0),
                                ],
                            )
                            for s in states
                        ],
                        axis=0,
                    )
                )
            out = mx.fast.scaled_dot_product_attention(
                q,
                mx.concatenate([caches[0], k], axis=2),
                mx.concatenate([caches[1], v], axis=2),
                scale=attn.scale,
                mask=mask,
            )
            out = attn.o_proj(
                out.transpose(0, 2, 1, 3).reshape(batch, self.block_size, -1)
            )
            h = (h + layer.attention_conv.finish(out, kernel))[:, :ffn_count]
            x, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
            h = h + layer.mlp_conv.finish(layer.mlp(x), kernel)
        hidden = self.norm(h)[:, 1 : num_drafts + 1]
        logits = lm_head(hidden)[..., : self.vocab_size].astype(mx.float32)
        multiplier = float(
            self.config["dflash_config"].get(
                "output_multiplier", self.config.get("output_multiplier", 1.0)
            )
        )
        logits = logits * multiplier
        softcap = self.config["dflash_config"].get(
            "final_logit_softcapping", self.config.get("final_logit_softcapping")
        )
        if softcap is not None and float(softcap) > 0:
            logits = mx.tanh(logits / float(softcap)) * float(softcap)
        return self.candidate_selector.select(hidden, logits, anchors)


def load_dflash2(model_dir: Path, target: Any) -> DFlash2Model:
    """Strictly load a local, unquantized BF16 snapshot and validate target geometry."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Expected a DFlash2DraftModel checkpoint")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("DFlash2 requires the original BF16 checkpoint")
    args = _config_dict(target.args)
    args.update(args.get("text_config") or {})
    for name in ("hidden_size", "vocab_size"):
        if args.get(name) != config.get(name):
            raise ValueError(f"DFlash2 target {name} does not match checkpoint")
    if args.get("num_hidden_layers") != config.get("num_target_layers"):
        raise ValueError("DFlash2 target depth does not match checkpoint")
    index_path = model_dir / "model.safetensors.index.json"
    expected = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        expected = set(weight_map)
        names = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in names):
            raise ValueError("Checkpoint shards must be local snapshot filenames")
        paths = [model_dir / name for name in names]
    else:
        paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise ValueError("No local DFlash2 safetensors found")
    weights = {}
    for path in paths:
        shard = mx.load(str(path))
        if not isinstance(shard, dict) or weights.keys() & shard.keys():
            raise ValueError("Invalid or duplicate DFlash2 checkpoint tensors")
        if expected is not None and any(
            weight_map[name] != path.name for name in shard if name in weight_map
        ):
            raise ValueError("DFlash2 checkpoint tensor in incorrect shard")
        for name, value in shard.items():
            if value.dtype != mx.bfloat16:
                raise ValueError(
                    f"DFlash2 tensor {name} must be BF16, got {value.dtype}"
                )
        weights.update(shard)
    if expected is not None and set(weights) != expected:
        raise ValueError("DFlash2 shard tensors disagree with checkpoint index")
    model = DFlash2Model(config)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
