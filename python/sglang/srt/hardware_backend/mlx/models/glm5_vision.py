# SPDX-License-Identifier: Apache-2.0
"""GLM5-Next vision tower with checkpoint-native BF16 weights."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn


class VisionRMSNorm(nn.Module):
    """Match checkpoint RMSNorm's cast before applying the learned scale."""

    def __init__(self, width, eps):
        super().__init__()
        self.weight = mx.ones((width,))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(mx.float32)
        normalized = h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + self.eps)
        return normalized.astype(x.dtype) * self.weight


class VisionMLP(nn.Module):
    def __init__(self, width, intermediate, limit, bias):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=bias)
        self.up_proj = nn.Linear(width, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, width, bias=bias)
        self.limit = limit

    def __call__(self, x):
        gate = mx.minimum(self.gate_proj(x), self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        activated = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
        return self.down_proj(activated * up)


class VisionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["hidden_size"]
        self.heads = cfg["num_heads"]
        self.head_dim = width // self.heads
        self.qkv = nn.Linear(width, width * 3, bias=cfg["attention_bias"])
        self.proj = nn.Linear(width, width, bias=cfg["attention_bias"])
        self.q_norm = VisionRMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = VisionRMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def __call__(self, x, cos, sin, lengths):
        q, k, v = mx.split(
            self.qkv(x).reshape(-1, 3, self.heads, self.head_dim), 3, axis=1
        )
        q, k, v = (a[:, 0] for a in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)

        def rotate(a):
            first, second = mx.split(a, 2, axis=-1)
            return (
                a * cos[:, None] + mx.concatenate([-second, first], -1) * sin[:, None]
            ).astype(a.dtype)

        q, k = rotate(q), rotate(k)
        groups, start = {}, 0
        for frame, length in enumerate(lengths):
            groups.setdefault(length, []).append((frame, start))
            start += length
        outputs = [None] * len(lengths)
        # Batch frames of equal patch count; variable-resolution groups retain
        # private attention domains without a quadratic cross-image mask.
        for length, frames in groups.items():
            qs, ks, vs = (
                mx.stack(
                    [
                        a[start : start + length].transpose(1, 0, 2)
                        for _, start in frames
                    ]
                )
                for a in (q, k, v)
            )
            result = mx.fast.scaled_dot_product_attention(
                qs, ks, vs, scale=self.head_dim**-0.5
            )
            for batch, (frame, _) in enumerate(frames):
                outputs[frame] = result[batch].transpose(1, 0, 2).reshape(length, -1)
        return self.proj(mx.concatenate(outputs, 0))


class VisionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = VisionRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norm2 = VisionRMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(
            cfg["hidden_size"], cfg["intermediate_size"], cfg["swiglu_limit"], True
        )

    def __call__(self, x, cos, sin, lengths):
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class VisionMerger(VisionMLP):
    def __init__(self, cfg):
        width = cfg["out_hidden_size"]
        super().__init__(
            width,
            cfg.get("projection_intermediate_size", cfg["intermediate_size"]),
            cfg["swiglu_limit"],
            False,
        )
        self.proj = nn.Linear(width, width, bias=False)
        self.post_projection_norm = nn.LayerNorm(width, eps=1e-5)

    def __call__(self, x):
        x = self.post_projection_norm(self.proj(x))
        return super().__call__(nn.gelu(x.astype(mx.float32)).astype(x.dtype))


class Glm5Vision(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width = cfg["hidden_size"]
        patch_dim = (
            cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
        )
        self.patch_embed = nn.Linear(patch_dim, width, bias=True)
        self.blocks = [VisionBlock(cfg) for _ in range(cfg["depth"])]
        self.post_layernorm = VisionRMSNorm(width, eps=cfg["rms_norm_eps"])
        self.downsample = nn.Linear(
            width * cfg["spatial_merge_size"] ** 2, cfg["out_hidden_size"], bias=True
        )
        self.merger = VisionMerger(cfg)

    def __call__(self, pixels, grid_thw):
        grid = np.asarray(grid_thw, dtype=np.int64).reshape(-1, 3)
        merge = self.cfg["spatial_merge_size"]
        coords, lengths = [], []
        for t, h, w in grid.tolist():
            if min(t, h, w) <= 0 or h % merge or w % merge:
                raise ValueError(
                    "Vision grid must contain positive dimensions divisible by spatial_merge_size"
                )
            row, col = np.indices((h, w))
            order = lambda a, h=h, w=w: (
                a.reshape(h // merge, merge, w // merge, merge)
                .transpose(0, 2, 1, 3)
                .reshape(-1)
            )
            coords.append(np.tile(np.stack([order(row), order(col)], -1), (t, 1)))
            lengths.extend([h * w] * t)
        positions = mx.array(np.concatenate(coords))
        if pixels.shape[0] != sum(lengths):
            raise ValueError("Pixel patch count does not match vision grid")
        head_dim = self.cfg["hidden_size"] // self.cfg["num_heads"]
        inv = 1.0 / (
            10000
            ** (mx.arange(0, head_dim // 2, 2, dtype=mx.float32) / (head_dim // 2))
        )
        angles = (positions[..., None] * inv).reshape(-1, head_dim // 2)
        cos, sin = mx.cos(angles), mx.sin(angles)
        cos, sin = mx.concatenate([cos, cos], -1), mx.concatenate([sin, sin], -1)
        x = self.patch_embed(pixels.astype(self.patch_embed.weight.dtype))
        for block in self.blocks:
            x = block(x, cos, sin, lengths)
        x = self.post_layernorm(x).reshape(-1, merge * merge, self.cfg["hidden_size"])
        # Conv2d's flattened checkpoint is channel-major, while patches are
        # spatial-major. Preserve the exact convolution input ordering.
        x = self.downsample(x.transpose(0, 2, 1).reshape(x.shape[0], -1))
        return self.merger(x)


def load_glm5_vision(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())["vision_config"]
    model = Glm5Vision(cfg)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(
            {
                v
                for k, v in mapping.items()
                if k.startswith(("model.visual.", "visual."))
            }
        )
    else:
        shards = [p.name for p in model_dir.glob("*.safetensors")]
    weights = {}
    for shard in shards:
        for key, value in mx.load(str(model_dir / shard)).items():
            if key.startswith("model.visual."):
                key = key[len("model.visual.") :]
            elif key.startswith("visual."):
                key = key[len("visual.") :]
            else:
                continue
            if key == "patch_embed.proj.weight":
                key, value = "patch_embed.weight", value.reshape(value.shape[0], -1)
            elif key == "patch_embed.proj.bias":
                key = "patch_embed.bias"
            elif key == "downsample.weight":
                value = value.reshape(value.shape[0], -1)
            weights[key] = value
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
