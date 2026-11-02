# SPDX-License-Identifier: Apache-2.0
"""Index-driven NVIDIA GLM ModelOpt checkpoint reader.

The reader preserves NVFP4/MXFP8 bytes and original FP32 scale tensors.
Execution contracts are explicit: MXFP8 A16 and block-FP8 dynamic A8.
"""

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sglang.kernels.ops.gemm.mlx_fp8_rows import mlx_fp8_dynamic_32, mlx_fp8_rows
from sglang.kernels.ops.quantization.mlx_csf import mlx_csf32_restore


def mxfp8_project_rows(x, weight, scales):
    """Use the same eight-row reduction geometry for AR and verification.

    Rows are independent; zero padding only stabilizes the native dispatch.
    Packed operands and output dtype are preserved, and real rows share weights.
    """
    values = x.reshape(-1, x.shape[-1])
    outputs = []
    for start in range(0, values.shape[0], 8):
        tile = values[start : start + 8]
        count = tile.shape[0]
        if count < 8:
            tile = mx.concatenate(
                [tile, mx.zeros((8 - count, tile.shape[1]), tile.dtype)], 0
            )
        out = mx.quantized_matmul(
            tile,
            weight.view(mx.uint32),
            scales,
            mode="mxfp8",
            group_size=32,
            bits=8,
        )
        outputs.append(out[:count])
    return mx.concatenate(outputs, 0).reshape(*x.shape[:-1], weight.shape[0])


class FP8Linear(nn.Module):
    def __init__(self, weight, scales, *, block_size, activation_mode, bias=None):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.block_size = block_size
        self.activation_mode = activation_mode
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        out = self.project(x)
        return out + self.bias if "bias" in self else out

    def project(self, x):
        quantized = (
            mlx_fp8_dynamic_32(x) if self.activation_mode == "dynamic32" else None
        )
        if self.block_size == 32 and quantized is None:
            return mxfp8_project_rows(x, self.weight, self.scales)
        return mlx_fp8_rows(
            x,
            self.weight,
            self.scales,
            block_size=self.block_size,
            quantized_input=quantized,
        )


def checkpoint_recipe(config):
    recipe = config.get("quantization_config", {})
    return recipe.get("quant_method") == "modelopt" and bool(
        recipe.get("quantized_layers")
    )


class IndexedWeights:
    """Cache shard headers and copy only each indexed tensor byte range."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.values = {}

    def get(self, key):
        shard = self.index[key]
        if shard not in self.values:
            with (self.path / shard).open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            self.values[shard] = (size + 8, header)
        origin, header = self.values[shard]
        item = header[key]
        dtypes = {
            "U8": np.uint8,
            "F8_E4M3": np.uint8,
            "U16": np.uint16,
            "U32": np.uint32,
            "I32": np.int32,
            "I64": np.int64,
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
        }
        source = np.memmap(
            self.path / shard,
            mode="r",
            dtype=dtypes[item["dtype"]],
            offset=origin + item["data_offsets"][0],
            shape=tuple(item["shape"]),
        )
        result = mx.array(source)
        if item["dtype"] == "BF16":
            result = result.view(mx.bfloat16)
        return result

    def csf(self, prefix, rows, columns):
        parts = [
            self.get(prefix + "." + suffix)
            for suffix in (
                "csf32_bases",
                "csf32_codes",
                "csf32_page_bases",
                "csf32_offsets",
                "csf32_exceptions",
            )
        ]
        out = mlx_csf32_restore(*parts, rows=rows, columns=columns)
        mx.eval(out)
        return out


def read_projection(reader, prefix, recipe):
    """Return original packed operands, including calibration; never requantize."""
    weight = reader.get(prefix + ".weight")
    algo = recipe["quant_algo"]
    if algo in ("NVFP4", "W4A16_NVFP4"):
        rows, packed_k = weight.shape
        if recipe.get("weight_scale_encoding") == "csf_32x2_v1":
            scales = reader.csf(prefix + ".weight_scale", rows, packed_k // 8)
        else:
            scales = reader.get(prefix + ".weight_scale")
        return dict(
            weight=weight,
            scales=scales,
            weight_scale_2=reader.get(prefix + ".weight_scale_2"),
            input_scale=(
                reader.get(prefix + ".input_scale")
                if prefix + ".input_scale" in reader.index
                else None
            ),
        )
    return dict(weight=weight, scales=reader.get(prefix + ".weight_scale"))


class NVFP4Linear(nn.Module):
    """Dense A16 projection for vision's serialized NVFP4 operands."""

    def __init__(self, weight, scales, weight_scale_2, input_scale=None, bias=None):
        super().__init__()
        self.weight, self.scales = weight, scales
        self.weight_scale_2 = weight_scale_2
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        out = mx.gather_qmm(
            x.reshape(-1, 1, 1, x.shape[-1]),
            self.weight.view(mx.uint32)[None],
            self.scales[None],
            rhs_indices=mx.zeros((x.size // x.shape[-1], 1), mx.uint32),
            mode="nvfp4",
            group_size=16,
            bits=4,
            global_scale=self.weight_scale_2.reshape(1) * 2688,
        ).reshape(*x.shape[:-1], self.weight.shape[0])
        return out + self.bias if "bias" in self else out


class NVFP4SwitchLinear(nn.Module):
    def __init__(self, weight, scales, weight_scale_2, input_scale=None):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(
            x,
            self.weight.view(mx.uint32),
            self.scales,
            rhs_indices=indices,
            mode="nvfp4",
            group_size=16,
            bits=4,
            global_scale=self.weight_scale_2 * 2688,
            sorted_indices=sorted_indices,
        )


def load_fp4_model(path, *, enable_mtp=False, memory_budget=None, runtime_cache=True):
    """Build model from indexed operands, one target layer at a time."""
    from sglang.srt.hardware_backend.mlx.models.glm5_next import Model, ModelArgs, MoE

    if memory_budget is None:
        memory_budget = int(mx.device_info()["max_recommended_working_set_size"] * 0.75)
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config), enable_mtp=enable_mtp)
    reader = IndexedWeights(path)
    recipes = config["quantization_config"]["quantized_layers"]
    from .mlx_fp4_cache import RuntimeWeightCache

    weight_cache = RuntimeWeightCache(path, config) if runtime_cache else None

    def replace_projection(owner, name, prefix, recipe):
        values = read_projection(reader, prefix, recipe)
        if recipe["quant_algo"] in ("NVFP4", "W4A16_NVFP4"):
            module = NVFP4SwitchLinear(**values)
        else:
            block = 32 if recipe["quant_algo"] == "MXFP8" else 128
            module = FP8Linear(
                **values,
                block_size=block,
                activation_mode="a16" if block == 32 else "dynamic32",
            )
        setattr(owner, name, module)

    def load_owner(owner, prefix):
        # Quantized projections replace their floating placeholders first.
        for name, child in list(owner.children().items()):
            if not isinstance(child, nn.Module):
                continue
            cp = prefix + "." + name
            if cp in recipes and recipes[cp]["quant_algo"] not in (
                "NVFP4",
                "W4A16_NVFP4",
            ):
                replace_projection(owner, name, cp, recipes[cp])
            elif hasattr(child, "weight") and cp + ".weight" in reader.index:
                child.weight = reader.get(cp + ".weight")
                if cp + ".bias" in reader.index:
                    child.bias = reader.get(cp + ".bias")
            elif name not in ("q_conv", "k_conv", "v_conv"):
                load_owner(child, cp)
        # Non-module parameters (mHC, recurrent gates, correction bias).
        for key, value in list(owner.items()):
            if not isinstance(value, mx.array):
                continue
            source = prefix + "." + key
            if source in reader.index:
                owner[key] = reader.get(source)
            elif (
                key == "e_score_correction_bias"
                and prefix + ".gate." + key in reader.index
            ):
                owner[key] = reader.get(prefix + ".gate." + key)

    for i, layer in enumerate(model.model.layers):
        prefix = f"model.language_model.layers.{i}"
        if weight_cache is not None and weight_cache.load(f"layer-{i:02d}", layer):
            if mx.get_active_memory() > memory_budget:
                raise MemoryError("Cached FP4 weights exceed load budget")
            print(f"FP4 cache hit layer {i + 1}/{len(model.model.layers)}", flush=True)
            continue
        # Conv layout uses the canonical checkpoint name, not MLX ShortConv.
        if layer.is_linear:
            for name in ("q", "k", "v"):
                key = prefix + f".self_attn.{name}_conv1d.weight"
                getattr(layer.self_attn, name + "_conv").conv.weight = reader.get(
                    key
                ).swapaxes(1, 2)
        load_owner(layer, prefix)
        if isinstance(layer.mlp, MoE):
            ep = prefix + ".mlp.experts"
            recipe = recipes.get(ep)
            if recipe is not None:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    if mx.get_active_memory() + 3 * 2**30 > memory_budget:
                        raise MemoryError(
                            "FP4 bank assembly exceeds pre-allocation budget"
                        )
                    parts = [
                        read_projection(reader, f"{ep}.{e}.{projection}", recipe)
                        for e in range(model.args.n_routed_experts)
                    ]
                    bank = {
                        k: mx.stack([v[k] for v in parts])
                        for k in parts[0]
                        if parts[0][k] is not None
                    }
                    mx.eval(bank)
                    parts.clear()
                    mx.clear_cache()
                    setattr(layer.mlp.switch_mlp, projection, NVFP4SwitchLinear(**bank))
        if layer.is_linear:
            layer.self_attn.A_log = layer.self_attn.A_log.reshape(-1)
        # An absorbed KV-B matrix needs decoded values; cache this small
        # operand once while retaining the serialized packed projection.
        if not layer.is_linear and isinstance(layer.self_attn.kv_b_proj, FP8Linear):
            layer.self_attn.kv_b_proj.decoded_weight = mx.dequantize(
                layer.self_attn.kv_b_proj.weight.view(mx.uint32),
                layer.self_attn.kv_b_proj.scales,
                mode="mxfp8",
                group_size=32,
                bits=8,
            )
        if weight_cache is not None:
            if layer.is_linear:
                layer.self_attn.prepare_for_inference()
            weight_cache.save(f"layer-{i:02d}", layer)
        mx.eval(layer.parameters())
        mx.clear_cache()
        print(
            "FP4 memory GiB",
            mx.get_active_memory() / 2**30,
            mx.get_peak_memory() / 2**30,
            flush=True,
        )
        if mx.get_active_memory() > memory_budget:
            raise MemoryError(
                "FP4 load exceeds bounded 180 GiB budget; inspect retained operands"
            )
        print(f"FP4 loaded layer {i + 1}/{len(model.model.layers)}", flush=True)
    auxiliary = nn.Module()
    auxiliary.embed_tokens = model.model.embed_tokens
    auxiliary.norm = model.model.norm
    auxiliary.lm_head = model.lm_head
    if weight_cache is not None and weight_cache.load("auxiliary", auxiliary):
        model.model.embed_tokens = auxiliary.embed_tokens
        model.model.norm = auxiliary.norm
        model.lm_head = auxiliary.lm_head
        print("FP4 cache hit embedding/norm/head", flush=True)
    else:
        model.model.embed_tokens.weight = reader.get(
            "model.language_model.embed_tokens.weight"
        )
        model.model.norm.weight = reader.get("model.language_model.norm.weight")
        replace_projection(model, "lm_head", "lm_head", recipes["lm_head"])
        auxiliary.lm_head = model.lm_head
        mx.eval(auxiliary.parameters())
        if weight_cache is not None:
            weight_cache.save("auxiliary", auxiliary)
    model.eval()
    return model, config


class FP8KDAInputs(nn.Module):
    """One packed QKV weight bank, explicit dense gate operands."""

    def __init__(self, projections, *, packed=None):
        super().__init__()
        q, k, v, fa, ga, beta = projections
        sizes = (q.weight.shape[0], k.weight.shape[0])
        if packed is None:
            self.weight = mx.concatenate([p.weight for p in (q, k, v)], 0)
            self.scales = mx.concatenate([p.scales for p in (q, k, v)], 0)
        else:
            self.weight, self.scales = packed
        mx.eval(self.weight, self.scales)
        for p, w, s in zip(
            (q, k, v),
            mx.split(self.weight, (sizes[0], sum(sizes)), 0),
            mx.split(self.scales, (sizes[0], sum(sizes)), 0),
        ):
            p.weight = w
            p.scales = s
        self.fa = fa
        self.ga = ga
        self.beta = beta

    def __call__(self, x):
        from .glm5_next import _project_rows

        qkv = mxfp8_project_rows(x, self.weight, self.scales)
        gates = tuple(_project_rows(p, x) for p in (self.fa, self.ga, self.beta))
        return mx.concatenate((qkv, *gates), -1)


def load_fp4_vision(path):
    """Read mixed vision projections by exact indexed ranges, preserving biases."""
    from sglang.srt.hardware_backend.mlx.models.glm5_vision import Glm5Vision

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Glm5Vision(config["vision_config"])
    reader = IndexedWeights(path)
    recipes = config["quantization_config"]["quantized_layers"]

    def visit(owner, prefix):
        for name, value in list(owner.items()):
            source = prefix + "." + name
            if isinstance(value, nn.Module):
                if source in recipes:
                    recipe = recipes[source]
                    operands = read_projection(reader, source, recipe)
                    if source + ".bias" in reader.index:
                        operands["bias"] = reader.get(source + ".bias")
                    if recipe["quant_algo"] in ("NVFP4", "W4A16_NVFP4"):
                        owner[name] = NVFP4Linear(**operands)
                    else:
                        owner[name] = FP8Linear(
                            **operands, block_size=32, activation_mode="a16"
                        )
                else:
                    visit(value, source)
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    visit(child, source + "." + str(i))
            elif isinstance(value, mx.array):
                if source.startswith("model.visual.patch_embed."):
                    source = source.replace("patch_embed.", "patch_embed.proj.")
                loaded = reader.get(source)
                if source in (
                    "model.visual.patch_embed.proj.weight",
                    "model.visual.downsample.weight",
                ):
                    loaded = loaded.reshape(loaded.shape[0], -1)
                owner[name] = loaded

    visit(model, "model.visual")
    model.eval()
    mx.eval(model.parameters())
    return model
