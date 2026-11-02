# SPDX-License-Identifier: Apache-2.0
"""Persistent lossless runtime-layout weights for ModelOpt GLM checkpoints."""

import hashlib
import json
import os
import re
import struct
import uuid
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_LAYOUT_VERSION = 1


def _walk_modules(module, prefix=""):
    for name, value in module.items():
        if name.startswith("_"):
            continue
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, nn.Module):
            yield key, value
            yield from _walk_modules(value, key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if isinstance(child, nn.Module):
                    path = f"{key}.{i}"
                    yield path, child
                    yield from _walk_modules(child, path)


def _owner(module, key):
    parts = key.split(".")
    for part in parts[:-1]:
        module = module[int(part)] if isinstance(module, list) else module[part]
    return module, parts[-1]


class RuntimeWeightCache:
    """Cache final per-layer arrays, validated against checkpoint/layout identity."""

    def __init__(self, checkpoint, config):
        checkpoint = Path(checkpoint)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        digest = hashlib.sha256()
        for name in ("config.json", "model.safetensors.index.json"):
            digest.update((checkpoint / name).read_bytes())
        # Immutable HF revisions plus headers and local file stats identify all
        # inputs without rereading 165 GiB just to decide whether cache hits.
        revision = checkpoint.name if checkpoint.parent.name == "snapshots" else None
        digest.update(str(revision).encode())
        for name in sorted(set(index["weight_map"].values())):
            file = checkpoint / name
            stat = file.stat()
            digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
            with file.open("rb") as handle:
                size_bytes = handle.read(8)
                size = struct.unpack("<Q", size_bytes)[0]
                digest.update(size_bytes)
                digest.update(handle.read(size))
        digest.update(f"metal-fp4-layout-{_LAYOUT_VERSION}".encode())
        model_id = checkpoint.parent.parent.name if revision else checkpoint.name
        model_id = re.sub(r"[^A-Za-z0-9._-]", "_", model_id.removeprefix("models--"))
        self.identity = digest.hexdigest()
        self.path = (
            Path.home()
            / ".cache"
            / "sglang"
            / "metal"
            / f"{model_id}-{self.identity[:24]}"
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = config

    def load(self, key, module):
        from .glm5_fp4 import FP8Linear, NVFP4SwitchLinear

        file = self.path / f"{key}.safetensors"
        meta_file = self.path / f"{key}.json"
        try:
            meta = json.loads(meta_file.read_text())
            if (
                meta["identity"] != self.identity
                or file.stat().st_size != meta["bytes"]
            ):
                return False
        except (OSError, ValueError, KeyError):
            return False
        try:
            with file.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size > min(file.stat().st_size - 8, 64 * 2**20):
                    return False
                header = json.loads(handle.read(header_size))
            embedded = header.get("__metadata__", {})
            if embedded.get("identity") != self.identity:
                return False
            if meta.get("generation") != embedded.get("generation"):
                return False
            arrays = mx.load(file)
            mx.eval(list(arrays.values()))
        except (OSError, ValueError, RuntimeError, struct.error):
            return False
        packed = None
        if "__qkv.weight" in arrays:
            packed = (arrays.pop("__qkv.weight"), arrays.pop("__qkv.scales"))
            dim = module.self_attn.dim
            for name, weight, scales in zip(
                ("q", "k", "v"),
                mx.split(packed[0], (dim, 2 * dim), 0),
                mx.split(packed[1], (dim, 2 * dim), 0),
            ):
                arrays[f"self_attn.{name}_proj.weight"] = weight
                arrays[f"self_attn.{name}_proj.scales"] = scales
        # Instantiate packed projection owners with their real operands before
        # loading the remaining ordinary norms, gates and recurrent weights.
        for name, description in meta["projections"].items():
            owner, field = _owner(module, name)

            def get(suffix):
                return arrays.pop(f"{name}.{suffix}", None)

            if description["kind"] == "fp8":
                owner[field] = FP8Linear(
                    get("weight"),
                    get("scales"),
                    block_size=description["block_size"],
                    activation_mode=description["activation_mode"],
                    bias=get("bias"),
                )
                decoded = get("decoded_weight")
                if decoded is not None:
                    owner[field].decoded_weight = decoded
            else:
                owner[field] = NVFP4SwitchLinear(
                    get("weight"),
                    get("scales"),
                    get("weight_scale_2"),
                    input_scale=get("input_scale"),
                )
        module.load_weights(list(arrays.items()), strict=False)
        if packed is not None:
            attn = module.self_attn
            splits = (attn.dim, 2 * attn.dim)
            for name, weight, scales in zip(
                ("q", "k", "v"),
                mx.split(packed[0], splits, 0),
                mx.split(packed[1], splits, 0),
            ):
                projection = getattr(attn, name + "_proj")
                projection.weight, projection.scales = weight, scales
            object.__setattr__(attn, "_cached_fp8_qkv", packed)
            attn.prepare_for_inference()
        mx.eval(module.parameters())
        return True

    def save(self, key, module):
        from .glm5_fp4 import FP8KDAInputs, FP8Linear, NVFP4SwitchLinear

        arrays = dict(tree_flatten(module.parameters()))
        descriptions = {}
        for name, projection in _walk_modules(module):
            if isinstance(projection, FP8Linear):
                descriptions[name] = dict(
                    kind="fp8",
                    block_size=projection.block_size,
                    activation_mode=projection.activation_mode,
                )
            elif isinstance(projection, NVFP4SwitchLinear):
                descriptions[name] = dict(kind="nvfp4")
        attn = getattr(module, "self_attn", None)
        if attn is not None and isinstance(
            attn._fused_qkv if hasattr(attn, "_fused_qkv") else None, FP8KDAInputs
        ):
            bank = attn._fused_qkv
            arrays["__qkv.weight"], arrays["__qkv.scales"] = bank.weight, bank.scales
            for name in ("q", "k", "v"):
                for field in ("weight", "scales"):
                    arrays.pop(f"self_attn.{name}_proj.{field}", None)
                # Descriptors need operands while loading; restore bank views
                # before constructing these projections on the warm path.
        token = uuid.uuid4().hex
        file = self.path / f"{key}.safetensors"
        pending = self.path / f".{key}.{token}.safetensors"
        meta_file = self.path / f"{key}.json"
        pending_meta = self.path / f".{key}.{token}.json"
        try:
            mx.save_safetensors(
                pending,
                arrays,
                metadata={"identity": self.identity, "generation": token},
            )
            pending_meta.write_text(
                json.dumps(
                    dict(
                        identity=self.identity,
                        generation=token,
                        bytes=pending.stat().st_size,
                        projections=descriptions,
                    ),
                    sort_keys=True,
                )
            )
            os.replace(pending, file)
            os.replace(pending_meta, meta_file)
        finally:
            pending.unlink(missing_ok=True)
            pending_meta.unlink(missing_ok=True)
