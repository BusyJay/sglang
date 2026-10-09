#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a Qwen3.8 runtime pack from already transformed layer tensors.

The GPU conversion step should export one directory per layer containing the
six per-expert tensors and two layer input scales as raw byte files plus ``metadata.json``. This CPU tool
assembles those artifacts into one mmap-friendly payload and an ABI manifest;
it never downloads a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from .format import (
        PACK_TENSORS,
        Qwen38ExpertPackManifest,
        TensorRecord,
        write_manifest,
    )
except ImportError:
    from format import (  # type: ignore
        PACK_TENSORS,
        Qwen38ExpertPackManifest,
        TensorRecord,
        write_manifest,
    )


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def build_pack(
    source: str | Path,
    payload_path: str | Path,
    manifest_path: str | Path,
    *,
    checkpoint_revision: str,
    sglang_commit: str,
    num_layers: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    alignment: int = 4096,
) -> Qwen38ExpertPackManifest:
    source = Path(source).resolve()
    payload_path = Path(payload_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    records: dict[int, dict[str, TensorRecord]] = {}
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_name(payload_path.name + ".partial")
    offset = 0
    with temporary.open("w+b") as output:
        for layer in range(num_layers):
            layer_dir = source / f"layer-{layer:03d}"
            metadata_path = layer_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing layer metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records[layer] = {}
            for name in PACK_TENSORS:
                info = metadata.get(name)
                if not isinstance(info, dict):
                    raise ValueError(f"{metadata_path} is missing {name}")
                tensor_file = layer_dir / str(info.get("file", f"{name}.bin"))
                nbytes = tensor_file.stat().st_size
                offset = align_up(offset, alignment)
                output.seek(offset)
                with tensor_file.open("rb") as tensor_input:
                    shutil.copyfileobj(tensor_input, output, length=1024 * 1024)
                records[layer][name] = TensorRecord(
                    offset=offset,
                    nbytes=nbytes,
                    shape=tuple(int(value) for value in info["shape"]),
                    dtype=str(info["dtype"]),
                )
                offset += nbytes
        output.truncate(offset)
        output.flush()
    manifest = Qwen38ExpertPackManifest(
        payload_path=payload_path,
        payload_nbytes=temporary.stat().st_size,
        checkpoint_revision=checkpoint_revision,
        sglang_commit=sglang_commit,
        flashinfer_backend="flashinfer_cutlass",
        compute_capability="sm120",
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tensors=records,
    )
    manifest.validate()
    temporary.replace(payload_path)
    write_manifest(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--intermediate-size", type=int, default=640)
    parser.add_argument("--alignment", type=int, default=4096)
    args = parser.parse_args()
    manifest = build_pack(
        args.source,
        args.payload,
        args.manifest,
        checkpoint_revision=args.checkpoint_revision,
        sglang_commit=args.sglang_commit,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        alignment=args.alignment,
    )
    print(json.dumps(manifest.to_dict(manifest_path=args.manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
