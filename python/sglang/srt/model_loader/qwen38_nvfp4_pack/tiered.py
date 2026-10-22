# SPDX-License-Identifier: Apache-2.0
"""Small cache slabs plus bucketed, immutable exception banks for large experts.

Only outlier experts externalize their complete replacement-word array. Keeping
it contiguous preserves both B12X readers; ordinary experts retain inline words.
Header 976 is the payload offset, 992 the GPU-only pointer, 1000 the bank extent,
and 1016/1020 the tier marker/base stride. On-disk GPU pointers are always zero.
"""

import argparse
import json
import mmap
import struct
from dataclasses import replace
from pathlib import Path

from .csf import BITMAP_ENCODING, validate_bitmap
from .format import REQUIRED_TENSORS, TensorRecord, load_manifest, write_manifest

TIERED_ENCODING = "b12x_csf_tiered_v1"
TIER_MAGIC = 0x31524954


def fixed_bytes(rows, columns):
    return 1024 + rows * (1 + columns // 2) + rows * columns // 16


def restore_slab(mapping, start, spec, region):
    """Bounds-check an external bank before producing a legacy bitmap slab."""
    stride = spec["stride"]
    head = bytearray(mapping[start : start + stride])
    offset = struct.unpack_from("<Q", head, 976)[0]
    gpu = struct.unpack_from("<Q", head, 992)[0]
    extent = struct.unpack_from("<I", head, 1000)[0]
    marker, stored_stride = struct.unpack_from("<2I", head, 1016)
    if (
        marker != TIER_MAGIC
        or stored_stride != stride
        or gpu
        or struct.unpack_from("<Q", head, 984)[0]
        or struct.unpack_from("<I", head, 1004)[0]
    ):
        raise ValueError("invalid tiered CSF header")
    count = struct.unpack_from("<I", head, 1008)[0]
    fixed = fixed_bytes(spec["rows"], spec["columns"])
    if offset:
        lo, hi = region["offset"], region["offset"] + region["nbytes"]
        if (
            offset % 16
            or extent % 16384
            or not lo <= offset <= hi
            or not count * 4 <= extent <= hi - offset
        ):
            raise ValueError("invalid tiered CSF bank extent")
        head = head[:fixed] + mapping[offset : offset + extent]
    elif extent or fixed + ((count + 3) // 4 * 16) > stride:
        raise ValueError("tiered CSF words exceed the base slab")
    head[976:1008] = bytes(32)
    head[1016:1024] = bytes(8)
    validate_bitmap(head, spec["rows"], spec["columns"])
    return head


def tier_pack(source, payload, manifest_path, *, base_strides=None):
    """Repack existing CSF once, without re-encoding scales or expanding weights."""
    src = load_manifest(source)
    if not src.scale_encoding or any(
        s["encoding"] != BITMAP_ENCODING for s in src.scale_encoding.values()
    ):
        raise ValueError("tiering requires an ordinary bitmap CSF source")
    payload, manifest_path = Path(payload).resolve(), Path(manifest_path).resolve()
    if payload == src.payload_path or payload.exists() or manifest_path.exists():
        raise ValueError("tiering requires fresh output paths")
    payload.parent.mkdir(parents=True, exist_ok=True)
    specs = {
        name: dict(
            s,
            encoding=TIERED_ENCODING,
            stride=(128 if name.startswith("w13") else 64) * 1024,
            overflow={},
        )
        for name, s in src.scale_encoding.items()
    }
    for name, stride in (base_strides or {}).items():
        specs[name]["stride"] = stride
    table, report = {}, {"planes": {}, "status": "running"}
    temporary = payload.with_suffix(payload.suffix + ".partial")
    with (
        src.payload_path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping,
        temporary.open("w+b") as out,
    ):
        for layer, records in src.tensors.items():
            table[layer] = {}
            for name, record in records.items():
                out.write(bytes((-out.tell()) % 4096))
                offset = out.tell()
                if name not in specs:
                    for position in range(0, record.nbytes, 8 << 20):
                        out.write(
                            mapping[
                                record.offset + position : record.offset
                                + min(position + (8 << 20), record.nbytes)
                            ]
                        )
                    table[layer][name] = replace(record, offset=offset)
                    continue
                old, spec = src.scale_encoding[name], specs[name]
                fixed, stride = (
                    fixed_bytes(spec["rows"], spec["columns"]),
                    spec["stride"],
                )
                if not fixed <= stride < spec["rows"] * spec["columns"]:
                    raise ValueError("tier base does not fit this model geometry")
                heads, banks = [], []
                for expert in range(src.num_experts):
                    start = record.offset + expert * old["stride"]
                    raw = mapping[start : start + old["stride"]]
                    validate_bitmap(raw, spec["rows"], spec["columns"])
                    count = struct.unpack_from("<I", raw, 1008)[0]
                    length = fixed + ((count + 3) // 4 * 16)
                    head = bytearray(stride)
                    head[: min(length, stride)] = raw[: min(length, stride)]
                    struct.pack_into("<2I", head, 1016, TIER_MAGIC, stride)
                    bank = b""
                    if length > stride:
                        # External words remain contiguous for the A4 decoder.
                        bank = raw[fixed : fixed + count * 4]
                        bank += bytes((-len(bank)) % 16384)
                        head[fixed:] = bytes(stride - fixed)
                    heads.append(head)
                    banks.append(bank)
                bank_offset = offset + src.num_experts * stride
                region = dict(offset=bank_offset, nbytes=sum(map(len, banks)))
                spec["overflow"][str(layer)] = region
                position = bank_offset
                for head, bank in zip(heads, banks):
                    if bank:
                        struct.pack_into("<Q", head, 976, position)
                        struct.pack_into("<I", head, 1000, len(bank))
                        position += len(bank)
                    out.write(head)
                for bank in banks:
                    out.write(bank)
                table[layer][name] = TensorRecord(
                    offset, src.num_experts * stride, (src.num_experts, stride), "uint8"
                )
                stats = report["planes"].setdefault(
                    name,
                    dict(
                        base_stride=stride,
                        external_experts=0,
                        resident_bytes=0,
                        buckets={},
                    ),
                )
                for bank in banks:
                    if bank:
                        stats["external_experts"] += 1
                        stats["resident_bytes"] += len(bank)
                        key = str(len(bank))
                        stats["buckets"][key] = stats["buckets"].get(key, 0) + 1
            print("tiered layer", layer, flush=True)
        result = replace(
            src,
            payload_path=payload,
            payload_nbytes=out.tell(),
            tensors=table,
            scale_encoding=specs,
        )
    try:
        result.validate_cutlass_layout()
        temporary.replace(payload)
        write_manifest(result, manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report.update(
        status="passed",
        payload_bytes=result.payload_nbytes,
        slot_bytes=sum(table[0][n].nbytes for n in REQUIRED_TENSORS) // src.num_experts,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(tier_pack(args.source, args.payload, args.manifest), indent=2))
