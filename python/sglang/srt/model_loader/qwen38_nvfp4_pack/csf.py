# SPDX-License-Identifier: Apache-2.0
"""Lossless native-order CSF slabs. Offsets refer to CUTLASS scale bytes.

Each expert uses B12X bitmap/rank operands in one aligned slab: a 1024-byte
zero header, interleaved row bases/nibbles, tile bitmaps, and replacement words.
The fixed expert stride supports cache relocation without runtime repacking.
Legacy exception-list helpers are used only by the offline converter.
"""

import argparse
import json
import mmap
import struct
from dataclasses import replace
from pathlib import Path

BITMAP_MAGIC = 0x32465343
BITMAP_ENCODING = "b12x_csf_bitmap_v1"

SCALE_NAMES = ("w13_blockscale_swizzled", "w2_blockscale_swizzled")


def native_rows(rows, columns):
    import numpy as np

    p = np.arange(rows * columns, dtype=np.uint32)
    return (p // (columns * 128)) * 128 + (p % 512) // 16 + ((p % 16) // 4) * 32


def encode_legacy(raw, rows, columns):
    import numpy as np

    if rows % 128 or columns % 4 or rows * columns >= 1 << 24:
        raise ValueError("unsupported CSF plane geometry")
    values = np.frombuffer(raw, dtype=np.uint8)
    if values.size != rows * columns:
        raise ValueError("CSF input extent differs from geometry")
    row = native_rows(rows, columns)
    logical = (
        values.reshape(rows // 128, columns // 4, 32, 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .reshape(rows, columns)
    )
    hist = np.zeros((rows, 256), dtype=np.int32)
    np.add.at(hist, (np.arange(rows)[:, None], logical), 1)
    scores = np.cumsum(hist, axis=1)
    scores[:, 16:] -= np.cumsum(hist, axis=1)[:, :-16]
    bases = np.maximum(scores.argmax(axis=1) - 15, 0).astype(np.uint8)
    delta = values.astype(np.int16) - bases[row].astype(np.int16)
    positions = np.flatnonzero((delta < 0) | (delta > 15)).astype(np.uint32)
    codes = np.clip(delta, 0, 15).astype(np.uint8)
    packed = codes[::2] | (codes[1::2] << 4)
    exceptions = positions | (values[positions].astype(np.uint32) << 24)
    return (
        struct.pack("<4I", len(positions), 0, 0, 0)
        + bases.tobytes()
        + packed.tobytes()
        + exceptions.astype("<u4").tobytes()
    )


def decode_legacy(raw, rows, columns):
    import numpy as np

    size = rows * columns
    if len(raw) < 16 + rows + size // 2:
        raise ValueError("truncated CSF slab")
    count, *reserved = struct.unpack_from("<4I", raw)
    end = 16 + rows + size // 2 + count * 4
    if any(reserved) or end > len(raw):
        raise ValueError("invalid CSF header/exception extent")
    bases = np.frombuffer(raw, np.uint8, rows, 16)
    packed = np.frombuffer(raw, np.uint8, size // 2, 16 + rows)
    values = np.empty(size, np.uint8)
    values[::2], values[1::2] = packed & 15, packed >> 4
    values += bases[native_rows(rows, columns)]
    exc = np.frombuffer(raw, "<u4", count, 16 + rows + size // 2)
    pos = exc & 0xFFFFFF
    if count and (int(pos[-1]) >= size or np.any(pos[1:] <= pos[:-1])):
        raise ValueError("invalid CSF exception positions")
    values[pos] = exc >> 24
    return values.tobytes()


def encode(raw, rows, columns):
    """Prepare one expert's native B12X inline operands, entirely offline."""
    import numpy as np

    legacy = encode_legacy(raw, rows, columns)
    count = struct.unpack_from("<I", legacy)[0]
    bases = np.frombuffer(legacy, np.uint8, rows, 16)
    codes = np.frombuffer(legacy, np.uint8, rows * columns // 2, 16 + rows)
    exceptions = np.frombuffer(legacy, "<u4", count, 16 + rows + rows * columns // 2)
    # B12X bases follow the 128-word native scale atom's lane order.
    lanes = np.arange(128)
    order = lanes // 4 + (lanes % 4) * 32
    fixed = np.concatenate(
        [
            np.concatenate(
                (
                    bases[r : r + 128][order],
                    codes[r * columns // 2 : (r + 128) * columns // 2],
                )
            )
            for r in range(0, rows, 128)
        ]
    )
    affected = np.unique((exceptions & 0xFFFFFF) // 4)
    tiles = rows * columns // 512
    masks = np.zeros((tiles, 4), dtype="<u4")
    np.bitwise_or.at(
        masks,
        (affected // 128, affected % 128 // 32),
        np.uint32(1) << (affected % 32).astype(np.uint32),
    )
    counts = np.bincount(affected // 32, minlength=tiles * 4).reshape(tiles, 4)
    prefix = np.cumsum(counts, axis=1, dtype=np.uint32) - counts.astype(np.uint32)
    prefixes = np.sum(
        prefix << (np.arange(4, dtype=np.uint32) * 8), axis=1, dtype=np.uint32
    )
    ends = np.cumsum(counts.sum(axis=1), dtype=np.uint32)
    metadata = np.column_stack(
        (ends - counts.sum(axis=1), prefixes, masks, ends, np.zeros(tiles, np.uint32))
    ).astype("<u4")
    replacements = np.frombuffer(raw, "<u4")[affected]
    header = bytearray(1024)
    struct.pack_into("<2I", header, 1008, len(affected), BITMAP_MAGIC)
    struct.pack_into("<I", header, 668, 1)  # B12X invalid-tile marker
    tail = replacements.tobytes()
    return (
        bytes(header)
        + fixed.tobytes()
        + metadata.tobytes()
        + tail
        + bytes((-len(tail)) % 16)
    )


def validate_bitmap(raw, rows, columns):
    """Check all bitmap/rank bounds without allocating an expanded plane."""
    import numpy as np

    fixed = rows * (1 + columns // 2)
    tiles = rows * columns // 512
    records = 1024 + fixed + tiles * 32
    if len(raw) < records:
        raise ValueError("truncated CSF slab")
    count, magic = struct.unpack_from("<2I", raw, 1008)
    if magic != BITMAP_MAGIC or records + ((count + 3) // 4 * 16) > len(raw):
        raise ValueError("invalid CSF slab header/exception extent")
    # Header is a safe zero operand; only the codec words and invalid marker differ.
    header = np.frombuffer(raw, "<u4", 256)
    if (
        header[167] != 1
        or np.any(header[:167])
        or np.any(header[168:252])
        or np.any(header[254:])
    ):
        raise ValueError("invalid CSF slab header")
    meta = np.frombuffer(raw, "<u4", tiles * 8, 1024 + fixed).reshape(tiles, 8)
    counts = (
        np.unpackbits(meta[:, 2:6].copy().view(np.uint8), axis=1)
        .reshape(tiles, 4, 32)
        .sum(axis=2)
    )
    prefix = np.cumsum(counts, axis=1) - counts
    prefixes = np.sum(
        prefix.astype(np.uint32) << (np.arange(4, dtype=np.uint32) * 8), axis=1
    )
    ends = np.cumsum(counts.sum(axis=1), dtype=np.uint64)
    if (
        np.any(meta[:, 0] != ends - counts.sum(axis=1))
        or np.any(meta[:, 1] != prefixes)
        or np.any(meta[:, 6] != ends)
        or np.any(meta[:, 7])
        or int(ends[-1]) != count
    ):
        raise ValueError("invalid CSF bitmap/rank exceptions")
    return meta, records


def decode(raw, rows, columns):
    """Offline/reference decoder; inference never calls this for GPU weights."""
    import numpy as np

    if len(raw) < 1024 or struct.unpack_from("<I", raw, 1012)[0] != BITMAP_MAGIC:
        return decode_legacy(raw, rows, columns)
    meta, records = validate_bitmap(raw, rows, columns)
    fixed = np.frombuffer(raw, np.uint8, rows * (1 + columns // 2), 1024).reshape(
        rows // 128, -1
    )
    codes = fixed[:, 128:].reshape(-1)
    values = np.empty(rows * columns, np.uint8)
    values[::2], values[1::2] = codes & 15, codes >> 4
    word_bases = np.tile(fixed[:, :128][:, None, :], (1, columns // 4, 1)).reshape(-1)
    values += np.repeat(word_bases, 4)
    flags = np.unpackbits(
        meta[:, 2:6].copy().view(np.uint8), axis=1, bitorder="little"
    ).reshape(-1)
    affected = np.flatnonzero(flags)
    values.view("<u4")[affected] = np.frombuffer(raw, "<u4", len(affected), records)
    return values.tobytes()


def compress_pack(source, payload, manifest_path):
    """Stream an existing native pack; never materialize the full model in RAM."""
    from .format import REQUIRED_TENSORS, TensorRecord, load_manifest, write_manifest

    m = load_manifest(source)
    if m.scale_encoding:
        raise ValueError("source pack is already compressed")
    payload, manifest_path = Path(payload).resolve(), Path(manifest_path).resolve()
    if payload == m.payload_path or manifest_path == Path(source).resolve():
        raise ValueError("compression requires separate output paths")
    encodings = {}
    payload.parent.mkdir(parents=True, exist_ok=True)
    with (
        m.payload_path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as src,
    ):
        for name in SCALE_NAMES:
            shape = m.tensors[0][name].shape
            rows, columns = shape[1:]
            maximum = 0
            for records in m.tensors.values():
                r = records[name]
                if r.shape != shape:
                    raise ValueError("CSF requires uniform scale layouts across layers")
                for expert in range(m.num_experts):
                    start = r.offset + expert * rows * columns
                    maximum = max(
                        maximum,
                        len(encode(src[start : start + rows * columns], rows, columns)),
                    )
            stride = (maximum + 15) // 16 * 16
            if stride < rows * columns:
                encodings[name] = dict(
                    encoding=BITMAP_ENCODING, rows=rows, columns=columns, stride=stride
                )
        if not encodings:
            raise ValueError("CSF does not reduce either plane; retain the native pack")
        table = {}
        temporary = payload.with_name(payload.name + ".partial")
        try:
            with temporary.open("wb") as out:
                for layer, records in m.tensors.items():
                    table[layer] = {}
                    for name, r in records.items():
                        out.write(bytes((-out.tell()) % 4096))
                        offset = out.tell()
                        if name in encodings:
                            spec = encodings[name]
                            size = spec["rows"] * spec["columns"]
                            for e in range(m.num_experts):
                                slab = encode(
                                    src[
                                        r.offset + e * size : r.offset + (e + 1) * size
                                    ],
                                    spec["rows"],
                                    spec["columns"],
                                )
                                out.write(slab)
                                out.write(bytes(spec["stride"] - len(slab)))
                            table[layer][name] = TensorRecord(
                                offset,
                                m.num_experts * spec["stride"],
                                (m.num_experts, spec["stride"]),
                                "uint8",
                            )
                        else:
                            stream.seek(r.offset)
                            left = r.nbytes
                            while left:
                                part = stream.read(min(left, 8 << 20))
                                if not part:
                                    raise OSError("short source pack read")
                                out.write(part)
                                left -= len(part)
                            table[layer][name] = replace(r, offset=offset)
            result = replace(
                m,
                payload_path=payload,
                payload_nbytes=temporary.stat().st_size,
                tensors=table,
                scale_encoding=encodings,
            )
            result.validate_cutlass_layout()
            temporary.replace(payload)
            write_manifest(result, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    native = sum(m.tensors[0][n].nbytes for n in REQUIRED_TENSORS) // m.num_experts
    compressed = (
        sum(result.tensors[0][n].nbytes for n in REQUIRED_TENSORS) // m.num_experts
    )
    return dict(
        native_slot_bytes=native,
        compressed_slot_bytes=compressed,
        capacity_ratio=native / compressed,
        scale_encoding=encodings,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--payload", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    print(json.dumps(compress_pack(a.source, a.payload, a.manifest), indent=2))


if __name__ == "__main__":
    main()
