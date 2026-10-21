# SPDX-License-Identifier: Apache-2.0
"""Check the production bitmap CSF ABI against identical native scale bytes.

Uses only the standard library; run on an AVX512-BF16 CPU after building the
library with build_qwen38_cpu_hybrid.py. Protects tile partitions, projection
barriers, masked/duplicate routes, native-plane fallback, and malformed metadata.
"""

import argparse
import ctypes as C
import os
import random
import struct
from pathlib import Path

from build_qwen38_cpu_hybrid import check_temperature
from check_qwen38_cpu_hybrid import bf

COMMON = [
    C.c_void_p,
    C.c_uint64,
    C.c_void_p,
    C.c_int,
    C.c_int,
    C.c_int,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
    C.c_int,
    C.c_int,
    C.c_int,
    C.c_void_p,
    C.c_int,
    C.c_void_p,
]


def scale_plane(rows, columns, rate, rng):
    """Construct native words and their bitmap representation independently."""
    raw, fixed, metadata, records = bytearray(), bytearray(), bytearray(), bytearray()
    count = 0
    for _ in range(rows // 128):
        bases = bytes(rng.randrange(20, 40) for _ in range(128))
        fixed.extend(bases)
        for _ in range(columns // 4):
            masks = [0] * 4
            start = count
            for lane in range(128):
                codes = [rng.randrange(16) for _ in range(4)]
                regular = [bases[lane] + code for code in codes]
                values = [
                    rng.randrange(64, 72) if rng.random() < rate else value
                    for value in regular
                ]
                fixed.extend(
                    struct.pack(
                        "<H", sum(code << (4 * k) for k, code in enumerate(codes))
                    )
                )
                raw.extend(values)
                if values != regular:
                    masks[lane // 32] |= 1 << (lane % 32)
                    records.extend(values)
                    count += 1
            prefixes, prefix = [], 0
            for mask in masks:
                prefixes.append(prefix)
                prefix += mask.bit_count()
            metadata.extend(
                struct.pack(
                    "<8I",
                    start,
                    sum(v << (8 * k) for k, v in enumerate(prefixes)),
                    *masks,
                    count,
                    0,
                )
            )
    header = bytearray(1024)
    struct.pack_into("<I", header, 668, 1)
    struct.pack_into("<2I", header, 1008, count, 0x32465343)
    slab = header + fixed + metadata + records
    slab.extend(bytes(-len(slab) % 16))
    return raw, slab


def assemble(chunks):
    payload, offsets = bytearray(), []
    for chunk in chunks:
        offsets.append(len(payload))
        payload.extend(chunk)
    return payload, (C.c_uint64 * 8)(*offsets)


def fixture(h, width, experts=4, rate=0.01, compression=(True, True), seed=42):
    rng = random.Random(seed)
    chunks = [
        rng.randbytes(experts * width * h),
        rng.randbytes(experts * h * width // 2),
    ]
    compressed, strides = [], []
    for enabled, (rows, columns) in zip(compression, ((2 * width, h), (h, width))):
        rows, columns = (rows + 127) // 128 * 128, (columns // 16 + 3) // 4 * 4
        planes = [scale_plane(rows, columns, rate, rng) for _ in range(experts)]
        native = b"".join(raw for raw, _ in planes)
        stride = max(len(slab) for _, slab in planes) if enabled else 0
        compressed.append(
            b"".join(slab + bytes(stride - len(slab)) for _, slab in planes)
            if enabled
            else native
        )
        chunks.append(native)
        strides.append(stride)
    chunks += [
        struct.pack(f"{experts}f", *(rng.uniform(0.7, 1.3) for _ in range(experts)))
        for _ in range(2)
    ]
    chunks += [struct.pack("f", 0.83), struct.pack("f", 1.23)]
    native, native_offsets = assemble(chunks)
    packed, packed_offsets = assemble(chunks[:2] + compressed + chunks[4:])
    x = (C.c_uint16 * (3 * h))(*(bf(rng.uniform(-0.005, 0.005)) for _ in range(3 * h)))
    return native, native_offsets, packed, packed_offsets, (C.c_uint64 * 2)(*strides), x


def physical_cores(value):
    cores = tuple(map(int, value.split(",")))
    if not 1 <= len(cores) <= 4 or not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("choose 1..4 available physical cores")
    siblings = [
        Path(
            f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        ).read_text()
        for c in cores
    ]
    if len(set(siblings)) != len(cores):
        raise ValueError("cores must have distinct physical siblings")
    return cores


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path.resolve()))
        if self.lib.qwen38_cpu_abi() != 2 or self.lib.qwen38_cpu_csf_abi() != 2:
            raise ValueError("expected production CPU and CSF ABI 2")
        self.lib.qwen38_cpu_error.restype = C.c_char_p
        for name in ("qwen38_cpu_run", "qwen38_cpu_run_b12x"):
            getattr(self.lib, name).argtypes = COMMON
        self.lib.qwen38_cpu_run_csf.argtypes = COMMON + [C.c_void_p, C.c_int]

    def call(self, data, h, width, rows, cores, backend, compressed=True):
        native, no, packed, po, strides, x = data
        payload, offsets = (packed, po) if compressed else (native, no)
        buf = (C.c_uint8 * len(payload)).from_buffer(payload)
        topk = max(map(len, rows))
        ids = (C.c_int32 * (len(rows) * topk))(
            *(e for row in rows for e in row + [-1] * (topk - len(row)))
        )
        weights = (C.c_float * len(ids))(
            *(0.125 + (k % 3) * 0.03125 for k in range(len(ids)))
        )
        out = (C.c_float * (len(rows) * h))()
        affinity = (C.c_int * len(cores))(*cores)
        fn = (
            self.lib.qwen38_cpu_run_csf
            if compressed
            else (
                self.lib.qwen38_cpu_run_b12x
                if backend == "b12x"
                else self.lib.qwen38_cpu_run
            )
        )
        args = [
            buf,
            len(payload),
            offsets,
            4,
            h,
            width,
            x,
            ids,
            weights,
            len(rows),
            topk,
            len(cores),
            affinity,
            len(cores),
            out,
        ]
        if compressed:
            args += [strides, int(backend == "b12x")]
        return fn, args, out

    def release(self):
        self.lib.qwen38_cpu_release()


def checked(fn, args, lib):
    if fn(*args):
        raise AssertionError(lib.lib.qwen38_cpu_error().decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    args = parser.parse_args()
    cores = physical_cores(args.cores)
    lib = Library(args.library)
    checks = 0
    try:
        for h, width, rate in ((64, 32, 0.0), (160, 160, 0.15), (2560, 640, 0.01)):
            for compression in (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ):
                data = fixture(h, width, rate=rate, compression=compression)
                for backend in ("b12x", "triton"):
                    for nt in (1, 2, 3):
                        routes = [[0, 1, 0, -1], [1, 2, -1, -1], [2, 0, 3, -1]][:nt]
                        fn, params, expected = lib.call(
                            data, h, width, routes, cores[:1], backend, False
                        )
                        checked(fn, params, lib)
                        for count in range(1, len(cores) + 1):
                            check_temperature()
                            fn, params, actual = lib.call(
                                data, h, width, routes, cores[:count], backend
                            )
                            checked(fn, params, lib)
                            assert bytes(actual) == bytes(expected), (
                                h,
                                width,
                                compression,
                                backend,
                                nt,
                                count,
                            )
                            # Repeat with the same pool to exercise reusable barriers/scratch.
                            checked(fn, params, lib)
                            assert bytes(actual) == bytes(expected)
                            checks += 1
        data = fixture(160, 160)
        for rows in ([[0, 1, 2, 3]] * 3, [[-1]] * 3, [[0], [-1], [-1]]):
            fn, params, actual = lib.call(data, 160, 160, rows, cores, "b12x")
            checked(fn, params, lib)
            if rows == [[-1]] * 3:
                assert not any(actual)
            checks += 1
        # Invalid extent/rank must reject before workers dereference exceptions.
        packed, offsets = data[2:4]
        meta = offsets[2] + 1024 + 384 * (1 + 12 // 2)
        for pos, value in (
            (offsets[2] + 1012, 0),
            (meta, 1),
            (meta + 4, 1),
            (meta + 24, 0xFFFFFFFF),
        ):
            original = packed[pos : pos + 4]
            struct.pack_into("<I", packed, pos, value)
            fn, params, _ = lib.call(data, 160, 160, [[0]] * 3, cores, "b12x")
            assert fn(*params) != 0, ("accepted malformed CSF", pos)
            packed[pos : pos + 4] = original
            checked(fn, params, lib)
            checks += 1
        print(
            f"PASS: {checks} native/CSF partition, fallback, routing and malformed-input checks"
        )
    finally:
        lib.release()


if __name__ == "__main__":
    main()
