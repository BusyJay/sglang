# SPDX-License-Identifier: Apache-2.0
"""Independent scalar oracle; tiny synthetic CUTLASS tensors, paced calls."""

import argparse
import ctypes as C
import importlib.util
import math
import os
import random
import struct
import time
from pathlib import Path

VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    if bits & 0x7FFFFFFF > 0x7F800000:
        return (bits >> 16) | 64
    return ((bits + 32767 + ((bits >> 16) & 1)) >> 16) & 65535


def unbf(x):
    return struct.unpack("f", struct.pack("I", x << 16))[0]


def e4m3(x):
    sign = -1 if x & 128 else 1
    x &= 127
    if x == 127:
        return math.nan
    return sign * (x / 512 if x < 8 else math.ldexp(1 + (x & 7) / 8, (x >> 3) - 7))


def offset(row, col, columns):
    # Independent inverse of CUTLASS's [row/128, col/4, row%32, row/32, col%4].
    groups = (columns + 3) // 4
    return (
        (row // 128 * groups + col // 4) * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + col % 4
    )


def fixture(h=64, width=32, experts=3, *, post_scale=False):
    rng = random.Random(731)
    chunks, decoded, offsets = [], [], []
    for rows, cols in ((2 * width, h), (h, width)):
        packed, scales, matrices = bytearray(), bytearray(), []
        scale_stride = (rows + 127) // 128 * 128 * ((cols // 16 + 3) // 4 * 4)
        for expert in range(experts):
            codes = [[rng.randrange(16) for _ in range(cols)] for _ in range(rows)]
            scale = bytearray(scale_stride)
            for row in range(rows):
                for col in range(cols // 16):
                    scale[offset(row, col, cols // 16)] = rng.choice(
                        (0, 1, 24, 32, 40, 48, 56, 64, 152)
                    )
                packed.extend(
                    codes[row][k] | codes[row][k + 1] << 4 for k in range(0, cols, 2)
                )
            scales.extend(scale)
            matrices.append(
                [
                    [
                        VALUES[codes[row][k]]
                        * e4m3(scale[offset(row, k // 16, cols // 16)])
                        for k in range(cols)
                    ]
                    for row in range(rows)
                ]
            )
        chunks.append((packed, scales))
        decoded.append(matrices)
    g1 = (0.73, 1.17, 0.91)
    g2 = (1.31, 0.69, 1.07)
    q1, q2 = f32(0.83), f32(1.23)
    components = [
        chunks[0][0],
        chunks[1][0],
        chunks[0][1],
        chunks[1][1],
        struct.pack("3f", *g1),
        struct.pack("3f", *g2),
        struct.pack("f", q1),
        struct.pack("f", q2),
    ]
    payload = bytearray()
    for component in components:
        offsets.append(len(payload))
        payload.extend(component)
    for group, factors, q in zip(decoded, (g1, g2), (q1, q2)):
        for expert, matrix in enumerate(group):
            global_scale = f32(f32(factors[expert]) * q)
            for row in range(len(matrix)):
                if not post_scale:
                    matrix[row] = [unbf(bf(f32(w * global_scale))) for w in matrix[row]]
    x = [bf(rng.uniform(-0.5, 0.5)) for _ in range(3 * h)]
    return payload, offsets, decoded, x


def reference(decoded, x, ids, weights, h, width, projection_scales=None):
    output = []
    for token, routes in enumerate(ids):
        total = [0.0] * h
        for expert in sorted(set(routes) - {-1}):
            gate_up = [
                sum(w * unbf(a) for w, a in zip(row, x[token * h : (token + 1) * h]))
                for row in decoded[0][expert]
            ]
            if projection_scales is not None:
                gate_up = [f32(f32(v) * projection_scales[0][expert]) for v in gate_up]
            hidden = [
                unbf(bf(f32((g / (1 + math.exp(-g))) * u)))
                for u, g in zip(gate_up[:width], gate_up[width:])
            ]
            route_weight = f32(
                sum(w for e, w in zip(routes, weights[token]) if e == expert)
            )
            down = [
                sum(w * a for w, a in zip(row, hidden)) for row in decoded[1][expert]
            ]
            if projection_scales is not None:
                down = [f32(f32(v) * projection_scales[1][expert]) for v in down]
            total = [f32(t + f32(route_weight * f32(v))) for t, v in zip(total, down)]
        output.extend(total)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    parser.add_argument("--backend", choices=("triton", "b12x"), default="triton")
    args = parser.parse_args()
    cores = tuple(map(int, args.cores.split(",")))
    if not 1 <= len(cores) <= 4 or len(set(cores)) != len(cores):
        raise ValueError("choose 1..4 distinct physical cores")
    if not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("cores outside process affinity")
    sibling_sets = [
        Path(
            f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        ).read_text()
        for c in cores
    ]
    if len(set(sibling_sets)) != len(cores):
        raise ValueError("cores share physical siblings")
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["GOMP_SPINCOUNT"] = "0"
    os.nice(15)
    spec = importlib.util.spec_from_file_location(
        "build_guard", Path(__file__).with_name("build_qwen38_cpu_hybrid.py")
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    lib = C.CDLL(str(args.library.resolve()))
    native_run = (
        lib.qwen38_cpu_run_b12x if args.backend == "b12x" else lib.qwen38_cpu_run
    )
    native_run.argtypes = [
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
    lib.qwen38_cpu_error.restype = C.c_char_p
    checks, maximum, durations = 0, 0.0, []
    for h, width in ((64, 32), (160, 160)):
        payload, offsets, decoded, x = fixture(
            h, width, post_scale=args.backend == "b12x"
        )
        projection_scales = None
        if args.backend == "b12x":
            projection_scales = []
            for global_offset, input_offset in zip(offsets[4:6], offsets[6:8]):
                factors = struct.unpack_from("3f", payload, global_offset)
                q = struct.unpack_from("f", payload, input_offset)[0]
                projection_scales.append([f32(v * q) for v in factors])
        buf = (C.c_uint8 * len(payload)).from_buffer(payload)
        off = (C.c_uint64 * 8)(*offsets)
        inp = (C.c_uint16 * len(x))(*x)
        for nt in (1, 2, 3):
            routes = [[0, 1, 0, -1], [1, 2, -1, -1], [2, 0, 1, -1]][:nt]
            weights = [
                [f32(0.13), f32(0.29), f32(0.07), 0.0],
                [f32(0.37), f32(0.19), 0.0, 0.0],
                [f32(0.21), f32(0.31), f32(0.17), 0.0],
            ][:nt]
            ids = (C.c_int32 * (nt * 4))(*(e for row in routes for e in row))
            rw = (C.c_float * (nt * 4))(*(w for row in weights for w in row))
            expected = reference(
                decoded, x, routes, weights, h, width, projection_scales
            )
            first = None
            for threads in sorted({1, min(2, len(cores)), len(cores)}):
                guard.check_temperature()
                out = (C.c_float * (nt * h))()
                affinity = (C.c_int * threads)(*cores[:threads])
                start = time.perf_counter()
                status = native_run(
                    buf,
                    len(payload),
                    off,
                    3,
                    h,
                    width,
                    inp,
                    ids,
                    rw,
                    nt,
                    4,
                    threads,
                    affinity,
                    threads,
                    out,
                )
                durations.append((time.perf_counter() - start) * 1000)
                if status:
                    raise AssertionError(lib.qwen38_cpu_error().decode())
                values = list(out)
                error = max(abs(a - b) for a, b in zip(values, expected))
                maximum = max(maximum, error)
                assert all(
                    abs(a - b) <= 0.002 + 0.02 * abs(b)
                    for a, b in zip(values, expected)
                ), (h, nt, threads, error)
                if first is None:
                    first = list(out)
                else:
                    assert list(out) == first, "thread count changed results"
                checks += 1
                time.sleep(0.2)
            # Every masked route must produce zero and an invalid ID must reject.
            for k in range(nt * 4):
                ids[k] = -1
            assert (
                native_run(
                    buf,
                    len(payload),
                    off,
                    3,
                    h,
                    width,
                    inp,
                    ids,
                    rw,
                    nt,
                    4,
                    1,
                    affinity,
                    1,
                    out,
                )
                == 0
            )
            assert not any(out)
            time.sleep(0.2)
            guard.check_temperature()
            ids[0] = 3
            assert (
                native_run(
                    buf,
                    len(payload),
                    off,
                    3,
                    h,
                    width,
                    inp,
                    ids,
                    rw,
                    nt,
                    4,
                    1,
                    affinity,
                    1,
                    out,
                )
                != 0
            )
            checks += 2
            time.sleep(0.2)
    print(
        f"PASS: {checks} scalar/partition/error checks; max_abs_error={maximum:.8g}; "
        f"longest_native_call_ms={max(durations):.3f}; temperature={guard.check_temperature():.1f} C"
    )


if __name__ == "__main__":
    main()
