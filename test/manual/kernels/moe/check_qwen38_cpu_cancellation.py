# SPDX-License-Identifier: Apache-2.0
"""Native opposite-expert cancellation regression for FP32 result ABI."""

import argparse
import ctypes as C
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--core", type=int, default=0)
    args = parser.parse_args()
    os.nice(15)
    from build_qwen38_cpu_hybrid import check_temperature
    from check_qwen38_cpu_hybrid import bf, f32, fixture, unbf

    check_temperature()
    lib = C.CDLL(args.library)
    assert lib.qwen38_cpu_abi() == 2
    lib.qwen38_cpu_run.argtypes = [
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
    h, width = 64, 32
    payload, offsets, _, x = fixture(h, width)
    strides = [
        2 * width * h // 2,
        h * width // 2,
        ((2 * width + 127) // 128 * 128) * ((h // 16 + 3) // 4 * 4),
        ((h + 127) // 128 * 128) * ((width // 16 + 3) // 4 * 4),
        4,
        4,
    ]
    for k, stride in enumerate(strides):
        first = payload[offsets[k] : offsets[k] + stride]
        payload[offsets[k] + stride : offsets[k] + 2 * stride] = (
            bytes(v ^ 0x88 for v in first) if k == 1 else first
        )
    buf = (C.c_uint8 * len(payload)).from_buffer(payload)
    off = (C.c_uint64 * 8)(*offsets)
    inp = (C.c_uint16 * h)(*x[:h])
    weights = (C.c_float * 2)(0.5, 0.499)
    cores = (C.c_int * 1)(args.core)

    def run(routes):
        check_temperature()
        ids = (C.c_int32 * 2)(*routes)
        out = (C.c_float * h)()
        status = lib.qwen38_cpu_run(
            buf,
            len(payload),
            off,
            3,
            h,
            width,
            inp,
            ids,
            weights,
            1,
            2,
            1,
            cores,
            1,
            out,
        )
        assert status == 0, lib.qwen38_cpu_error()
        time.sleep(0.2)
        return list(out)

    whole, a, b = run([0, 1]), run([0, -1]), run([-1, 1])
    once = [unbf(bf(f32(u + v))) for u, v in zip(a, b)]
    expected = [unbf(bf(v)) for v in whole]
    assert once == expected
    legacy = [unbf(bf(unbf(bf(u)) + unbf(bf(v)))) for u, v in zip(a, b)]
    lost = sum(e != 0 and old == 0 for e, old in zip(expected, legacy))
    assert lost > 0, "fixture failed to exercise cancellation"
    print(
        f"PASS FP32 split==combined after single BF16 cast; legacy lost {lost}/{h} coordinates; coordinate62={expected[62]}"
    )
    lib.qwen38_cpu_release()


if __name__ == "__main__":
    main()
