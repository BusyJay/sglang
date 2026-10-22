# SPDX-License-Identifier: Apache-2.0
"""Compare the native layered batch service with the synchronous CPU ABI."""

import argparse
import ctypes as c
import importlib.util
import os
import threading
from pathlib import Path


def load_fixture(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("qwen38_cpu_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--second-core", type=int)
    parser.add_argument("--cores", type=str)
    parser.add_argument("--backend", choices=("b12x", "triton"), default="b12x")
    parser.add_argument("--csf", action="store_true")
    args = parser.parse_args()
    selected = (
        tuple(int(core) for core in args.cores.split(","))
        if args.cores else
        ((args.core,) if args.second_core is None else (args.core, args.second_core))
    )
    if len(set(selected)) != len(selected) or not set(selected) <= os.sched_getaffinity(0):
        raise ValueError("cores must be distinct and available")
    fixture = load_fixture(
        "check_qwen38_cpu_csf.py" if args.csf else "check_qwen38_cpu_hybrid.py"
    ).fixture
    lib = c.CDLL(str(args.library.resolve()))
    lib.qwen38_cpu_error.restype = c.c_char_p
    direct = (
        lib.qwen38_cpu_run_csf if args.csf else
        lib.qwen38_cpu_run_b12x if args.backend == "b12x" else
        lib.qwen38_cpu_run
    )
    direct.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_int, c.c_int,
                       c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_int,
                       c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_void_p]
    if args.csf:
        direct.argtypes += [c.c_void_p, c.c_int]
    service = lib.qwen38_cpu_layered_service
    service.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_void_p,
                        c.c_int, c.c_int, c.c_int, c.c_int, c.c_int,
                        c.c_int, c.c_void_p, c.c_void_p, c.c_void_p,
                        c.c_void_p, c.c_int]
    lib.qwen38_cpu_layered_stop.argtypes = [c.c_void_p]
    core = (c.c_int * len(selected))(*selected)
    stop = c.c_int32(0)
    failed = c.c_int32(0)
    for h, width in ((64, 32), (160, 160)):
        if args.csf:
            _, _, payload, offsets, strides, values = fixture(h, width)
            experts = 4
        else:
            payload, offsets, _, values = fixture(
                h, width, post_scale=args.backend == "b12x"
            )
            strides = None
            experts = 3
        buf = (c.c_uint8 * len(payload)).from_buffer(payload)
        off = (c.c_uint64 * 8)(*offsets)
        topk = 4
        # head/layer/tokens + fixed 3-token payload and result layout.
        page = (c.c_uint8 * (12 + 3 * h * 2 + 3 * topk * 8 + 3 * h * 4 + 4))()
        head = c.cast(page, c.POINTER(c.c_int32))
        x = (c.c_uint16 * (3 * h)).from_buffer(page, 12)
        x[:] = values
        ids = (c.c_int32 * (3 * topk)).from_buffer(page, 12 + 3 * h * 2)
        weights = (c.c_float * (3 * topk)).from_buffer(page, 12 + 3 * h * 2 + 3 * topk * 4)
        result = (c.c_float * (3 * h)).from_buffer(page, 12 + 3 * h * 2 + 3 * topk * 8)
        done = (c.c_int32 * 1).from_buffer(page, len(page) - 4)
        stop.value = failed.value = 0
        error = []

        def serve():
            rc = service(buf, len(payload), off, strides, 1, experts, h, width,
                         topk, len(selected), core, page, c.byref(failed), c.byref(stop),
                         int(args.backend == "b12x"))
            if rc:
                error.append(lib.qwen38_cpu_error().decode())

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            route_cases = (
                [[0, 1, 0, -1], [1, 2, -1, -1], [2, 0, 1, -1]],
                [[2, -1, -1, -1]],
                [[1, 2, -1, -1], [0, 2, 0, -1]],
            )
            for sequence in range(1, 31):
                routes = route_cases[(sequence - 1) % len(route_cases)]
                token_count = len(routes)
                for index in range(3 * topk):
                    ids[index] = routes[index // topk][index % topk] if index < token_count * topk else -1
                    weights[index] = 0.125 + (index % 3) * 0.03125
                expected = (c.c_float * (token_count * h))()
                direct_args = (buf, len(payload), off, experts, h, width, x,
                               ids, weights, token_count, topk, len(selected),
                               core, len(selected), expected)
                if args.csf:
                    direct_args += (strides, int(args.backend == "b12x"))
                rc = direct(*direct_args)
                assert rc == 0, lib.qwen38_cpu_error()
                head[1], head[2], head[0] = 0, token_count, sequence
                for _ in range(5000):
                    if done[0] == sequence or error or failed.value:
                        break
                    threading.Event().wait(0.001)
                assert done[0] == sequence, (error, failed.value)
                assert bytes(result)[: token_count * h * 4] == bytes(expected)
            ids[0] = experts
            head[1], head[2], head[0] = 0, 1, 31
            for _ in range(5000):
                if failed.value:
                    break
                threading.Event().wait(0.001)
            assert failed.value == 1 and done[0] == 30
        finally:
            lib.qwen38_cpu_layered_stop(c.byref(stop))
            thread.join(timeout=5)
            assert not thread.is_alive(), "service did not stop"
            lib.qwen38_cpu_release()
    print(f"PASS layered service matches synchronous {args.backend}"
          f" CSF={args.csf} for reused pages")


if __name__ == "__main__":
    main()
