# SPDX-License-Identifier: Apache-2.0
"""Paired native CPU benchmark with identical routes, weights and FP32 outputs.

Uses the serving H/I and 1..24 miss experts, including ragged M=3 routes.
This measures native execution only, not CPU/GPU overlap or serving throughput.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import random
import statistics
import struct
import time
from pathlib import Path

from build_qwen38_cpu_hybrid import check_temperature
from check_qwen38_cpu_hybrid import bf


def fixture(experts, seed):
    h, width = 2560, 640
    rng = random.Random(seed)
    chunks = [
        rng.randbytes(experts * width * h),
        rng.randbytes(experts * h * width // 2),
    ]
    for rows, cols in ((2 * width, h), (h, width)):
        size = experts * ((rows + 127) // 128 * 128) * ((cols // 16 + 3) // 4 * 4)
        # Varied finite FP8 scales, without Python per-byte fixture overhead.
        choices = (24, 32, 40, 48)
        chunks.append(
            rng.randbytes(size).translate(bytes(choices[k % 4] for k in range(256)))
        )
    chunks += [
        struct.pack(f"{experts}f", *(rng.uniform(0.7, 1.3) for _ in range(experts))),
        struct.pack(f"{experts}f", *(rng.uniform(0.7, 1.3) for _ in range(experts))),
        struct.pack("f", 0.83),
        struct.pack("f", 1.23),
    ]
    payload, offsets = bytearray(), []
    for chunk in chunks:
        offsets.append(len(payload))
        payload.extend(chunk)
    x = (C.c_uint16 * (3 * h))(*(bf(rng.uniform(-0.05, 0.05)) for _ in range(3 * h)))
    return payload, (C.c_uint64 * 8)(*offsets), x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--backend", choices=("b12x", "triton"), default="b12x")
    args = parser.parse_args()
    cores = tuple(map(int, args.cores.split(",")))
    if not 1 <= len(cores) <= 4 or not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("choose 1..4 available physical cores")
    siblings = [
        Path(
            f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        ).read_text()
        for c in cores
    ]
    if len(set(siblings)) != len(cores) or args.samples < 3:
        raise ValueError("distinct physical cores and at least three samples required")
    os.nice(15)
    libraries, identity = [], []
    symbol = "qwen38_cpu_run_b12x" if args.backend == "b12x" else "qwen38_cpu_run"
    for path in (args.baseline_library, args.candidate_library):
        lib = C.CDLL(str(path.resolve()))
        if lib.qwen38_cpu_abi() != 2:
            raise ValueError("expected FP32 ABI 2")
        fn = getattr(lib, symbol)
        fn.argtypes = [
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
        libraries.append((lib, fn))
        identity.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = {
        "libraries": identity,
        "backend": args.backend,
        "cores": cores,
        "scope": "native only; synthetic weights; no GPU overlap",
        "cases": [],
    }
    rng = random.Random(42)
    try:
        payload, offsets, x = fixture(24, 42)
        buf = (C.c_uint8 * len(payload)).from_buffer(payload)
        affinity = (C.c_int * len(cores))(*cores)
        cases = [(1, e, "shared") for e in (1, 3, 8)]
        cases += [(3, e, "shared") for e in (1, 2, 3, 4, 6, 8)]
        cases += [(3, e, "ragged") for e in (3, 8, 12, 16, 24)]
        for nt, experts, routing in cases:
            rows = (
                [list(range(experts)) for _ in range(nt)]
                if routing == "shared"
                else [list(range(t, experts, nt)) for t in range(nt)]
            )
            # Include duplicates, holes, and unequal active token counts.
            if routing == "ragged":
                rows[0] += [0, -1]
            topk = max(map(len, rows))
            ids = (C.c_int32 * (nt * topk))(
                *(e for row in rows for e in row + [-1] * (topk - len(row)))
            )
            weights = (C.c_float * (nt * topk))(*(rng.uniform(-0.2, 0.2) for _ in ids))
            outs = [(C.c_float * (nt * 2560))() for _ in libraries]
            samples = [[], []]
            for iteration in range(args.samples + 2):
                order = [0, 1]
                rng.shuffle(order)
                for index in order:
                    check_temperature()
                    lib, fn = libraries[index]
                    start = time.perf_counter_ns()
                    rc = fn(
                        buf,
                        len(payload),
                        offsets,
                        24,
                        2560,
                        640,
                        x,
                        ids,
                        weights,
                        nt,
                        topk,
                        len(cores),
                        affinity,
                        len(cores),
                        outs[index],
                    )
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    if rc:
                        raise RuntimeError(lib.qwen38_cpu_error().decode())
                    if iteration >= 2:
                        samples[index].append(elapsed)
                    time.sleep(0.2)
                if bytes(outs[0]) != bytes(outs[1]):
                    raise AssertionError(f"FP32 mismatch: M={nt} E={experts} {routing}")
            medians = [statistics.median(v) for v in samples]
            row = {
                "tokens": nt,
                "experts": experts,
                "routing": routing,
                "median_us": medians,
                "speedup": medians[0] / medians[1],
                "samples_us": samples,
                "exact_fp32_match": True,
            }
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                flush=True,
            )
    finally:
        for lib, _ in libraries:
            lib.qwen38_cpu_release()


if __name__ == "__main__":
    main()
