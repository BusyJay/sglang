# SPDX-License-Identifier: Apache-2.0
"""Paired synthetic CPU CSF decode-plus-expert benchmark; no serving claim.

The experiment library preserves kernel.cpp arithmetic and adds native-order
four-bit scale codes, row bases, and uint32 exact exceptions. Each call expands
every selected expert again into one 300 KiB scratch, without an expanded cache.
This tests a proposed CPU runtime layout, not B12X's compact block decoder.
"""

import argparse
import ctypes as C
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

from build_qwen38_cpu_hybrid import check_temperature


class Plane(C.Structure):
    _fields_ = [
        ("bases", C.c_void_p),
        ("codes", C.c_void_p),
        ("exceptions", C.c_void_p),
        ("count", C.c_uint64),
        ("rows", C.c_int),
        ("columns", C.c_int),
    ]


def native(logical):
    rows, columns = logical.shape
    return np.ascontiguousarray(
        logical.reshape(rows // 128, 4, 32, columns // 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .reshape(-1)
    )


def fixture(experts, exception_rate, seed):
    rng = np.random.default_rng(seed)
    owners, planes, raw = [], [], [[], []]
    logical_bytes = compressed_bytes = exception_count = 0
    for expert in range(experts):
        for projection, (rows, columns) in enumerate(((1280, 160), (2560, 40))):
            bases = rng.integers(20, 29, rows, dtype=np.uint8)
            codes = rng.integers(0, 16, (rows, columns), dtype=np.uint8)
            logical = bases[:, None] + codes
            mask = rng.random((rows, columns)) < exception_rate
            logical[mask] = rng.integers(64, 72, int(mask.sum()), dtype=np.uint8)
            original = native(logical)
            nc = native(codes)
            packed = np.ascontiguousarray(nc[::2] | nc[1::2] << 4)
            nm = native(mask)
            positions = np.flatnonzero(nm).astype(np.uint32)
            exceptions = np.ascontiguousarray(
                positions | original[positions].astype(np.uint32) << 24
            )
            owners.extend((bases, packed, exceptions))
            planes.append(
                Plane(bases.ctypes.data, packed.ctypes.data, exceptions.ctypes.data,
                      len(exceptions), rows, columns)
            )
            raw[projection].append(original)
            logical_bytes += original.nbytes
            compressed_bytes += bases.nbytes + packed.nbytes + exceptions.nbytes
            exception_count += len(exceptions)
    chunks = [
        rng.integers(0, 256, experts * 2560 * 640, dtype=np.uint8),
        rng.integers(0, 256, experts * 2560 * 640 // 2, dtype=np.uint8),
        np.concatenate(raw[0]), np.concatenate(raw[1]),
        rng.uniform(.7, 1.3, experts).astype(np.float32).view(np.uint8),
        rng.uniform(.7, 1.3, experts).astype(np.float32).view(np.uint8),
        np.array([.83], dtype=np.float32).view(np.uint8),
        np.array([1.23], dtype=np.float32).view(np.uint8),
    ]
    offsets = np.array([0, *np.cumsum([v.nbytes for v in chunks])[:-1]], dtype=np.uint64)
    baseline = np.concatenate(chunks)
    # Candidate retains weights/alphas only. Dummy scale offsets satisfy the
    # unchanged baseline ABI range checks; candidate math never reads them.
    cc = [chunks[i] for i in (0, 1, 4, 5, 6, 7)]
    co = np.array([0, *np.cumsum([v.nbytes for v in cc])[:-1]], dtype=np.uint64)
    candidate_offsets = co[[0, 1, 0, 0, 2, 3, 4, 5]].copy()
    candidate = np.concatenate(cc)
    x = rng.uniform(-.005, .005, (3, 2560)).astype(np.float32)
    bits = x.view(np.uint32)
    x = ((bits + 32767 + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (baseline, offsets, candidate, candidate_offsets, x,
            (Plane * len(planes))(*planes), owners, raw,
            dict(scale_bytes=logical_bytes, compressed_bytes=compressed_bytes,
                 exception_count=exception_count,
                 realized_exception_rate=exception_count / logical_bytes,
                 compression_ratio=compressed_bytes / logical_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--cores", default="0,1,2,3")
    args = parser.parse_args()
    cores = tuple(map(int, args.cores.split(",")))
    if not 1 <= len(cores) <= 4 or not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("choose 1..4 available CPU cores")
    siblings = [Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
                .read_text().strip() for c in cores]
    if len(set(siblings)) != len(cores) or args.samples < 5:
        raise ValueError("distinct physical cores and at least five samples required")
    os.nice(15)
    affinity = (C.c_int * len(cores))(*cores)
    lib = C.CDLL(str(args.library.resolve()))
    common = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_int, C.c_int, C.c_int,
              C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int,
              C.c_void_p, C.c_int, C.c_void_p]
    lib.qwen38_cpu_run_b12x.argtypes = common
    lib.qwen38_cpu_run_csf.argtypes = common + [C.POINTER(Plane), C.c_int]
    lib.qwen38_cpu_error.restype = C.c_char_p
    lib.csf_expand_check.argtypes = [C.POINTER(Plane), C.c_void_p]
    lib.csf_decode_only.argtypes = [C.POINTER(Plane), C.c_int, C.c_int,
                                   C.c_void_p, C.c_int, C.c_int]
    lib.csf_decode_only.restype = C.c_double
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo")
               .read_text().splitlines() if line.startswith("model name"))
    result = dict(cpu=cpu, cores=cores, samples=args.samples,
                  scope="synthetic H=2560/I=640; native CPU only; custom native-order slab CSF",
                  paths=["native", "csf_serial", "csf_parallel", "native_copy_parallel"], cases=[])
    rng = random.Random(421)
    cases = [(1, e, "shared") for e in (1, 3, 8)]
    cases += [(3, e, "shared") for e in (1, 2, 4, 8)]
    cases += [(3, e, "ragged") for e in (8, 12, 16, 24)]
    for rate in (0., .01, .05, .15):
        base, bo, compressed, co, x, planes, owners, raw, sizes = fixture(24, rate, 42)
        for e in range(24):
            for p in range(2):
                restored = np.empty_like(raw[p][e])
                lib.csf_expand_check(C.byref(planes[2 * e + p]), restored.ctypes.data)
                if not np.array_equal(restored, raw[p][e]):
                    raise AssertionError("scale bytes differ")
        for nt, count, routing in cases:
            rows = ([list(range(count)) for _ in range(nt)] if routing == "shared"
                    else [list(range(t, count, nt)) for t in range(nt)])
            topk = 8
            ids = np.array([row + [-1] * (topk-len(row)) for row in rows], dtype=np.int32)
            route_weights = np.full((nt, topk), 1/8, dtype=np.float32)
            outputs = [np.empty((nt, 2560), dtype=np.float32) for _ in range(4)]
            samples = [[], [], [], []]
            def call(index):
                payload, offsets = (base, bo) if index in (0, 3) else (compressed, co)
                params = (payload.ctypes.data, payload.nbytes, offsets.ctypes.data,
                          24, 2560, 640, x.ctypes.data, ids.ctypes.data,
                          route_weights.ctypes.data, nt, topk, len(cores), affinity,
                          len(cores), outputs[index].ctypes.data)
                rc = (lib.qwen38_cpu_run_b12x(*params) if index == 0 else
                      lib.qwen38_cpu_run_csf(*params, planes, index - 1))
                if rc:
                    raise RuntimeError(lib.qwen38_cpu_error().decode())
            for iteration in range(args.samples + 4):
                order = [0, 1, 2, 3]
                rng.shuffle(order)
                for index in order:
                    check_temperature()
                    start = time.perf_counter_ns()
                    call(index)
                    elapsed = (time.perf_counter_ns()-start)/1000
                    if iteration >= 4:
                        samples[index].append(elapsed)
                    time.sleep(.015)
                if not all(outputs[0].tobytes() == out.tobytes() for out in outputs[1:]):
                    raise AssertionError(f"FP32 mismatch M={nt} E={count}")
            medians = [statistics.median(v) for v in samples]
            decode = [lib.csf_decode_only(planes, count, len(cores), affinity, p, 10)
                      for p in (0, 1)]
            row = dict(tokens=nt, experts=count, routing=routing, exception_rate=rate,
                       **sizes, median_us=medians,
                       slowdown_percent=[100*(v/medians[0]-1) for v in medians[1:]],
                       samples_us=samples, decode_only_us=decode,
                       exact_fp32_match=True, exact_scale_match=True)
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({k: row[k] for k in ("tokens", "experts", "routing",
                             "exception_rate", "median_us", "slowdown_percent",
                             "decode_only_us")}), flush=True)
        lib.qwen38_cpu_release()
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
