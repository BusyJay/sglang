# SPDX-License-Identifier: Apache-2.0
"""Paired production-ABI benchmark for three tokens and 1..4 miss experts.

Includes shared and partially overlapping routes, native and CSF scales, both
projection math conventions. Synthetic weights; CPU-only timing, not serving.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from build_qwen38_cpu_hybrid import check_temperature
from check_qwen38_cpu_csf import Library, fixture, physical_cores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--pause-ms", type=float, default=2)
    args = parser.parse_args()
    if args.samples < 5 or args.pause_ms < 0:
        parser.error("require at least five samples and a nonnegative pause")
    cores = physical_cores(args.cores)
    libraries = [Library(p) for p in (args.baseline_library, args.candidate_library)]
    result = {
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "cores": cores,
        "samples": args.samples,
        "pause_ms": args.pause_ms,
        "scope": "synthetic H=2560/I=640, 3 tokens, 1..4 distinct experts, warmed CPU ABI only",
        "libraries": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (args.baseline_library, args.candidate_library)
        ],
        "cases": [],
    }
    rng = random.Random(731)
    data = fixture(2560, 640)
    try:
        for backend in ("b12x", "triton"):
            for compressed in (False, True):
                for routing in ("shared", "mixed"):
                    for experts in (1, 2, 3, 4):
                        rows = (
                            [list(range(experts)) for _ in range(3)]
                            if routing == "shared"
                            else [
                                [e for e in range(experts) if (e + t) % 3 != 0]
                                for t in range(3)
                            ]
                        )
                        rows = [row + [-1] * (8 - len(row)) for row in rows]
                        calls = [
                            lib.call(data, 2560, 640, rows, cores, backend, compressed)
                            for lib in libraries
                        ]
                        samples = [[], []]
                        for iteration in range(args.samples + 5):
                            check_temperature()
                            order = [0, 1]
                            rng.shuffle(order)
                            for arm in order:
                                fn, params, out = calls[arm]
                                start = time.perf_counter_ns()
                                rc = fn(*params)
                                elapsed = (time.perf_counter_ns() - start) / 1000
                                if rc:
                                    raise RuntimeError(
                                        libraries[arm].lib.qwen38_cpu_error().decode()
                                    )
                                if iteration >= 5:
                                    samples[arm].append(elapsed)
                                if args.pause_ms:
                                    time.sleep(args.pause_ms / 1000)
                            if bytes(calls[0][2]) != bytes(calls[1][2]):
                                raise AssertionError(
                                    (backend, compressed, routing, experts)
                                )
                        medians = list(map(statistics.median, samples))
                        row = dict(
                            backend=backend,
                            compressed=compressed,
                            routing=routing,
                            experts=experts,
                            median_us=medians,
                            p95_us=[
                                sorted(values)[int(0.95 * (len(values) - 1))]
                                for values in samples
                            ],
                            reduction_percent=100 * (1 - medians[1] / medians[0]),
                            exact_fp32_match=True,
                            samples_us=samples,
                        )
                        result["cases"].append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                        print(
                            json.dumps(
                                {k: v for k, v in row.items() if k != "samples_us"}
                            ),
                            flush=True,
                        )
    finally:
        for lib in libraries:
            lib.release()


if __name__ == "__main__":
    main()
