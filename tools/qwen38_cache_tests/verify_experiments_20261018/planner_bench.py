#!/usr/bin/env python3
"""LFU planner CUDA-event timing under repeatable hit/miss churn; GPU required."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from sglang.kernels.ops.moe.qwen38_expert_cache import _module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no skipped result")
    reports = []
    for slots in (80, 144, 192, 256):
        for width in (2, 3):
            for churn in (0, 3, 30):
                for mode in (1, 2):
                    state = [
                        torch.zeros(513, dtype=torch.int32, device="cuda"),
                        torch.full((512,), -1, dtype=torch.int32, device="cuda"),
                        torch.full((slots,), -1, dtype=torch.int32, device="cuda"),
                        torch.zeros(slots, device="cuda"),
                        torch.zeros(slots, dtype=torch.int64, device="cuda"),
                        torch.zeros(slots, dtype=torch.int64, device="cuda"),
                        torch.zeros(1, dtype=torch.int64, device="cuda"),
                        torch.zeros(4 + 4 * slots, dtype=torch.int32, device="cuda"),
                    ]
                    ids = torch.empty((1, width * 10), dtype=torch.int32, device="cuda")
                    output = torch.empty_like(ids)
                    samples = []
                    # Pre-created route sequences include identical warmup for each mode.
                    sequence = [
                        (
                            torch.arange(width * 10, device="cuda", dtype=torch.int32)
                            + i * churn
                        )
                        .remainder(512)
                        .view_as(ids)
                        for i in range(250)
                    ]
                    events = [
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        for _ in sequence
                    ]
                    for i, (routes, (start, end)) in enumerate(zip(sequence, events)):
                        ids.copy_(routes)
                        start.record()
                        _module().decode_plan_parallel(
                            ids, output, *state, 0.95, 0, 0.7, 0, 0, mode
                        )
                        end.record()
                    torch.cuda.synchronize()
                    samples = [a.elapsed_time(b) for a, b in events[50:]]
                    reports.append(
                        dict(
                            slots=slots,
                            tokens=width,
                            churn=churn,
                            planner_mode=mode,
                            median_ms=statistics.median(samples),
                            p95_ms=sorted(samples)[189],
                        )
                    )
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                scope="planner only; excludes expert fill/compute; run parity test first",
                reports=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
