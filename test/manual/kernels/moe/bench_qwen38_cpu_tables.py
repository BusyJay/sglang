# SPDX-License-Identifier: Apache-2.0
"""Isolate table setup using a hot synthetic pack; not a serving throughput claim."""

import argparse
import json
import statistics
import struct
import time
from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-library", required=True)
    parser.add_argument("--cores", default="0,1,2,3")
    args = parser.parse_args()
    cores = tuple(map(int, args.cores.split(",")))
    h, width, experts = 2560, 640, 3
    names = (
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
        "w2_blockscale_swizzled",
        "g1_alphas",
        "g2_alphas",
        "w13_input_scale_quant",
        "w2_input_scale_quant",
    )
    chunks = [
        bytearray([0x21]) * (experts * width * h),
        bytearray([0x12]) * (experts * h * width // 2),
        bytearray([0x38]) * (experts * 2 * width * h // 16),
        bytearray([0x38]) * (experts * h * width // 16),
        struct.pack("3f", 0.73, 1.17, 0.91),
        struct.pack("3f", 1.31, 0.69, 1.07),
        struct.pack("f", 0.83),
        struct.pack("f", 1.23),
    ]
    payload, records = bytearray(), {}
    for name, chunk in zip(names, chunks):
        records[name] = SimpleNamespace(offset=len(payload))
        payload.extend(chunk)
    pack = SimpleNamespace(
        manifest=SimpleNamespace(
            hidden_size=h,
            intermediate_size=width,
            num_experts=experts,
            top_k=4,
            tensors={0: records},
        ),
        cuda_source_mapping=lambda: payload,
    )
    operators = [
        CpuNvfp4(args.baseline_library, pack, cores, gpu_backend="b12x"),
        CpuNvfp4("", pack, cores, gpu_backend="b12x"),
    ]
    result = {"libraries": [op.library._name for op in operators], "cases": []}
    try:
        torch.manual_seed(42)
        for tokens in (1, 3):
            x = (torch.randn(tokens, h) * 0.05).bfloat16()
            ids = torch.tensor([[0, 1, 2, -1]] * tokens, dtype=torch.int32)
            weights = torch.ones_like(ids, dtype=torch.float32) / 3
            outputs = [torch.empty_like(x, dtype=torch.float32) for _ in operators]
            times = [[], []]
            for iteration in range(35):
                for index, op in enumerate(operators):
                    start = time.perf_counter()
                    op.run(0, x, ids, weights, outputs[index])
                    elapsed = time.perf_counter() - start
                    if iteration >= 5:
                        times[index].append(elapsed * 1e6)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            result["cases"].append(
                {
                    "tokens": tokens,
                    "baseline_median_us": statistics.median(times[0]),
                    "cached_median_us": statistics.median(times[1]),
                    "exact_fp32_match": True,
                }
            )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for op in operators:
            op.release_worker()
            op.close()


if __name__ == "__main__":
    main()
