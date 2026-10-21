# SPDX-License-Identifier: Apache-2.0
"""Native value validation preserves Python errors without Torch hot-path scans."""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
from check_qwen38_cpu_hybrid import fixture

from sglang.kernels.ops.moe.cpu_nvfp4 import CpuNvfp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    payload, offsets, _, _ = fixture(64, 32)
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
    pack = SimpleNamespace(
        manifest=SimpleNamespace(
            hidden_size=64,
            intermediate_size=32,
            num_experts=3,
            top_k=4,
            tensors={
                0: {
                    name: SimpleNamespace(offset=offset)
                    for name, offset in zip(names, offsets)
                }
            },
        ),
        cuda_source_mapping=lambda: payload,
    )
    cpu = CpuNvfp4("", pack, (args.core,), gpu_backend="b12x")
    x = torch.ones((3, 64), dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 2, -1]] * 3, dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    output = torch.empty_like(x, dtype=torch.float32)
    try:
        with patch.object(
            torch, "isfinite", side_effect=AssertionError("duplicate Torch validation")
        ):
            cpu.run(0, x, ids, weights, output)
        assert torch.isfinite(output).all()
        for tensor, index, bad in (
            (ids, (0, 0), 3),
            (ids, (1, 1), -2),
            (x, (0, 0), float("nan")),
            (x, (2, 1), float("inf")),
            (weights, (0, 2), float("nan")),
        ):
            saved = tensor[index].clone()
            tensor[index] = bad
            try:
                cpu.run(0, x, ids, weights, output)
            except ValueError:
                print(f"PASS rejected {tensor.dtype} value={bad}", flush=True)
            else:
                raise AssertionError("invalid input was accepted")
            finally:
                tensor[index] = saved
    finally:
        cpu.release_worker()
        cpu.close()
    print("PASS valid native call performs no Torch finite-value scans", flush=True)


if __name__ == "__main__":
    main()
