# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 single-row GEMV; original weights, FP32 accumulation."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(
    X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    acc = tl.full((ROWS, COLS), 0, tl.float32)
    for first in range(tl.cdiv(K, COLS)):
        k = first * COLS + cols
        x = tl.load(X + k, k < K, 0).to(tl.float32)
        w = tl.load(
            W + rows[:, None] * K + k[None, :],
            (rows[:, None] < N) & (k[None, :] < K),
            0,
        ).to(tl.float32)
        acc += w * x[None, :]
    result = tl.sum(acc, 1)
    tl.store(Y + rows, result, rows < N)


def applicable(x, weight, bias):
    # Only enable geometries measured against cuBLAS on the target SM120.
    # In particular the 512-output router regressed in the graph benchmark.
    measured_shapes = {
        (16480, 2560),
        (2560, 6144),
        (13312, 2560),
        (2560, 2560),
        (1280, 2560),
        (2560, 640),
    }
    # Keep opt-in: local kernel speedups did not produce the best observed
    # full-model throughput, and numerical parity is not a quality evaluation.
    return (
        os.getenv("SGLANG_QWEN38_DECODE_GEMV", "0") == "1"
        and tuple(weight.shape) in measured_shapes
        and bias is None
        and x.is_cuda
        and x.ndim == 2
        and x.shape[0] == 1
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and 512 <= weight.shape[0] <= 32768
        and 256 <= weight.shape[1] <= 32768
        and not x.requires_grad
        and not weight.requires_grad
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    )


def decode_gemv(x, weight):
    output = torch.empty((1, weight.shape[0]), device=x.device, dtype=x.dtype)
    _gemv[(triton.cdiv(weight.shape[0], 4),)](
        x, weight, output, weight.shape[0], weight.shape[1], 4, 256, num_warps=4
    )
    return output
