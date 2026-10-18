#!/usr/bin/env python3
"""Offline SM120a compile and real vendor ABI marshaling; no GPU execution.

Hardware facts are mocked and weight pointers are meta tensors. This verifies
specialization construction and argument conversion, not numerical correctness.
"""

import os

os.environ["CUTE_DSL_ARCH"] = "sm_120a"
from types import SimpleNamespace
from unittest.mock import patch

import torch
from b12x.moe._shared.kernels import micro
from b12x.moe._shared.kernels.w4a16 import kernel

from sglang.srt.layers.quantization.qwen38_cached_b12x import _PreparedSmallMLaunch

with (
    patch.object(micro, "get_num_sm", return_value=170),
    patch.object(micro, "get_max_active_clusters", return_value=170),
    patch.object(kernel, "current_cuda_stream", return_value=kernel.cuda.CUstream(0)),
):
    for m in (1, 2, 3):
        for dtype in (torch.int32, torch.int64):
            original = kernel._compile_w4a16_small_m_direct
            launch = original(
                m=m,
                hidden_size=2560,
                intermediate_size=640,
                num_experts=112,
                topk=10,
                activation="silu",
                fast_math=True,
                topk_ids_dtype=dtype,
                device=None,
                w13_layout="w13",
            )
            args_box = []
            fake = SimpleNamespace(
                compiled=lambda *args: args_box.extend(args), grid_x=launch.grid_x
            )
            runner = _PreparedSmallMLaunch(
                kernel,
                torch.zeros(682, dtype=torch.int32),
                torch.empty(147456 // 2, dtype=torch.bfloat16),
            )
            weights = {
                "w13_weight": torch.empty(
                    112, 1280, 1280, dtype=torch.uint8, device="meta"
                ),
                "w2_weight": torch.empty(
                    112, 2560, 320, dtype=torch.uint8, device="meta"
                ),
                "w13_blockscale_swizzled": torch.empty(
                    112, 1280, 160, dtype=torch.uint8, device="meta"
                ),
                "w2_blockscale_swizzled": torch.empty(
                    112, 2560, 40, dtype=torch.uint8, device="meta"
                ),
            }
            x = torch.empty(m, 2560, dtype=torch.bfloat16)
            with patch.object(
                kernel, "_compile_w4a16_small_m_direct", return_value=fake
            ):
                runner.run(
                    x,
                    torch.empty(m, 10, dtype=dtype),
                    torch.empty(m, 10),
                    weights,
                    torch.ones(112),
                    torch.ones(112),
                    torch.empty_like(x),
                    SimpleNamespace(cuda_stream=0),
                )
            # Execute actual vendor argument conversion, without launching CUDA.
            executable = launch.compiled._executable
            print("compiled", m, dtype, type(executable), len(args_box), flush=True)
            packed = executable.generate_execution_args(*args_box)
            print("MARSHALLED", m, dtype, type(packed), flush=True)
