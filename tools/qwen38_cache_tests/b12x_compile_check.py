#!/usr/bin/env python3
"""Offline SM120a compile with mocked hardware facts; no CUDA execution."""

import os

os.environ["CUTE_DSL_ARCH"] = "sm_120a"
from unittest.mock import patch
import torch
from cuda.bindings import driver as cuda
from b12x.moe._shared.kernels.w4a16 import kernel
from b12x.moe._shared.kernels import micro

with patch.object(micro, "get_num_sm", return_value=170), patch.object(
    micro, "get_max_active_clusters", return_value=170
), patch.object(kernel, "current_cuda_stream", return_value=cuda.CUstream(0)):
    for m in (1, 2, 3):
        for dtype in (torch.int32, torch.int64):
            print("COMPILING", m, dtype, flush=True)
            result = kernel._compile_w4a16_small_m_direct(
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
            print("COMPILED", m, dtype, result, flush=True)
