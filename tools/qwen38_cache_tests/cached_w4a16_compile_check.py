#!/usr/bin/env python3
"""Offline compile of the optional cache W4A16 kernels for SM120."""

import json

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sglang.srt.layers.quantization.qwen38_cached_w4a16 import _cache_project


def main():
    for first, k, n in [(True, 2560, 640), (False, 640, 2560)]:
        source = ASTSource(
            _cache_project,
            signature={
                "X": "*bf16",
                "W": "*u8",
                "S": "*fp8e4nv",
                "G": "*fp32",
                "Q": "*fp32",
                "IDs": "*i32",
                "Router": "*fp32",
                "Y": "*bf16" if first else "*fp32",
                "E": "i32",
            },
            constexprs={
                "K": k,
                "N": n,
                "TOPK": 10,
                "FIRST": first,
                "BN": 32,
                "BK": 256,
            },
        )
        triton.compile(
            source, target=GPUTarget("cuda", 120, 32), options={"num_warps": 4}
        )
        print(
            json.dumps(
                {
                    "kernel": "cache_gate_up" if first else "cache_down",
                    "sm120_compiled": True,
                }
            )
        )

    from sglang.srt.layers.quantization.qwen38_fp32_norm import _norm

    for hidden, group in ((10240, 2560), (160, 160)):
        source = ASTSource(
            _norm,
            signature={"X": "*bf16", "W": "*fp32", "Y": "*bf16"},
            constexprs={
                "H": hidden,
                "G": group,
                "EPS": 1e-6,
                "B": triton.next_power_of_2(group),
            },
        )
        triton.compile(
            source, target=GPUTarget("cuda", 120, 32), options={"num_warps": 4}
        )
        print(
            json.dumps(
                {"kernel": "fp32_grouped_norm", "group": group, "sm120_compiled": True}
            )
        )


if __name__ == "__main__":
    main()
