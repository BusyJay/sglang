# SPDX-License-Identifier: Apache-2.0
"""Mixed FP8 shared and NVFP4 routed projections in one Metal dispatch."""

import re
from functools import cache

import mlx.core as mx

from sglang.kernels.ops.gemm.mlx_block_fp8 import _DECODE as _FP8_HEADER
from sglang.kernels.ops.gemm.mlx_block_fp8 import _SOURCE as _FP8_PAIR
from sglang.kernels.ops.gemm.mlx_fp8_rows import _DECODE as _FP8_DOWN_HEADER
from sglang.kernels.ops.gemm.mlx_fp8_rows import _SOURCE as _FP8_DOWN
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _HEADER as _NV_HEADER
from sglang.kernels.ops.gemm.mlx_nvfp4_gather import _SOURCE as _NV_DOWN
from sglang.kernels.ops.gemm.mlx_nvfp4_swiglu import _pair_source

_SHARED_ACT = r"""
    if (lane == 0 && start+t < COUNT && row+r < N) {
        T gate_value=T(g[t][r]),up_value=T(u[t][r]);
        if(CLAMP) {
            float bound=float(T(limit[0]));
            gate_value=T(metal::min(float(gate_value),bound));
            up_value=T(metal::clamp(float(up_value),-bound,bound));
        }
        auto fraction=1/(1+metal::precise::exp(metal::abs(gate_value)));
        T sigmoid=T(gate_value<T(0)?fraction:1-fraction);
        T activated=T(gate_value*sigmoid);
        shared_act[(start+t)*N+row+r]=T(activated*up_value);
    }
"""


@cache
def _kernel():
    routed = (
        _pair_source()
        .replace(
            "threadgroup_position_in_grid.z",
            "(threadgroup_position_in_grid.z-SHARED_TILES)",
        )
        .replace("2*RPS", "4*RPS")
    )
    shared = (
        _FP8_PAIR.replace("dims", "shared_dims")
        .replace("threadgroup_position_in_grid.x", "threadgroup_position_in_grid.y")
        .replace("threadgroup_position_in_grid.z", "threadgroup_position_in_grid.z")
        .replace("* RPS", "* 2")
        .replace("[RPS]", "[2]")
        .replace("r < RPS", "r < 2")
        .replace("ROWS", "SHARED_ROWS")
    )
    start = shared.index("        if (lane == 0 && start+t < COUNT")
    end = shared.index("\n        }", start) + len("\n        }")
    shared = shared[:start] + _SHARED_ACT + shared[end:]
    source = (
        "if(threadgroup_position_in_grid.z<SHARED_TILES) {\n"
        + shared
        + "\nreturn;\n}\n"
        + routed
    )
    return mx.fast.metal_kernel(
        name="sglang_mixed_fp8_nvfp4_swiglu",
        input_names=[
            "x",
            "wg",
            "sgate",
            "global_gate",
            "wu",
            "sup",
            "global_up",
            "ids",
            "dims",
            "limit",
            "owners",
            "partners",
            "counts",
            "gate_weight",
            "gate_scales",
            "up_weight",
            "up_scales",
            "xq",
            "xscale",
            "shared_dims",
        ],
        output_names=["out", "shared_act"],
        source=source,
        header=_NV_HEADER + _FP8_HEADER,
        compile_options={"math_mode": "safe"},
    )


def mlx_mixed_fp8_nvfp4_swiglu(
    x, gate, up, indices, worklist, shared_gate, shared_up, quantized, *, limit
):
    """Return routed and shared activations with independent quantization.

    Operand tuples are (NVFP4 bytes,/16 scale,encoded global) and
    (FP8 bytes,/128 block scales). Shared activations retain dynamic-/32
    quantization supplied by the caller. Compact routed tasks scatter back to
    their original picks; a separate role in the same dispatch owns shared rows.
    """
    if x.ndim < 2 or not x.shape[-1]:
        raise ValueError("Expected nonempty activation rows")
    rows, k = x.size // x.shape[-1], x.shape[-1]
    _validate_nvfp4(gate, k)
    _validate_nvfp4(up, k)
    _validate_shared(shared_gate, x, quantized)
    _validate_shared(shared_up, x, quantized)
    if gate[0].shape != up[0].shape or shared_gate[0].shape != shared_up[0].shape:
        raise ValueError("Expected matching gate/up projection geometry")
    if (
        indices.dtype != mx.uint32
        or indices.ndim != 3
        or indices.shape[:2] != x.shape[:2]
    ):
        raise ValueError("Expected one sequence of router picks for these rows")
    n, sn = gate[0].shape[-2], shared_gate[0].shape[-2]
    if (
        rows > 17
        or rows < 1
        or k % 512
        or n % 8
        or sn % 8
        or x.dtype not in (mx.bfloat16, mx.float32)
    ):
        raise ValueError("Expected covered mixed projection rows and native widths")
    count, picks = indices.size, indices.shape[-1]
    owners, partners, counts, _ = _validate_worklist(worklist, count)
    out, shared = _kernel()(
        inputs=[
            x.reshape(-1, k),
            *gate,
            *up,
            indices.reshape(-1),
            mx.array([k, n, picks], mx.uint32),
            mx.array([limit or 0], mx.float32),
            owners,
            partners,
            counts,
            shared_gate[0],
            shared_gate[1].reshape(-1),
            shared_up[0],
            shared_up[1].reshape(-1),
            *quantized,
            mx.array([k, sn, rows], mx.uint32),
        ],
        template=[
            ("T", x.dtype),
            ("O", x.dtype),
            ("RPS", 1),
            ("EXPERT", False),
            ("CLAMP", limit is not None),
            ("SHARED_ROWS", min(rows, 4)),
            ("SHARED_TILES", (rows + 3) // 4),
        ],
        grid=(32, max((n + 3) // 4, (sn + 7) // 8) * 4, count + (rows + 3) // 4),
        threadgroup=(32, 4, 1),
        output_shapes=[(count, n), (rows, sn)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return out.reshape(*indices.shape, 1, n), shared.reshape(*x.shape[:-1], sn)


@cache
def _down_kernel():
    routed = _NV_DOWN
    original = "uint pick=threadgroup_position_in_grid.z,nmatch=counts[pick];"
    compact = """uint task=threadgroup_position_in_grid.z-SHARED_TILES;
uint nmatch=counts[task];
uint pick=owners[task];"""
    routed = (
        routed.replace(original, compact)
        .replace("partner=partners[pick]", "partner=partners[task]")
        .replace("2*RPS", "4*RPS")
    )
    shared = (
        re.sub(r"\bK\b", "SK", re.sub(r"\bN\b", "SN", _FP8_DOWN))
        .replace("RPS", "SRPS")
        .replace("threadgroup_position_in_grid.x", "threadgroup_position_in_grid.y")
        .replace("e4m3(", "shared_e4m3(")
    )
    return mx.fast.metal_kernel(
        name="sglang_mixed_fp8_nvfp4_down",
        input_names=[
            "x",
            "weight",
            "scales",
            "global_scale",
            "ids",
            "dims",
            "owners",
            "partners",
            "counts",
            "shared_weight",
            "shared_scales",
            "xq",
            "xscale",
        ],
        output_names=["out", "shared_out"],
        source="if(threadgroup_position_in_grid.z<SHARED_TILES) {\n"
        + shared.replace("weight[", "shared_weight[")
        .replace("block_scales[", "shared_scales[")
        .replace("out[", "shared_out[")
        + "\nreturn;\n}\n"
        + routed,
        header=_NV_HEADER + _FP8_DOWN_HEADER.replace("e4m3(", "shared_e4m3("),
        compile_options={"math_mode": "safe"},
    )


def mlx_mixed_fp8_nvfp4_down(
    routed, shared, down, indices, worklist, shared_down, quantized
):
    """Project independent shared/routed activations in one launch.

    Routed output retains each original pick and its native QMV rounding;
    shared output retains dynamic-/32 activation and /128 block-FP8 arithmetic.
    The caller performs its original weighting, reduction and shared addition.
    """
    if indices.ndim != 3 or indices.shape[0] != 1:
        raise ValueError("Expected one sequence of routed picks")
    rows, picks = indices.shape[1:]
    if not 1 <= rows <= 17 or not 1 <= picks <= 32:
        raise ValueError("Expected covered row and pick counts")
    if routed.shape[:-2] != indices.shape or routed.shape[-2] != 1:
        raise ValueError("Expected one activation vector per routed pick")
    k, n = routed.shape[-1], down[0].shape[-2]
    _validate_nvfp4(down, k)
    _validate_shared(shared_down, shared, quantized)
    if (
        indices.dtype != mx.uint32
        or shared.dtype != routed.dtype
        or routed.dtype not in (mx.bfloat16, mx.float32)
        or shared.size // shared.shape[-1] != rows
        or shared_down[0].shape[0] != n
        or k % 512
        or n % 8
    ):
        raise ValueError("Expected compatible native mixed-down geometry")
    count = indices.size
    owners, partners, counts, _ = _validate_worklist(worklist, count)
    shared_k = shared.shape[-1]
    shared_rps = 2 if n >= 4096 else 1
    rps = 1
    shared_tiles = (rows + 3) // 4
    out, shared_out = _down_kernel()(
        inputs=[
            routed.reshape(count, k),
            *down,
            indices.reshape(-1),
            mx.array([k, n, picks], mx.uint32),
            owners,
            partners,
            counts,
            shared_down[0],
            shared_down[1].reshape(-1),
            *quantized,
        ],
        template=[
            ("T", routed.dtype),
            ("O", routed.dtype),
            ("RPS", rps),
            ("EXPERT", True),
            ("SINGLES", True),
            ("SHARED_TILES", shared_tiles),
            ("SN", n),
            ("SK", shared_k),
            ("COUNT", rows),
            ("BLOCK", 128),
            ("MX", False),
            ("A8", True),
            ("ROWS", min(rows, 4)),
            ("SRPS", shared_rps),
        ],
        grid=(
            32,
            max((n + 3) // 4, (n + 4 * shared_rps - 1) // (4 * shared_rps)) * 4,
            count + shared_tiles,
        ),
        threadgroup=(32, 4, 1),
        output_shapes=[(count, n), (rows, n)],
        output_dtypes=[routed.dtype, routed.dtype],
    )
    return out.reshape(*indices.shape, 1, n), shared_out.reshape(*shared.shape[:-1], n)


def _validate_nvfp4(values, k):
    weight, scales, global_scale = values
    if (
        weight.ndim != 3
        or weight.dtype != mx.uint8
        or weight.shape[-1] * 2 != k
        or scales.dtype != mx.uint8
        or scales.shape != (*weight.shape[:2], k // 16)
        or global_scale.dtype != mx.float32
        or global_scale.size != weight.shape[0]
    ):
        raise ValueError("Expected native NVFP4 bytes and encoded expert scales")


def _validate_shared(values, x, quantized):
    weight, scales = values
    if (
        weight.ndim != 2
        or weight.dtype != mx.uint8
        or x.ndim < 2
        or weight.shape[-1] != x.shape[-1]
        or weight.shape[-1] % 128
        or scales.dtype != mx.float32
        or scales.size != ((weight.shape[0] + 127) // 128) * (weight.shape[1] // 128)
    ):
        raise ValueError("Expected /128 block-FP8 shared projection")
    rows = x.size // x.shape[-1]
    codes, activation_scales = quantized
    if (
        codes.dtype != mx.uint8
        or codes.shape != (rows, x.shape[-1])
        or activation_scales.dtype != mx.float32
        or activation_scales.shape != (rows, x.shape[-1] // 32)
    ):
        raise ValueError("Expected explicit dynamic-/32 activation codes and scales")


def _validate_worklist(worklist, count):
    if (
        len(worklist) != 4
        or any(
            value.dtype != mx.uint32 or value.shape != (count,)
            for value in worklist[:3]
        )
        or worklist[3].dtype != mx.uint32
        or worklist[3].shape != (1,)
    ):
        raise ValueError("Expected the matching compact router worklist")
    return worklist
