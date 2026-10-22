# SPDX-License-Identifier: Apache-2.0
"""Inline CSF loads for native-order scale tensors inside W4A16 projections."""

import triton
import triton.language as tl


@triton.jit
def load_scale(
    S,
    expert,
    row,
    col,
    rows: tl.constexpr,
    columns: tl.constexpr,
    stride: tl.constexpr,
    valid,
):
    pos = (
        ((row // 128 * (columns // 4) + col // 4) * 32 + row % 32) * 4
        + (row % 128) // 32
    ) * 4 + col % 4
    if stride == 0:
        return tl.load(S + expert * rows * columns + pos, mask=valid, other=0.0).to(
            tl.float32
        )
    else:
        base = S + expert * stride
        lane = (pos % 512) // 4
        fixed = base + 1024 + (row // 128) * (128 * (1 + columns // 2))
        b = tl.load(fixed + lane, mask=valid, other=0).to(tl.uint32)
        packed = tl.load(
            fixed + 128 + (pos % (128 * columns)) // 2, mask=valid, other=0
        ).to(tl.uint32)
        value = b + ((packed >> ((pos % 2) * 4)) & 15)
        meta = (base + 1024 + rows * (1 + columns // 2)).to(
            tl.pointer_type(tl.uint32)
        ) + (pos // 512) * 8
        group, bit = lane // 32, lane % 32
        mask = tl.load(meta + 2 + group, mask=valid, other=0)
        flag = tl.full(bit.shape, 1, tl.uint32) << bit
        present = valid & ((mask & flag) != 0)
        start = tl.load(meta, mask=present, other=0)
        prefix = tl.load(meta + 1, mask=present, other=0)
        rank = ((prefix >> (group * 8)) & 255) + tl.extra.cuda.libdevice.popc(
            (mask & (flag - 1)).to(tl.int32)
        )
        records = (base + 1024 + rows * (1 + columns // 2) + rows * columns // 16).to(
            tl.pointer_type(tl.uint32)
        )
        external = tl.load(
            base.to(tl.pointer_type(tl.uint64)) + 124 + tl.full(pos.shape, 0, tl.int32),
            mask=present,
            other=0,
        )
        records = tl.where(
            external != 0, external.to(tl.pointer_type(tl.uint32)), records
        )
        replacement = tl.load(records + start + rank, mask=present, other=0)
        value = tl.where(present, (replacement >> ((pos % 4) * 8)) & 255, value)
        return value.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
