# SPDX-License-Identifier: Apache-2.0
"""Lossless NVIDIA CSF32 scale restoration on Metal, preserving scale bytes."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    uint block = threadgroup_position_in_grid.x;
    uint local = thread_index_in_threadgroup;
    uint row_tiles = (R + 127) / 128;
    uint col_tiles = (C + 3) / 4;
    uint full_row_tiles = R / 128;
    uint tail_rows = (R % 128) / 32;
    uint full_tile_blocks = 4 * ((C % 4 == 0) ? 2 : 2);
    // Enumerate complete tiles, then the final row tile. C is a multiple
    // of four for all GLM projections; odd C is rejected by the wrapper.
    uint tile = block / 8;
    uint within = block % 8;
    if (tile >= full_row_tiles * col_tiles) {
        uint relative = block - full_row_tiles * col_tiles * 8;
        tile = full_row_tiles * col_tiles + relative / (2 * tail_rows);
        within = relative % (2 * tail_rows);
    }
    uint row_tile = tile / col_tiles;
    uint col_tile = tile % col_tiles;
    uint blocks_per_pair = row_tile < full_row_tiles ? 4 : tail_rows;
    uint r = row_tile * 128 + (within % blocks_per_pair) * 32 + local / 2;
    uint c = col_tile * 4 + (within / blocks_per_pair) * 2 + local % 2;
    uint packed = codes[block * 32 + local / 2];
    uint value = uint(bases[r]) + ((packed >> ((local % 2) * 4)) & 15);
    uint page = tile / 32;
    uint prefix = block + page;
    uint begin = page_bases[page] / 2 + offsets[prefix];
    uint end = page_bases[page] / 2 + offsets[prefix + 1];
    // At most 64 sorted exception records per block; binary search avoids
    // scanning the arena once per scale byte.
    uint lo = begin, hi = end;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if ((exceptions[mid] & 255) < local) lo = mid + 1;
        else hi = mid;
    }
    if (lo < end && (exceptions[lo] & 255) == local) value = exceptions[lo] >> 8;
    out[r * C + c] = uchar(value);
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_csf32_restore",
        input_names=["bases", "codes", "page_bases", "offsets", "exceptions"],
        output_names=["out"],
        source=_SOURCE,
    )


def mlx_csf32_restore(bases, codes, page_bases, offsets, exceptions, *, rows, columns):
    """Restore exact E4M3 scale codes from csf_32x2_v1 components.

    This entry covers GLM matrices (32-row alignment, four-column alignment).
    It retains the compressed source until the caller materializes the result.
    """
    if rows <= 0 or rows % 32 or columns <= 0 or columns % 4:
        raise ValueError("CSF restore requires rows aligned to 32 and columns to 4")
    blocks = rows // 32 * (columns // 2)
    tiles = ((rows + 127) // 128) * (columns // 4)
    pages = (tiles + 31) // 32
    arrays = (bases, codes, page_bases, offsets, exceptions)
    expected = (mx.uint8, mx.uint8, mx.uint32, mx.uint16, mx.uint16)
    if any(a.dtype != d or a.ndim != 1 for a, d in zip(arrays, expected)):
        raise ValueError("Invalid CSF component dtype or rank")
    if (bases.size, codes.size, page_bases.size, offsets.size) != (
        rows,
        rows * columns // 2,
        pages,
        blocks + pages,
    ):
        raise ValueError("Invalid CSF component lengths")
    return _kernel()(
        inputs=list(arrays),
        template=[("R", rows), ("C", columns)],
        grid=(blocks * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, columns)],
        output_dtypes=[mx.uint8],
    )[0]
