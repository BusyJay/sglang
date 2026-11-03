# SPDX-License-Identifier: Apache-2.0
"""GPU expert pairing reused explicitly across NVFP4 projections."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
uint pick=thread_position_in_grid.x,count=dims[0];
if(pick>=count)return;
uint expert=ids[pick],prior=0,next=pick;
for(uint i=0;i<pick;++i)prior+=ids[i]==expert;
if(prior&1){counts[pick]=0;partners[pick]=pick;return;}
for(uint i=pick+1;i<count;++i)if(ids[i]==expert){next=i;break;}
partners[pick]=next;counts[pick]=next==pick?1:2;
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_mlx_expert_pairs",
        input_names=["ids", "dims"],
        output_names=["partners", "counts"],
        source=_SOURCE,
    )


def mlx_expert_pair_worklist(indices):
    """Partition repeated expert picks into pairs, preserving output order.

    Each pair is written by its first pick. Odd occurrences retain a singleton.
    Supports repeated experts within one token and across tokens, including
    three or more occurrences; outputs are uint32 arrays of indices.size.
    Caller owns this per-forward result and passes it explicitly to projections.
    """
    if indices.dtype != mx.uint32 or not indices.size:
        raise ValueError("Expected nonempty uint32 expert indices")
    return tuple(
        _kernel()(
            inputs=[indices.reshape(-1), mx.array([indices.size], mx.uint32)],
            grid=(indices.size, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(indices.size,), (indices.size,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
    )
