# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity GPU token sets for indexed vocabulary projections."""

from functools import cache

import mlx.core as mx

_SENTINEL = 0xFFFFFFFF
_THREADS = 128


@cache
def _kernels():
    return (
        mx.fast.metal_kernel(
            name="sglang_vocab_set_bits",
            input_names=["ids", "dims"],
            output_names=["bits"],
            atomic_outputs=True,
            source=r"""
                uint i=thread_position_in_grid.x;
                if (i<dims[0]) {
                    uint token=ids[i];
                    if (token<dims[1]) atomic_fetch_or_explicit(
                        bits+token/32, 1u<<(token%32), memory_order_relaxed);
                }
            """,
        ),
        mx.fast.metal_kernel(
            name="sglang_vocab_compact_bits",
            input_names=["bits", "dims"],
            output_names=["ids", "count"],
            source=r"""
                uint tid=thread_position_in_threadgroup.x;
                uint group=threadgroup_position_in_grid.x;
                uint lane=thread_index_in_simdgroup;
                uint sg=simdgroup_index_in_threadgroup;
                uint words=dims[0], capacity=dims[1], start=group*128;
                uint previous=0, total=0;
                // A normal vocabulary has only ~5k words. Read the bitset
                // from cache in each group to eliminate a global scan pass.
                for(uint i=tid;i<words;i+=128) {
                    uint n=popcount(bits[i]);
                    total+=n;
                    if(i<start)previous+=n;
                }
                uint i=start+tid, word=i<words?bits[i]:0;
                uint n=popcount(word), local=simd_prefix_exclusive_sum(n);
                previous=simd_sum(previous);
                total=simd_sum(total);
                uint span=simd_sum(n);
                threadgroup uint summary[4][3];
                if(lane==0){summary[sg][0]=previous;summary[sg][1]=total;summary[sg][2]=span;}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                previous=0; total=0;
                for(uint j=0;j<4;++j) {
                    previous+=summary[j][0];total+=summary[j][1];
                    if(j<sg)local+=summary[j][2];
                }
                uint dst=previous+local;
                while(word) {
                    ids[dst++]=32*i+ctz(word);
                    word &= word-1;
                }
                uint stride=((words+127)/128)*128;
                for(uint j=total+i;j<capacity;j+=stride) ids[j]=0xffffffffu;
                if(i==0)count[0]=total;
            """,
        ),
        mx.fast.metal_kernel(
            name="sglang_vocab_append_recent",
            input_names=["bits", "base", "count", "recent", "dims"],
            output_names=["ids", "limit"],
            source=r"""
                uint i=thread_position_in_grid.x;
                uint n=count[0], end=n+dims[1];
                if(i==0) limit[0]=end;
                if(i>=dims[0]) return;
                uint token=0xffffffffu;
                if(i<n) token=base[i];
                else if(i<end) {
                    uint candidate=recent[i-n];
                    if(candidate<dims[2] && !(bits[candidate/32] & (1u<<(candidate%32))))
                        token=candidate;
                }
                ids[i]=token;
            """,
        ),
    )


def _grid(size):
    return (_THREADS * ((size + _THREADS - 1) // _THREADS), 1, 1)


def _set_bits(ids, vocab_size):
    words = (vocab_size + 31) // 32
    return _kernels()[0](
        inputs=[ids, mx.array([ids.size, vocab_size], mx.uint32)],
        grid=_grid(ids.size),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(words,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )[0]


def _compact(bits, capacity):
    dims = mx.array([bits.size, capacity], mx.uint32)
    return tuple(
        _kernels()[1](
            inputs=[bits, dims],
            grid=_grid(bits.size),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(capacity,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
    )


def mlx_vocabulary_set(ids, vocab_size, capacity):
    """Return (bitset, unique global IDs, device count) without sorting IDs.

    Capacity must cover every input ID; padding and out-of-vocabulary IDs
    are ignored. Only the first device-count elements are valid output IDs.
    """
    if (
        ids.ndim != 1
        or ids.dtype != mx.uint32
        or not 0 < ids.size <= capacity < 2**31
        or not 0 < vocab_size < _SENTINEL
    ):
        raise ValueError("Expected nonempty uint32 IDs and a covering capacity")
    bits = _set_bits(ids, vocab_size)
    return bits, *_compact(bits, capacity)


def mlx_vocabulary_candidates(pools, recent, vocab_size):
    """Merge request pools with distinct recent IDs, returning IDs and limit.

    A single pool keeps its compact prefix and masks recent IDs already in
    the base bitset. The limit includes the small recent segment, which may
    contain sentinel holes. Multiple pools compact the union instead.
    """
    if not pools or recent.ndim != 1 or recent.dtype != mx.uint32 or not recent.size:
        raise ValueError("Expected at least one pool and uint32 recent IDs")
    capacity = sum(ids.size for _, ids, _ in pools) + recent.size
    bits, base, count = pools[0]
    if len(pools) > 1:
        for other, _, _ in pools[1:]:
            bits = mx.bitwise_or(bits, other)
        return _compact(mx.bitwise_or(bits, _set_bits(recent, vocab_size)), capacity)
    return tuple(
        _kernels()[2](
            inputs=[
                bits,
                base,
                count,
                recent,
                mx.array([capacity, recent.size, vocab_size], mx.uint32),
            ],
            grid=_grid(capacity),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(capacity,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
    )
