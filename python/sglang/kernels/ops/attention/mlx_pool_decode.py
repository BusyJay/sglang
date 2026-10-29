"""Single-query GQA/MHA attention reading SGLang's token-indexed MLX KV pool.

The slot table is a page table with one token per page. Split-KV partitions
bound the serial work of each SIMD group; only partial outputs and log-sum-exp
values are materialized, never a gathered/padded copy of the KV history.
"""

from functools import cache

import mlx.core as mx

_PARTITION_SIZE = 128
_DTYPES = (mx.float16, mx.bfloat16, mx.float32)

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint part = threadgroup_position_in_grid.z;
    const uint kh = h / (Q_HEADS / KV_HEADS);
    const uint start = part * PARTITION_SIZE;
    const uint end = min(start + PARTITION_SIZE, uint(lengths[b]));
    const size_t qbase = (size_t(b) * Q_HEADS + h) * HEAD_DIM;
    float q[HEAD_DIM / 32];
    float acc[HEAD_DIM / 32];
    for (uint i = 0; i < HEAD_DIM / 32; ++i) {
        q[i] = float(queries[qbase + i * 32 + lane]);
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY;
    float normalizer = 0.0f;
    for (uint t = start; t < end; ++t) {
        const uint slot = uint(slots[size_t(b) * slots_shape[1] + t]);
        const size_t base = (size_t(slot) * KV_HEADS + kh) * HEAD_DIM;
        float dot = 0.0f;
        for (uint i = 0; i < HEAD_DIM / 32; ++i) {
            dot += q[i] * float(keys[base + i * 32 + lane]);
        }
        const float score = simd_sum(dot) * scale;
        const float next_max = max(maximum, score);
        const float old_weight = exp(maximum - next_max);
        const float weight = exp(score - next_max);
        normalizer = normalizer * old_weight + weight;
        for (uint i = 0; i < HEAD_DIM / 32; ++i) {
            acc[i] = acc[i] * old_weight
                + weight * float(values[base + i * 32 + lane]);
        }
        maximum = next_max;
    }
    const uint partitions = (slots_shape[1] + PARTITION_SIZE - 1) / PARTITION_SIZE;
    const size_t row = (size_t(b) * Q_HEADS + h) * partitions + part;
    for (uint i = 0; i < HEAD_DIM / 32; ++i) {
        partial[row * HEAD_DIM + i * 32 + lane] =
            normalizer > 0.0f ? acc[i] / normalizer : 0.0f;
    }
    if (lane == 0) {
        lse[row] = normalizer > 0.0f ? maximum + log(normalizer) : -INFINITY;
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_pool_decode_attention",
        input_names=["queries", "keys", "values", "slots", "lengths", "scale"],
        output_names=["partial", "lse"],
        source=_SOURCE,
    )


def mlx_pool_decode_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    slots: mx.array,
    lengths: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Attend ``[B,H,1,D]`` queries to ``[capacity,KVH,D]`` pool buffers.

    ``slots`` is an int32 ``[B,width]`` token-to-slot table. ``lengths`` is
    int32 ``[B]`` and includes the current token (already scattered into the
    pool). The caller owns slot lifetime and must supply 1 <= lengths <= width,
    and 0 < slots < capacity in every valid prefix. Unused table tails are not
    read. D must be 32, 64, 96, ..., 256; H must be divisible by KVH.
    """
    if queries.ndim != 4 or queries.shape[2] != 1:
        raise ValueError("pool decode expects queries shaped [B,H,1,D]")
    batch, heads, _, dim = queries.shape
    if keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError("pool K/V must have matching [capacity,KVH,D] shapes")
    kv_heads = keys.shape[1]
    if (
        not batch
        or not heads
        or not kv_heads
        or heads % kv_heads
        or dim < 32
        or dim > 256
        or dim % 32
        or keys.shape[2] != dim
    ):
        raise ValueError(
            "pool decode requires uniform D=32..256 and H divisible by KVH"
        )
    if (
        queries.dtype not in _DTYPES
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
    ):
        raise ValueError("pool decode requires matching FP16/BF16/FP32 Q/K/V")
    if slots.ndim != 2 or slots.shape[0] != batch or slots.shape[1] < 1:
        raise ValueError("slots must have shape [B,width] with width >= 1")
    if (
        lengths.shape != (batch,)
        or slots.dtype != mx.int32
        or lengths.dtype != mx.int32
    ):
        raise ValueError("slots and lengths must be int32 with lengths shaped [B]")

    width = slots.shape[1]
    partitions = (width + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    partial, lse = _kernel()(
        inputs=[queries, keys, values, slots, lengths, mx.array(scale, mx.float32)],
        template=[
            ("Q_HEADS", heads),
            ("KV_HEADS", kv_heads),
            ("HEAD_DIM", dim),
            ("PARTITION_SIZE", _PARTITION_SIZE),
        ],
        grid=(heads * 32, batch, partitions),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, heads, partitions, dim), (batch, heads, partitions)],
        output_dtypes=[mx.float32, mx.float32],
    )
    if partitions == 1:
        return partial.astype(queries.dtype)
    weights = mx.softmax(lse, axis=-1)
    return mx.sum(partial * weights[..., None], axis=2, keepdims=True).astype(
        queries.dtype
    )
