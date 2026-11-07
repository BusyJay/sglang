# SPDX-License-Identifier: Apache-2.0
"""Exact tiled Metal top-k for MLX logits, without a vocabulary-wide sort."""

from functools import cache

import mlx.core as mx

_THREADS = 256
_TILE = 2048

# Each lane retains eight input pairs. Sixteen SIMD reductions select the tile
# winners without sorting the tile or maintaining a sixteen-entry lane heap.
# All lanes participate in both barriers, including padded tiles and merge tails.
_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint block = threadgroup_position_in_grid.x;
    const uint width = params[0];
    const uint blocks = params[1];
    float local_values[ITEMS];
    uint local_ids[ITEMS];
    for (uint i = 0; i < ITEMS; ++i) {
        uint col = block * TILE + tid + i * 256;
        bool valid = col < width;
        local_values[i] = valid ? float(logits[size_t(row) * width + col]) : -INFINITY;
        local_ids[i] = valid ? (MERGE ? input_ids[size_t(row) * width + col] : col) : 0xffffffffu;
    }
    threadgroup float scores[8];
    threadgroup uint tokens[8];
    threadgroup uint categories[8];
    threadgroup uint chosen;
    for (uint rank = 0; rank < 16; ++rank) {
        float best = -INFINITY;
        uint id = 0xffffffffu;
        uint category = 0;
        for (uint i = 0; i < ITEMS; ++i) {
            uint candidate_id = local_ids[i];
            float value = local_values[i];
            uint cat = candidate_id == 0xffffffffu ? 0 : (isnan(value) ? 1 : 2);
            float score = isnan(value) ? -INFINITY : value;
            if (cat > category || (cat == category &&
                (score > best || (score == best && candidate_id < id)))) {
                category = cat; best = score; id = candidate_id;
            }
        }
        uint max_category = simd_max(category);
        best = simd_max(category == max_category ? best : -INFINITY);
        // Resolve the id from the local arrays against the reduced score.
        uint winner_id = 0xffffffffu;
        for (uint i = 0; i < ITEMS; ++i) {
            float value = local_values[i];
            uint cat = local_ids[i] == 0xffffffffu ? 0 : (isnan(value) ? 1 : 2);
            float score = isnan(value) ? -INFINITY : value;
            if (cat == max_category && score == best)
                winner_id = min(winner_id, local_ids[i]);
        }
        winner_id = simd_min(winner_id);
        if (lane == 0) {
            scores[sg] = best; tokens[sg] = winner_id; categories[sg] = max_category;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            uint cat = lane < 8 ? categories[lane] : 0;
            float score = lane < 8 ? scores[lane] : -INFINITY;
            uint token = lane < 8 ? tokens[lane] : 0xffffffffu;
            uint final_category = simd_max(cat);
            float final_score = simd_max(cat == final_category ? score : -INFINITY);
            uint final_id = simd_min(cat == final_category && score == final_score ? token : 0xffffffffu);
            if (lane == 0) {
                chosen = final_id;
                size_t out = (size_t(row) * blocks + block) * 16 + rank;
                values[out] = T(final_category == 1 ? NAN : final_score);
                indices[out] = final_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < ITEMS; ++i)
            if (local_ids[i] == chosen) local_ids[i] = 0xffffffffu;
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="sglang_exact_top16",
        input_names=["logits", "input_ids", "params"],
        output_names=["values", "indices"],
        source=_SOURCE,
    )


def _select(logits, input_ids, *, merge, tile):
    rows, width = logits.shape
    blocks = (width + tile - 1) // tile
    return _kernel()(
        inputs=[logits, input_ids, mx.array([width, blocks], mx.uint32)],
        template=[
            ("T", logits.dtype),
            ("MERGE", merge),
            ("TILE", tile),
            ("ITEMS", (tile + _THREADS - 1) // _THREADS),
        ],
        grid=(_THREADS * blocks, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, blocks * 16), (rows, blocks * 16)],
        output_dtypes=[logits.dtype, mx.uint32],
    )


def mlx_topk(logits, k=16):
    """Return ``(values, indices)`` along the last axis, best first.

    Equal values prefer the lower vocabulary id, including negative infinity.
    Values retain the input dtype; indices are uint32. NaNs rank after numbers,
    with lower ids first. Leading dimensions are preserved, e.g. [B,7,V].

    FP32/BF16/FP16 top16 uses two Metal dispatches for vocabularies through
    262144, including V=154880 and any nonempty row batch. Larger vocabularies
    recursively reduce tile winners. Width and batch size are runtime data;
    the main path has no shape-dependent templates or host synchronization.
    Other k use stable MLX argsort as a correctness fallback, not the top16
    serving path. DFlash2 uses this operator for its candidate shortlist.
    """
    if logits.ndim < 1:
        raise ValueError("mlx_topk requires at least one dimension")
    width = logits.shape[-1]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= width:
        raise ValueError("k must be an integer in [1, vocabulary size]")
    if logits.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError("mlx_topk requires FP16, BF16, or FP32 logits")
    if k != 16 or logits.size == 0:
        ids = mx.argsort(-logits, axis=-1)[..., :k]
        return mx.take_along_axis(logits, ids, axis=-1), ids
    shape = logits.shape[:-1] + (k,)
    flat = logits.reshape(-1, width)
    values, ids = _select(flat, mx.array([0], mx.uint32), merge=False, tile=_TILE)
    while values.shape[-1] > _TILE:
        values, ids = _select(values, ids, merge=True, tile=_TILE)
    # A fixed merge tile avoids recompilation as row counts / vocabulary grow.
    values, ids = _select(values, ids, merge=True, tile=_TILE)
    return values.reshape(shape), ids.reshape(shape)


def mlx_topk_candidates(logits, token_ids, k=16):
    """Exact top-k over explicit global IDs, excluding uint32 sentinel padding.

    Numbers precede NaNs and equal scores prefer the smaller global ID, as in
    mlx_topk. Duplicate IDs are selected once, at their maximum score. The
    caller supplies at least k distinct non-sentinel IDs.
    """
    if (
        logits.ndim < 1
        or not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k <= 16
        or logits.shape[-1] < k
    ):
        raise ValueError("Expected at least k candidates, with k in 1..16")
    if logits.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise TypeError("Expected floating candidate scores")
    if token_ids.dtype != mx.uint32 or token_ids.shape not in (
        (logits.shape[-1],),
        logits.shape,
    ):
        raise ValueError("Expected shared or per-row uint32 token IDs")
    shape = logits.shape[:-1] + (k,)
    scores = logits.reshape(-1, logits.shape[-1])
    ids = mx.broadcast_to(token_ids, logits.shape).reshape(scores.shape)
    values, ids = _select(scores, ids, merge=True, tile=_TILE)
    while values.shape[-1] > _TILE:
        values, ids = _select(values, ids, merge=True, tile=_TILE)
    values, ids = _select(values, ids, merge=True, tile=_TILE)
    return values[:, :k].reshape(shape), ids[:, :k].reshape(shape)
