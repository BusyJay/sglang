"""Compare one MLX attention wrapper's padded and direct-pool decode paths.

Uses random attention weights and fragmented token slots. Timings include
projections, RoPE, request-cache writes and per-step batch metadata (including
the CPU slot-table snapshot for pool attention), followed by mx.eval. This is
an attention microbenchmark, not a model-serving throughput measurement.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes + args.context_lengths + [args.repeats, args.warmup]) < 1:
        parser.error(
            "batch sizes, context lengths, repeats and warmup must be positive"
        )

    import mlx.core as mx
    import numpy as np
    import torch
    from mlx_lm.models.llama import Attention, ModelArgs

    from sglang.srt.hardware_backend.mlx.aot import MlxAOTKernelSet
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MlxAttentionKVPool,
        MLXAttentionWrapper,
        clear_context,
        set_context,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.pool_decode import (
        MlxPoolDecodeContext,
    )

    dtype = mx.bfloat16
    heads, kv_heads, dim = 16, 4, 128
    records = []
    for batch in args.batch_sizes:
        for context in args.context_lengths:
            for distribution in ("equal", "mixed"):
                mx.random.seed(23)
                rng = np.random.default_rng(23)
                lengths = (
                    [context] * batch
                    if distribution == "equal"
                    else [max(1, context // (1 + i % 4)) for i in range(batch)]
                )
                pool = MlxAttentionKVPool(sum(lengths) + 1, 1, kv_heads, dim, dtype)
                pool.k_buffer[0] = mx.random.normal(pool.k_buffer[0].shape).astype(
                    dtype
                )
                pool.v_buffer[0] = mx.random.normal(pool.v_buffer[0].shape).astype(
                    dtype
                )
                table = torch.zeros((batch, context), dtype=torch.int32)
                permutation = rng.permutation(np.arange(1, sum(lengths) + 1))
                caches = []
                cursor = 0
                for i, length in enumerate(lengths):
                    slots = permutation[cursor : cursor + length]
                    cursor += length
                    table[i, :length] = torch.tensor(slots, dtype=torch.int32)
                    k, v = pool.get_kv(0, mx.array(slots, mx.int32))
                    cache = ContiguousAttentionKVCache(max_seq_len=length)
                    cache.update_and_fetch(
                        k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None]
                    )
                    caches.append([cache])
                inner = Attention(
                    ModelArgs(
                        model_type="llama",
                        hidden_size=heads * dim,
                        num_hidden_layers=1,
                        intermediate_size=heads * dim,
                        num_attention_heads=heads,
                        num_key_value_heads=kv_heads,
                        rms_norm_eps=1e-5,
                        vocab_size=32,
                    )
                )
                inner.set_dtype(dtype)
                wrapper = MLXAttentionWrapper(inner, layer_idx=0)
                inputs = mx.random.normal((batch, 1, heads * dim)).astype(dtype)
                req_ids = [str(i) for i in range(batch)]
                row_ids = {rid: i for i, rid in enumerate(req_ids)}
                req_pool = SimpleNamespace(req_to_token=table)
                mx.eval(
                    inputs,
                    inner.parameters(),
                    *pool.all_buffers(),
                    *[array for cache in caches for array in cache[0].state],
                )

                def forward(use_pool, *, caches=caches, pool=pool, wrapper=wrapper):
                    for cache, length in zip(caches, lengths):
                        cache[0].offset = length - 1
                    ctx = BatchedDecodeContext.from_decode(
                        caches=caches,
                        req_ids=req_ids,
                        aot_kernels=MlxAOTKernelSet(),
                        kv_pool=pool,
                        req_pool_idx=row_ids,
                        req_to_token_pool=req_pool,
                        full_kv_pool_index_by_layer={0: 0},
                    )
                    if use_pool:
                        ctx.pool_attention = MlxPoolDecodeContext.from_requests(
                            pool=pool,
                            req_ids=req_ids,
                            seq_lens=ctx.seq_lens,
                            req_pool_idx=row_ids,
                            req_to_token_pool=req_pool,
                        )
                    set_context(ctx)
                    try:
                        return wrapper(inputs)
                    finally:
                        clear_context()

                def evaluate(use_pool, *, forward=forward, caches=caches):
                    output = forward(use_pool)
                    # Pool reads and fallback mirror writes are separate lazy
                    # branches. Evaluate both, as runner.eval_pending does.
                    mx.eval(output, *[a for cache in caches for a in cache[0].state])
                    return output

                reference = evaluate(False)
                result = evaluate(True)
                max_error = mx.max(
                    mx.abs(reference.astype(mx.float32) - result.astype(mx.float32))
                ).item()
                if not mx.allclose(reference, result, atol=0.03, rtol=0.03).item():
                    raise RuntimeError(
                        f"pool attention parity failed: max_error={max_error}"
                    )
                for _ in range(args.warmup):
                    for mode in (False, True):
                        evaluate(mode)
                timings = {False: [], True: []}
                # Interleave and reverse order to reduce clock/thermal drift.
                for repeat in range(args.repeats):
                    for mode in (False, True) if repeat % 2 == 0 else (True, False):
                        start = time.perf_counter()
                        evaluate(mode)
                        timings[mode].append((time.perf_counter() - start) * 1000)
                medians = {
                    mode: statistics.median(times) for mode, times in timings.items()
                }
                memory = {}
                for mode in (False, True):
                    mx.synchronize()
                    mx.clear_cache()
                    before = mx.get_active_memory()
                    mx.reset_peak_memory()
                    output = evaluate(mode)
                    memory[mode] = dict(
                        active_before_bytes=before,
                        peak_extra_bytes=max(0, mx.get_peak_memory() - before),
                    )
                    del output
                row = dict(
                    batch=batch,
                    context=context,
                    distribution=distribution,
                    lengths=lengths,
                    sdpa_ms=medians[False],
                    pool_ms=medians[True],
                    speedup=medians[False] / medians[True],
                    max_abs_error=max_error,
                    sdpa_trials_ms=timings[False],
                    pool_trials_ms=timings[True],
                    sdpa_memory=memory[False],
                    pool_memory=memory[True],
                    padded_kv_bytes=2 * batch * context * kv_heads * dim * dtype.size,
                    pool_partial_bytes=batch
                    * heads
                    * ((context + 127) // 128)
                    * (dim + 1)
                    * 4,
                )
                records.append(row)
                print(
                    f"B={batch} context={context} {distribution}: "
                    f"SDPA {medians[False]:.3f} ms, pool {medians[True]:.3f} ms, "
                    f"{row['speedup']:.2f}x",
                    flush=True,
                )
                del (
                    evaluate,
                    forward,
                    reference,
                    result,
                    caches,
                    cache,
                    pool,
                    wrapper,
                    inner,
                    k,
                    v,
                )
                gc.collect()
                mx.clear_cache()

    report = dict(
        device=mx.device_info(),
        platform=platform.platform(),
        mlx=version("mlx"),
        mlx_lm=version("mlx-lm"),
        torch=version("torch"),
        mlx_enable_tf32=os.environ.get("MLX_ENABLE_TF32", "unset"),
        dtype=str(dtype),
        query_heads=heads,
        kv_heads=kv_heads,
        head_dim=dim,
        repeats=args.repeats,
        warmup=args.warmup,
        scope="single attention wrapper including batch metadata; not HTTP serving",
        memory_scope="one warmed MLX peak-minus-active sample per path; excludes allocator cache and system footprint",
        rows=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
