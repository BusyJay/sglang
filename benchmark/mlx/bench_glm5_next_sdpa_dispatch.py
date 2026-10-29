"""Audit GLM's D512 decode dispatch and compare equivalent MLX SDPA layouts.

force_fused is diagnostic only: eligibility does not establish a speedup. The
matrix fallback also runs on Metal and can use accelerated matrix multiplication.
Run with MLX_ENABLE_TF32 explicitly set, with no other GPU workload running.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def measure(functions, inputs, rounds, iterations, mx):
    compiled, timings, eligibility, errors = {}, {}, {}, {}
    reference = functions["original"](*inputs)
    mx.eval(reference)
    for name, function in functions.items():
        try:
            fn = mx.compile(function)
            result = fn(*inputs)
            mx.eval(result)
            errors[name] = mx.max(
                mx.abs(result.astype(mx.float32) - reference.astype(mx.float32))
            ).item()
            for _ in range(3):
                mx.eval(fn(*inputs))
            compiled[name], timings[name] = fn, []
            eligibility[name] = "accepted"
        except ValueError as exc:
            eligibility[name] = str(exc)
    for trial in range(rounds):
        names = list(compiled)
        if trial % 2:
            names.reverse()
        for name in names:
            mx.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                mx.async_eval(compiled[name](*inputs))
            mx.synchronize()
            timings[name].append((time.perf_counter() - start) * 1000 / iterations)
    return {
        "eligibility": eligibility,
        "median_ms": {name: statistics.median(t) for name, t in timings.items()},
        "trials_ms": timings,
        "max_abs_difference_from_original": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[32, 1024, 2048, 2051, 4096]
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.lengths, args.rounds, args.iterations) < 1:
        parser.error("Lengths, rounds and iterations must be positive")
    import mlx.core as mx

    mx.random.seed(42)
    rows, padding = [], []
    scale = 128**-0.5
    for dtype in (mx.bfloat16, mx.float32):
        for length in args.lengths:
            q = mx.random.normal((1, 64, 1, 512)).astype(dtype)
            kv = mx.random.normal((1, 1, length, 512)).astype(dtype)
            mx.eval(q, kv)

            def original(q, kv):
                return mx.fast.scaled_dot_product_attention(q, kv, kv, scale=scale)

            def fused_original(q, kv):
                return mx.fast.scaled_dot_product_attention(
                    q, kv, kv, scale=scale, force_fused=True
                )

            def fused_vector(q, kv):
                shared = mx.broadcast_to(kv, (8, 1, kv.shape[2], 512))
                return mx.fast.scaled_dot_product_attention(
                    q.reshape(8, 8, 1, 512),
                    shared,
                    shared,
                    scale=scale,
                    force_fused=True,
                ).reshape(q.shape)

            def fused_matrix(q, kv):
                return mx.fast.scaled_dot_product_attention(
                    q.reshape(1, 1, 64, 512), kv, kv, scale=scale, force_fused=True
                ).reshape(q.shape)

            row = {"dtype": str(dtype), "key_length": length}
            row.update(
                measure(
                    {
                        "original": original,
                        "fused_original": fused_original,
                        "fused_vector": fused_vector,
                        "fused_matrix": fused_matrix,
                    },
                    (q, kv),
                    args.rounds,
                    args.iterations,
                    mx,
                )
            )
            rows.append(row)
        for tail in range(4):
            valid_length = 2048 + tail
            q = mx.random.normal((1, 64, 1, 512)).astype(dtype)
            kv = mx.random.normal((1, 1, 2051, 512)).astype(dtype)
            mask = mx.arange(2051) < valid_length
            mx.eval(q, kv, mask)

            def padded(q, kv, mask):
                return mx.fast.scaled_dot_product_attention(
                    q, kv, kv, scale=scale, mask=mask
                )

            def trimmed(q, kv, mask, valid_length=valid_length):
                selected = kv[:, :, :valid_length]
                return mx.fast.scaled_dot_product_attention(
                    q, selected, selected, scale=scale
                )

            row = {"dtype": str(dtype), "valid_key_length": valid_length}
            row.update(
                measure(
                    {"original": padded, "trimmed": trimmed},
                    (q, kv, mask),
                    args.rounds,
                    args.iterations,
                    mx,
                )
            )
            padding.append(row)
    report = {
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "method": "Identical random inputs, compiled calls, three warmups; alternating candidate order across rounds, asynchronous submissions followed by synchronization. Measures amortized operator latency, not serving throughput.",
        "shapes": {
            "query": [1, 64, 1, 512],
            "kv_heads": 1,
            "sparse_padded_width": 2051,
        },
        "rounds": args.rounds,
        "iterations": args.iterations,
        "dispatch": rows,
        "padding": padding,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
