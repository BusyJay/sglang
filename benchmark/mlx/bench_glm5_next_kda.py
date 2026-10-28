"""Experimental block-parallel bounded KDA; not enabled in the server.

Uses random inputs at production dimensions and compares outputs/state against
mlx-lm's recurrent Metal kernel. The 32-token FP32 block solver avoids CPU solves.
No checkpoint weights are loaded. This probe measures the recurrence only.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import (
    compute_lower_bound_g,
    gated_delta_update,
    normalize_qk,
)


@mx.compile
def _kda_chunk_step(qg, decayed_k, causal_qk, u, w, decay, state):
    projected = mx.concatenate([w, qg], axis=-2) @ state.swapaxes(-1, -2)
    correction, base = mx.split(projected, 2, axis=-2)
    delta = u - correction
    out = base + causal_qk @ delta
    state = state * decay[..., None, :] + delta.swapaxes(-1, -2) @ decayed_k
    return out, state


@mx.compile
def _kda_chunk_prepare(qf, kf, vf, gf, beta):
    block = qf.shape[-2]
    log_decay = mx.cumsum(mx.log(gf), axis=-2)
    # Center exponents: with 32 tokens and log gates in [-5, 0], each
    # transformed operand stays in the normal FP32 range (exponents +/-80).
    shift = log_decay[..., -1:, :] * 0.5
    left = mx.exp(log_decay - shift)
    right = mx.exp(shift - log_decay)
    lower = mx.tril(((kf * left) @ (kf * right).swapaxes(-1, -2)) * beta, k=-1)
    power = -lower
    inverse = mx.eye(block) + power
    for _ in range(4):
        power = power @ power
        inverse = inverse + power @ inverse
    u = inverse @ (beta * vf)
    w = (inverse @ (beta * kf * left)) * mx.exp(shift)
    causal_qk = mx.tril((qf * left) @ (kf * right).swapaxes(-1, -2))
    qg = qf * mx.exp(log_decay)
    decayed_k = kf * mx.exp(log_decay[..., -1:, :] - log_decay)
    return qg, decayed_k, causal_qk, u, w, mx.exp(log_decay[..., -1, :])


def _chunk_kda_update(q, k, v, a, b, A_log, dt_bias, state, lower_bound):
    """Solve bounded vector-gated delta-rule blocks, then propagate block states.

    The innovations satisfy a unit-lower-triangular system with
    L_ij = beta_i <k_i G_i, k_j / G_j> (j < i). Its finite Neumann
    series is evaluated by doubling, with centered cumulative gate factors.
    State/block algebra uses FP32; outputs return to the activation dtype.
    """
    batch, length, heads, dim = q.shape
    block = 32
    padding = (-length) % block
    g = compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    beta = mx.sigmoid(b).astype(mx.float32)
    arrays = [q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), g]
    if padding:
        arrays = [
            mx.pad(x, ((0, 0), (0, padding), (0, 0), (0, 0)), constant_values=fill)
            for x, fill in zip(arrays, (0, 0, 0, 1))
        ]
        beta = mx.pad(beta, ((0, 0), (0, padding), (0, 0)))
    chunks = (length + padding) // block
    qf, kf, vf, gf = [
        x.transpose(0, 2, 1, 3).reshape(batch, heads, chunks, block, -1) for x in arrays
    ]
    beta = beta.transpose(0, 2, 1).reshape(batch, heads, chunks, block, 1)
    qg, decayed_k, causal_qk, u, w, decay = _kda_chunk_prepare(qf, kf, vf, gf, beta)
    if state is None:
        state = mx.zeros((batch, heads, v.shape[-1], dim), dtype=mx.float32)
    outputs = []
    for i in range(chunks):
        out, state = _kda_chunk_step(
            qg[:, :, i],
            decayed_k[:, :, i],
            causal_qk[:, :, i],
            u[:, :, i],
            w[:, :, i],
            decay[:, :, i],
            state,
        )
        outputs.append(out)
    out = mx.stack(outputs, axis=2).reshape(batch, heads, length + padding, -1)
    return out[:, :, :length].transpose(0, 2, 1, 3).astype(q.dtype), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not all(t > 0 for t in args.query_lengths):
        parser.error("Require positive repeats and query lengths")
    mx.random.seed(39)
    heads, dim = 64, 128
    rows = []
    for length in args.query_lengths:
        q, k, v, a = [
            mx.random.normal((1, length, heads, dim)).astype(mx.bfloat16)
            for _ in range(4)
        ]
        q, k = normalize_qk(q, k, inv_scale=dim**-0.5, eps=1e-6)
        b = mx.random.normal((1, length, heads)).astype(mx.bfloat16)
        A, dt = mx.zeros((heads, 1)), mx.zeros((heads, dim))
        state = mx.random.normal((1, heads, dim, dim)) * 0.1
        mx.eval(q, k, v, a, b, A, dt, state)
        times = {"recurrent": [], "block32": []}
        final = {}
        for trial in range(args.repeats + 1):
            order = (
                ("recurrent", "block32") if trial % 2 == 0 else ("block32", "recurrent")
            )
            for label in order:
                fn = gated_delta_update if label == "recurrent" else _chunk_kda_update
                start = time.perf_counter()
                y, next_state = fn(q, k, v, a, b, A, dt, state=state, lower_bound=-5.0)
                mx.eval(y, next_state)
                if trial:
                    times[label].append(time.perf_counter() - start)
                final[label] = (y, next_state)
        errors = [
            float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
            for x, y in zip(final["recurrent"], final["block32"])
        ]
        if (
            not all(
                float(mx.all(mx.isfinite(x))) for pair in final.values() for x in pair
            )
            or max(errors) > 0.01
        ):
            raise AssertionError(
                f"Block/recurrent numerical comparison failed: {errors}"
            )
        row = {
            "query_tokens": length,
            "recurrent_samples_s": times["recurrent"],
            "block32_samples_s": times["block32"],
            "recurrent_median_s": statistics.median(times["recurrent"]),
            "block32_median_s": statistics.median(times["block32"]),
            "max_output_abs_error": errors[0],
            "max_state_abs_error": errors[1],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "method": "Random BF16, 64 heads x 128 dimensions, bounded vector log gates [-5, 0], nonzero FP32 initial state, alternating timed pairs after one warmup pair. No full-model speed or quality claim.",
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "serving_default": "mlx-lm recurrent Metal kernel; block32 remains experimental",
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
