"""Compare indexed attention stages against the installed MLX singleton path."""

import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from sglang.kernels.ops.attention import mlx_ragged as r


def error(a, b):
    d = a.astype(mx.float32) - b.astype(mx.float32)
    return dict(
        equal=mx.array_equal(a, b).item(),
        max_abs=mx.max(mx.abs(d)).item(),
        rms=mx.sqrt(mx.mean(d * d)).item(),
    )


records = []
for dtype in (mx.float16, mx.bfloat16, mx.float32):
    for n in (17, 512, 2048, 2050):
        mx.random.seed(714)
        q = mx.random.normal((1, 64, 1, 512)).astype(dtype)
        kv = mx.random.normal((1, 4096, 512)).astype(dtype)
        idx = mx.random.permutation(4096)[:n].astype(mx.int32)[None, None]
        stages = []
        has_nax, tf32, _ = r._matmul_policy()
        nax = has_nax and (dtype != mx.float32 or tf32)
        kernels = r._kernels(nax, nax and dtype == mx.float32)

        def wrap(k):
            def call(**kw):
                out = k(**kw)
                stages.append(out[0])
                return out

            return call

        with patch.object(r, "_kernels", return_value=tuple(wrap(k) for k in kernels)):
            actual = r.mlx_ragged_latent_attention(q, kv, idx, [n], scale=256**-0.5)
        selected = mx.take(kv, idx[0, 0], axis=1)[:, None]
        scores = (q * mx.array(256**-0.5, dtype)) @ selected.swapaxes(-1, -2)
        probs = mx.softmax(scores, axis=-1, precise=True)
        values = probs @ selected
        native = mx.fast.scaled_dot_product_attention(
            q, selected, selected, scale=256**-0.5
        )
        row = dict(
            dtype=str(dtype),
            length=n,
            scores=error(stages[0], scores.reshape(1, 64, n)),
            softmax=error(stages[1], mx.softmax(stages[0], -1, precise=True)),
            values=error(actual, stages[1].reshape(1, 64, 1, n) @ selected),
            decomposed=error(native, values),
            final=error(actual, native),
        )
        records.append(row)
        print(json.dumps(row), flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
