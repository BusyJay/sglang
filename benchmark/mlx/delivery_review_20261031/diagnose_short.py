import mlx.core as mx

mx.random.seed(15)
for dtype, vdtype in [
    (mx.float32, mx.bfloat16),
    (mx.bfloat16, mx.bfloat16),
    (mx.float32, mx.float32),
]:
    for lengths in ([2, 3, 4], [1, 2, 3], [6, 7, 8], [7, 8, 9]):
        w = max(lengths)
        q = mx.random.normal((1, 64, 3, 512)).astype(dtype)
        kv = mx.random.normal((1, w, 512)).astype(vdtype)
        selected = mx.broadcast_to(kv[:, None], (1, 3, w, 512)).reshape(3, 1, w, 512)
        qs = q.transpose(0, 2, 1, 3).reshape(3, 64, 1, 512)
        mask = mx.arange(w)[None, :] < mx.array(lengths)[:, None]
        out = mx.fast.scaled_dot_product_attention(
            qs, selected, selected, scale=256**-0.5, mask=mask[:, None, None, :]
        )
        ref = mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    q[:, :, j : j + 1],
                    kv[:, None, :n],
                    kv[:, None, :n],
                    scale=256**-0.5,
                )
                for j, n in enumerate(lengths)
            ],
            2,
        )
        out = out.reshape(1, 3, 64, 512).transpose(0, 2, 1, 3)
        print(
            str(dtype),
            lengths,
            mx.array_equal(out, ref).item(),
            mx.max(mx.abs(out - ref)).item(),
            flush=True,
        )
