# SPDX-License-Identifier: Apache-2.0
"""Paired FP8 projections retain codebook, scale and row-boundary semantics."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxBlockFp8(CustomTestCase):
    def test_encoding_midpoints_scales_and_signed_zero(self):
        """Codebook ties must round to even, including the subnormal boundary."""
        import mlx.core as mx
        import numpy as np
        from sglang.kernels.ops.quantization.mlx_fp8_dynamic import mlx_fp8_dynamic_32

        book = np.array(
            [
                m * 2**-9 if e == 0 else (1 + m / 8) * 2 ** (e - 7)
                for e in range(16)
                for m in range(8)
                if e * 8 + m < 127
            ],
            np.float32,
        )
        mid = (book[:-1] + book[1:]) / 2
        values = np.concatenate(
            (
                book,
                mid,
                np.nextafter(mid, -np.inf),
                np.nextafter(mid, np.inf),
                [0.0, -0.0, 448.0, 1000.0],
            )
        ).astype(np.float32)
        values = np.concatenate((values, -values))
        blocks = np.zeros((len(values), 32), np.float32)
        blocks[:, 0], blocks[:, 1] = values, 448
        for dtype in (mx.float32, mx.bfloat16):
            x = mx.array(blocks).astype(dtype)
            actual = np.array(x.astype(mx.float32))
            maximum = np.abs(actual).max(1)
            scales = np.exp2(np.ceil(np.log2(maximum / 448))).astype(np.float32)
            distance = np.abs(
                np.minimum(np.abs(actual[:, 0]) / scales, 448)[:, None] - book
            )
            expected = []
            for row in distance:
                candidates = np.flatnonzero(row == row.min())
                code = next((i for i in candidates if i % 2 == 0), candidates[0])
                expected.append(code)
            expected = np.array(expected, np.uint8)
            expected |= np.where(np.signbit(actual[:, 0]), 128, 0).astype(np.uint8)
            codes, actual_scales = mlx_fp8_dynamic_32(x)
            self.assertTrue(np.array_equal(np.array(codes)[:, 0], expected))
            self.assertTrue(np.array_equal(np.array(actual_scales).ravel(), scales))

    def test_pair_block_scales_tails_and_singleton_reduction(self):
        """A partial output block or row tile cannot alias the other matrix."""
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_block_fp8 import mlx_block_fp8_pair
        from sglang.kernels.ops.quantization.mlx_fp8_dynamic import mlx_fp8_dynamic_32

        mx.random.seed(20261103)
        # Independent scalar FP32 reduction with the native 32-lane tree.
        header = r"""
inline float decode(uchar code) {
    uint mag = uint(code) & 127, e = mag >> 3, m = mag & 7;
    float v = e == 0 ? float(m) / 512.0f :
        as_type<float>(((e+120)<<23)|(m<<20));
    return code & 128 ? -v : v;
}
"""
        kernel = mx.fast.metal_kernel(
            name="test_block_fp8_single_matrix",
            input_names=["w", "s", "q", "qs", "dims"],
            output_names=["out"],
            header=header,
            source=r"""
uint lane=thread_index_in_simdgroup, n=threadgroup_position_in_grid.x;
uint t=threadgroup_position_in_grid.y,K=dims[0],N=dims[1];
float acc=0;
for(uint b=0;b<K/128;++b)for(uint step=0;step<4;++step){
 uint k=b*128+step*32+lane;
 float xv=decode(q[t*K+k])*qs[t*(K/32)+k/32];
 float wv=decode(w[n*K+k])*s[(n/128)*(K/128)+b];
 acc+=xv*wv;
}
for(ushort off=16;off>=1;off>>=1)acc+=simd_shuffle_down(acc,off);
if(lane==0)out[t*N+n]=T(acc);
""",
        )
        for dtype in (mx.float32, mx.bfloat16):
            for count in (1, 3, 4, 5, 8, 17):
                n, k = 259, 256
                x = mx.random.normal((count, k)).astype(dtype)
                gw = mx.random.randint(0, 126, (n, k), dtype=mx.uint8)
                uw = mx.random.randint(128, 254, (n, k), dtype=mx.uint8)
                gs = mx.random.uniform(0.001, 0.02, (3, 2))
                us = mx.random.uniform(0.002, 0.03, (3, 2))
                # Real ModelOpt checkpoint scales include singleton axes.
                gs = gs.reshape(3, 1, 2, 1)
                us = us.reshape(3, 1, 2, 1)
                q = mlx_fp8_dynamic_32(x)
                expected = [
                    kernel(
                        inputs=[w, s.reshape(-1), *q, mx.array([k, n], mx.uint32)],
                        template=[("T", dtype)],
                        grid=(n * 32, count, 1),
                        threadgroup=(32, 1, 1),
                        output_shapes=[(count, n)],
                        output_dtypes=[dtype],
                    )[0]
                    for w, s in ((gw, gs), (uw, us))
                ]
                actual = mlx_block_fp8_pair(x, gw, gs, uw, us, q)
                for a, b in zip(expected, actual):
                    self.assertTrue(mx.array_equal(a, b).item())

        # Cover all signed finite codes and independently varying block scales.
        # A NaN code must propagate through its projection, not the other bank.
        x = mx.ones((1, 256), mx.float32)
        weight = mx.broadcast_to(mx.arange(256, dtype=mx.uint8), (259, 256))
        scales = mx.ones((3, 2), mx.float32)
        quant = mlx_fp8_dynamic_32(x)
        gate, up = mlx_block_fp8_pair(
            x, weight, scales, mx.zeros_like(weight), scales, quant
        )
        self.assertTrue(mx.all(mx.isnan(gate)).item())
        self.assertTrue(mx.all(up == 0).item())


if __name__ == "__main__":
    unittest.main()
