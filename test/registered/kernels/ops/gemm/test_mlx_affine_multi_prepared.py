# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=8, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestPreparedMultiAffine(CustomTestCase):
    def test_independent_views_and_explicit_compiled_replacement(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_affine_rows import (
            mlx_affine_multi_prepare_strides,
            mlx_affine_multi_rows,
        )

        mx.random.seed(113)
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            parts = []
            for bits, stride in ((4, 2), (5, 3), (6, 2)):
                q = mx.quantize(
                    mx.random.normal((8, 512)).astype(dtype), group_size=64, bits=bits
                )
                views = []
                for a in q:
                    storage = mx.zeros((16, a.shape[1] * stride), a.dtype)
                    storage[::2, ::stride] = a
                    views.append(storage[::2, ::stride])
                parts.append((*views, bits, 64))
            strides = mlx_affine_multi_prepare_strides(parts)

            def make_run(parts, strides):
                def evaluate_rows(value, operands):
                    live = [(*v, p[3], p[4]) for v, p in zip(operands, parts)]
                    return mlx_affine_multi_rows(value, live, prepared_strides=strides)

                return evaluate_rows

            run = make_run(parts, strides)

            compiled = mx.compile(run)
            for rows in (1, 3, 5, 17):
                x = mx.random.normal((1, rows, 512)).astype(dtype)
                arrays = [p[:3] for p in parts]
                for actual in (run(x, arrays), compiled(x, arrays)):
                    expected = mlx_affine_multi_rows(x, parts)
                    self.assertTrue(
                        all(
                            mx.array_equal(a, b).item()
                            for a, b in zip(actual, expected)
                        )
                    )
                # Same prepared layout, changed values are explicit compiled inputs.
                arrays = [(p[0], p[1] * 2, p[2] + 1) for p in parts]
                changed = [(*v, p[3], p[4]) for v, p in zip(arrays, parts)]
                fresh = mlx_affine_multi_prepare_strides(changed)

                def make_replaced(parts, fresh):
                    def evaluate_replaced(value, operands):
                        return mlx_affine_multi_rows(
                            value,
                            [(*v, p[3], p[4]) for v, p in zip(operands, parts)],
                            prepared_strides=fresh,
                        )

                    return evaluate_replaced

                replaced = make_replaced(parts, fresh)

                actual = mx.compile(replaced)(x, arrays)
                expected = mlx_affine_multi_rows(x, changed)
                self.assertTrue(
                    all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))
                )


if __name__ == "__main__":
    unittest.main()
