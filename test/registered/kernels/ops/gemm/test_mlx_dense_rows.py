# SPDX-License-Identifier: Apache-2.0
"""Dense shared-weight rows preserve each native GEMV's arithmetic."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxDenseRows(CustomTestCase):
    def test_independent_gate_banks_keep_native_split_k_and_rounding(self):
        """Narrow gates need BN=8 sums; wide paired outputs need BN=1 sums."""
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_dense_gates import (
            mlx_dense_gate_pair,
            mlx_dense_gate_rows,
        )

        mx.random.seed(711)
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            for k, sizes in ((4096, (128, 128, 64)), (256, (128, 128, 4))):
                weights = tuple(mx.random.normal((n, k)).astype(dtype) for n in sizes)
                for count in (1, 4, 5, 17):
                    # Strided inputs must be made contiguous once, with no
                    # cross-token addressing or output-tail contamination.
                    x = mx.random.normal((1, count, 2 * k)).astype(dtype)[..., ::2]
                    actual = mlx_dense_gate_rows(x, weights)
                    for got, w in zip(actual, weights):
                        expected = mx.concatenate(
                            [x[:, j : j + 1] @ w.T for j in range(count)], 1
                        )
                        self.assertTrue(mx.array_equal(got, expected).item())
            weights = tuple(
                mx.random.normal((8192, 128)).astype(dtype) for _ in range(2)
            )
            for count in (1, 4, 5, 17):
                inputs = tuple(
                    mx.random.normal((1, count, 128)).astype(dtype) for _ in range(2)
                )
                actual = mlx_dense_gate_pair(inputs, weights)
                for got, x, w in zip(actual, inputs, weights):
                    expected = mx.concatenate(
                        [x[:, j : j + 1] @ w.T for j in range(count)], 1
                    )
                    self.assertTrue(mx.array_equal(got, expected).item())

    def test_headed_scale_preserves_projection_rounding_and_row_layout(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_headed_dense_rows

        mx.random.seed(934)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            for transposed in (False, True):
                k, n = (256, 512) if transposed else (512, 128)
                shape = (3, k, n) if transposed else (3, n, k)
                weight = mx.random.normal(shape).astype(dtype)
                for count in (1, 4, 5, 17):
                    x = mx.random.normal((1, 3, count, k)).astype(dtype)
                    for scale in (0.0625, 128**-0.5):
                        expected = mlx_headed_dense_rows(
                            x, weight, transposed=transposed
                        ) * mx.array(scale, dtype)
                        actual = mlx_headed_dense_rows(
                            x, weight, transposed=transposed, output_scale=scale
                        )
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertTrue(
                            mx.array_equal(actual, expected).item(),
                            (dtype, transposed, count, scale),
                        )

    def test_batched_greedy_preserves_rounding_tail_and_ties(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_argmax_rows

        for dtype in (mx.float16, mx.bfloat16):
            for count in (1, 2, 3, 5, 9, 17):
                x = mx.random.normal((1, count, 256)).astype(dtype)
                weight = mx.random.normal((258, 256)).astype(dtype)
                expected = mx.concatenate(
                    [mx.argmax(x[:, j : j + 1] @ weight.T, -1) for j in range(count)], 1
                )
                actual = mlx_dense_argmax_rows(x, weight)
                self.assertTrue(mx.array_equal(actual, expected).item())
                self.assertTrue(
                    mx.array_equal(
                        mlx_dense_argmax_rows(x, mx.zeros_like(weight)),
                        mx.zeros((1, count), mx.uint32),
                    ).item()
                )

    def test_greedy_target_keeps_first_nan_and_finite_ties(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_argmax_rows

        for dtype in (mx.float16, mx.bfloat16):
            x = mx.ones((1, 3, 256), dtype)
            weight = mx.zeros((256, 256), dtype)
            weight[31, 0] = float("nan")
            weight[65, 0] = float("nan")
            expected = mx.argmax(
                mx.concatenate([x[:, j : j + 1] @ weight.T for j in range(3)], 1), -1
            )
            self.assertTrue(
                mx.array_equal(mlx_dense_argmax_rows(x, weight), expected).item()
            )

    def test_shared_loads_match_singletons_and_tail_rows(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows

        mx.random.seed(901)
        for dtype in (mx.float16, mx.bfloat16):
            for n, k in (
                (258, 256),
                (1026, 256),
                (4096, 512),
                (8192, 1024),
                (4096, 4096),
            ):
                weight = (mx.random.normal((n, k)) * 0.02).astype(dtype)
                for count in (1, 2, 3, 5, 9, 17):
                    with self.subTest(dtype=dtype, n=n, k=k, count=count):
                        x = mx.random.normal((1, count, k)).astype(dtype)
                        result = mlx_dense_rows(x, weight)
                        self.assertIsNotNone(result)
                        expected = mx.concatenate(
                            [x[:, j : j + 1] @ weight.T for j in range(count)], 1
                        )
                        self.assertTrue(mx.array_equal(result, expected).item())

    def test_kda_verification_rows_preserve_singleton_projection(self):
        """Wider row tiles must retain the large KDA bank's reduction order."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows

        mx.random.seed(20261102)
        for n, k in ((24896, 4096), (4096, 8192)):
            weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
            mx.eval(weight)
            for count in (4, 5, 6, 7, 8):
                x = mx.random.normal((1, count, k)).astype(mx.bfloat16)
                expected = mx.concatenate(
                    [mlx_dense_rows(x[:, j : j + 1], weight) for j in range(count)], 1
                )
                self.assertTrue(
                    mx.array_equal(mlx_dense_rows(x, weight), expected).item()
                )

    def test_strided_inputs_and_unsupported_reductions(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_rows

        mx.random.seed(902)
        weight = mx.random.normal((256, 258)).astype(mx.bfloat16)[:, :256]
        x = mx.random.normal((1, 6, 512)).astype(mx.bfloat16)[:, ::2, ::2]
        actual = mlx_dense_rows(x, weight)
        expected = mx.concatenate([x[:, j : j + 1] @ weight.T for j in range(3)], 1)
        self.assertTrue(mx.array_equal(actual, expected).item())
        for value, matrix in (
            (x, weight[:, :128]),
            (mx.zeros((1, 18, 256), mx.bfloat16), weight),
            (mx.zeros((2, 3, 256), mx.bfloat16), weight),
            (mx.zeros((1, 2, 256), mx.bfloat16), weight[:4]),
        ):
            with self.subTest(
                shape=value.shape, matrix=matrix.shape, dtype=value.dtype
            ):
                self.assertIsNone(mlx_dense_rows(value, matrix))

    def test_headed_tiles_preserve_singleton_and_tail_rows(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_headed_dense_rows

        mx.random.seed(20261103)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            for transposed in (False, True):
                w = mx.random.normal((64, 256, 512)).astype(mx.bfloat16)
                for count in (1, 4, 5, 6, 7, 8, 17):
                    x = mx.random.normal((1, 64, count, 512 if transposed else 256)).astype(
                        dtype
                    )
                    operand = w.swapaxes(-1, -2) if transposed else w
                    actual = mlx_headed_dense_rows(
                        x,
                        operand if transposed else operand.swapaxes(-1, -2),
                        transposed=transposed,
                    )
                    expected = mx.concatenate(
                        [
                            mlx_headed_dense_rows(
                                x[:, :, j : j + 1],
                                operand if transposed else operand.swapaxes(-1, -2),
                                transposed=transposed,
                            )
                            for j in range(count)
                        ],
                        2,
                    )
                    self.assertTrue(mx.array_equal(actual, expected).item())

    def test_mixed_projection_and_latent_singleton_reductions(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import (
            mlx_dense_rows,
            mlx_headed_dense_rows,
        )

        mx.random.seed(711)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            x = mx.random.normal((1, 3, 1536)).astype(dtype)
            w = mx.random.normal((4096, 1536)).astype(mx.bfloat16)
            expected = mx.concatenate([x[:, j : j + 1] @ w.T for j in range(3)], 1)
            actual = mlx_dense_rows(x, w)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(mx.array_equal(actual, expected).item())
            for k, n in ((256, 512), (512, 256)):
                x = mx.random.normal((1, 8, 3, k)).astype(dtype)
                w = mx.random.normal((8, k, n)).astype(dtype)
                expected = mx.concatenate([x[:, :, j : j + 1] @ w for j in range(3)], 2)
                actual = mlx_headed_dense_rows(x, w, transposed=True)
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_greedy_epilogue_matches_rounded_logits_and_first_tie(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_rows import mlx_dense_argmax

        mx.random.seed(92)
        for dtype in (mx.float16, mx.bfloat16):
            for n in (65, 256, 4097):
                x = mx.random.normal((1, 1, 256)).astype(dtype)
                w = mx.random.normal((n, 256)).astype(dtype)
                self.assertTrue(
                    mx.array_equal(
                        mlx_dense_argmax(x, w), mx.argmax(x @ w.T, -1)
                    ).item()
                )
                w = mx.zeros((n, 256), dtype)
                self.assertEqual(mlx_dense_argmax(x, w).item(), 0)
                w[31] = x[0, 0]
                w[32] = x[0, 0]
                self.assertEqual(mlx_dense_argmax(x, w).item(), 31)

    def test_nax_greedy_tensor_views_tail_nan_and_ties(self):
        """MPP head loads preserve views and rounded first-top1 semantics."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_dense_argmax_nax import (
            _mlx_dense_argmax_nax,
        )

        if not mx.device_info()["architecture"].startswith("applegpu_g17"):
            self.skipTest("requires M5 MPP tensor operations")
        n, k = 211, 256
        mx.random.seed(20261102)
        storage = mx.random.normal((n * 2, k * 2)).astype(mx.bfloat16)
        weight = storage[::2, ::2]
        for count in (8, 9, 16):
            inputs = mx.random.normal((1, count * 2, k * 2)).astype(mx.bfloat16)
            x = inputs[:, ::2, ::2]
            actual = _mlx_dense_argmax_nax(x, weight)
            expected = mx.argmax(x @ weight.T, -1)
            self.assertTrue(mx.array_equal(actual, expected).item(), count)
        x = mx.ones((1, 8, k), mx.bfloat16)
        # Integer-valued dots are exact in both reductions. Equal maxima
        # straddle output tiles, and NaNs must select their first index.
        weight = mx.zeros((n, k), mx.bfloat16)
        weight[35] = 1
        weight[71] = 1
        self.assertTrue(
            mx.array_equal(
                _mlx_dense_argmax_nax(x, weight),
                mx.full((1, 8), 35, mx.uint32),
            ).item()
        )
        weight[42, 0] = float("nan")
        weight[85, 0] = float("nan")
        self.assertTrue(
            mx.array_equal(
                _mlx_dense_argmax_nax(x, weight),
                mx.full((1, 8), 42, mx.uint32),
            ).item()
        )
        weight = mx.zeros((n, k), mx.bfloat16)
        self.assertTrue(
            mx.array_equal(
                _mlx_dense_argmax_nax(x, weight),
                mx.zeros((1, 8), mx.uint32),
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
