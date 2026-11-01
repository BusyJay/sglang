# SPDX-License-Identifier: Apache-2.0
"""Batched affine rows retain native singleton QMV arithmetic."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxAffineRows(CustomTestCase):
    def test_multi_projection_independent_strides_and_promotion(self):
        """Independent banks must not inherit another bank's strides or dtype."""
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_affine_rows import (
            mlx_affine_mixed_rows,
            mlx_affine_multi_rows,
        )

        mx.random.seed(111)
        for count in (1, 2, 3, 5):
            x = mx.random.normal((1, count, 512)).astype(mx.bfloat16)
            parts, contiguous = [], []
            for n, dtype, bits, gs in (
                (32, mx.bfloat16, 4, 64),
                (36, mx.float16, 5, 32),
                (28, mx.bfloat16, 6, 128),
            ):
                bank = mx.quantize(
                    mx.random.normal((n * 2, 512 * 2)).astype(dtype),
                    bits=bits,
                    group_size=gs,
                )
                # Different row and packed/metadata strides, with nonzero offsets.
                w, s, b = bank
                w = w[1::2, ::2]
                s = s[1::2, ::2]
                b = b[1::2, ::2]
                # Put poison in distinct physical padding for every operand.
                w = mx.stack((mx.full(w.shape, 0xFFFFFFFF, mx.uint32), w), -1).reshape(
                    n, -1
                )[:, 1::2]
                s = mx.stack((s, mx.full(s.shape, float("nan"), s.dtype)), -1).reshape(
                    n, -1
                )[:, ::2]
                b = mx.stack(
                    (mx.full(b.shape, float("nan"), b.dtype), b, mx.zeros_like(b)), -1
                ).reshape(n, -1)[:, 1::3]
                parts.append((w, s, b, bits, gs))
                contiguous.append(tuple(mx.contiguous(v) for v in (w, s, b)))
            outputs = mlx_affine_multi_rows(x, parts)
            for actual, part, bank in zip(outputs, parts, contiguous):
                expected = mlx_affine_mixed_rows(
                    x, *bank, bits=part[3], group_size=part[4]
                )
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_four_bit_masks_preserve_native_rounding_and_duplicate_picks(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import (
            _grouped_gate_up_kernel,
            mlx_affine_routed_swiglu,
        )
        from sglang.kernels.ops.moe.mlx_glm import _group_picks

        for dtype in (mx.float16, mx.bfloat16):
            gate = mx.quantize(
                mx.random.normal((8, 32, 512)).astype(dtype), group_size=64, bits=4
            )
            up = mx.quantize(
                mx.random.normal((8, 32, 512)).astype(dtype), group_size=64, bits=4
            )
            for count in (2, 3, 5, 17):
                x = mx.random.normal((1, count, 512)).astype(dtype)
                ids = (mx.arange(count * 8).reshape(1, count, 8) % 4).astype(mx.uint32)
                grouped = _group_picks(ids, min(3, count))
                actual = mlx_affine_routed_swiglu(
                    x, gate, up, ids, bits=4, group_size=64, limit=10.0,
                    grouped=grouped,
                )
                expected = _grouped_gate_up_kernel()(
                    inputs=[x, *gate, *up, ids, mx.array([10.0], mx.float32), *grouped],
                    template=[
                        ("T", dtype), ("K", 512), ("N", 32), ("GS", 64),
                        ("BITS", 4), ("TOPK", 8), ("COUNT", count),
                        ("TILE", min(3, count)), ("GROUP_WIDTH", grouped[0].shape[1]), ("RPS", 2), ("HAS_LIMIT", True),
                    ],
                    grid=(128 * 4, count * 8, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(1, count, 8, 32)], output_dtypes=[dtype],
                )[0]
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_routed_gate_up_reads_independent_strided_views(self):
        """Prepared expert views and packed columns retain native arithmetic."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_routed_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(602)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                operands = []
                for stride in (2, 3):
                    values = mx.quantize(
                        (mx.random.normal((4, 32, 512)) * 0.02).astype(dtype),
                        group_size=64,
                        bits=bits,
                    )
                    views = []
                    for value in values:
                        storage = mx.zeros(
                            (4 * stride, 64, value.shape[-1] * stride), value.dtype
                        )
                        storage[::stride, ::2, ::stride] = value
                        views.append(storage[::stride, ::2, ::stride])
                    operands.append(tuple(views))
                for count in (1, 3, 4, 5, 6, 8, 17):
                    x = mx.random.normal((1, 2 * count, 1024)).astype(dtype)[
                        :, ::2, ::2
                    ]
                    table = mx.array([[[1, 3, 1, 0]] * (2 * count)], mx.uint32)
                    indices = table[:, ::2]
                    expected = []
                    for j in range(count):
                        row = x[:, j : j + 1, None, None, :]
                        gate, up = [
                            mx.gather_qmm(
                                row,
                                *values,
                                rhs_indices=indices[:, j : j + 1],
                                transpose=True,
                                group_size=64,
                                bits=bits,
                            ).squeeze(-2)
                            for values in operands
                        ]
                        expected.append(_compiled_swiglu(gate, up, 1.3))
                    actual = mlx_affine_routed_swiglu(
                        x, *operands, indices, bits=bits, group_size=64, limit=1.3
                    )
                    self.assertTrue(
                        mx.array_equal(actual, mx.concatenate(expected, 1)).item()
                    )

    def test_routed_prepared_views_do_not_copy_all_expert_weights(self):
        """Decode scratch must not scale with the resident expert bank."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_routed_swiglu

        packed = mx.quantize(
            mx.ones((64, 2048, 1024), mx.float16), bits=4, group_size=64
        )
        gate, up = (
            tuple(v[:, :1024] for v in packed),
            tuple(v[:, 1024:] for v in packed),
        )
        mx.eval(packed)
        for count in (1, 3):
            x = mx.ones((1, count, 1024), mx.float16)
            indices = mx.array([[[0, 3, 7, 11]] * count], mx.uint32)
            mx.eval(x, indices)
            mx.eval(
                mlx_affine_routed_swiglu(x, gate, up, indices, bits=4, group_size=64)
            )
            mx.synchronize()
            before = mx.get_active_memory()
            mx.reset_peak_memory()
            actual = mlx_affine_routed_swiglu(
                x, gate, up, indices, bits=4, group_size=64
            )
            mx.eval(actual)
            self.assertLess(mx.get_peak_memory() - before, 1024**2)

    def test_dense_and_routed_rows_match_native_singletons(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        mx.random.seed(123)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for count in (1, 3, 17):
                    with self.subTest(dtype=dtype, bits=bits, count=count):
                        x = mx.random.normal((1, count, 512)).astype(dtype)
                        w, s, b = mx.quantize(
                            (mx.random.normal((8, 256, 512)) * 0.02).astype(dtype),
                            group_size=64,
                            bits=bits,
                        )
                        indices = mx.array([[[1, 3, 7]] * count], mx.uint32)
                        for per_expert in (False, True):
                            act = (
                                mx.repeat(x[:, :, None], 3, axis=2) if per_expert else x
                            )
                            if per_expert:
                                act = (
                                    act
                                    + mx.array([0, 0.125, -0.25], dtype)[
                                        None, None, :, None
                                    ]
                                )
                            actual = mlx_affine_rows(
                                act,
                                w,
                                s,
                                b,
                                bits=bits,
                                group_size=64,
                                indices=indices,
                            )
                            expected = []
                            for j in range(count):
                                row = act[:, j : j + 1]
                                row = (
                                    row[..., None, :]
                                    if per_expert
                                    else row[:, :, None, None, :]
                                )
                                expected.append(
                                    mx.gather_qmm(
                                        row,
                                        w,
                                        s,
                                        b,
                                        rhs_indices=indices[:, j : j + 1],
                                        transpose=True,
                                        group_size=64,
                                        bits=bits,
                                    ).squeeze(-2)
                                )
                            self.assertTrue(
                                mx.array_equal(
                                    actual, mx.concatenate(expected, 1)
                                ).item()
                            )
                        actual = mlx_affine_rows(
                            x, w[0], s[0], b[0], bits=bits, group_size=64
                        )
                        expected = mx.concatenate(
                            [
                                mx.quantized_matmul(
                                    x[:, j : j + 1],
                                    w[0],
                                    s[0],
                                    b[0],
                                    transpose=True,
                                    group_size=64,
                                    bits=bits,
                                )
                                for j in range(count)
                            ],
                            1,
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_group_sizes_and_long_reductions_match_native(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        mx.random.seed(617)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for group, dim in ((32, 512), (64, 1536), (128, 4096)):
                    with self.subTest(dtype=dtype, bits=bits, group=group, dim=dim):
                        x = mx.random.normal((1, 3, dim)).astype(dtype)
                        w, s, b = mx.quantize(
                            (mx.random.normal((32, dim)) * 0.02).astype(dtype),
                            group_size=group,
                            bits=bits,
                        )
                        actual = mlx_affine_rows(
                            x, w, s, b, bits=bits, group_size=group
                        )
                        expected = mx.concatenate(
                            [
                                mx.quantized_matmul(
                                    x[:, j : j + 1],
                                    w,
                                    s,
                                    b,
                                    transpose=True,
                                    group_size=group,
                                    bits=bits,
                                )
                                for j in range(3)
                            ],
                            1,
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_unsupported_tiles_are_explicit(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_rows

        w, s, b = mx.quantize(mx.ones((16, 512), mx.float16), group_size=64, bits=4)
        for x in (
            mx.ones((2, 2, 512), mx.float16),
            mx.ones((1, 18, 512), mx.float16),
            mx.ones((1, 2, 512), mx.float32),
        ):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(mlx_affine_rows(x, w, s, b, bits=4, group_size=64))

    def test_shared_gate_up_epilogue_preserves_mixed_bits_and_clamp(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_swiglu_rows
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(85)
        for dtype in (mx.bfloat16, mx.float16):
            for bits in (4, 5, 6, 8):
                x = mx.random.normal((1, 9, 512)).astype(dtype)
                weights = [
                    mx.quantize(
                        (mx.random.normal((256, 512)) * 0.02).astype(dtype),
                        bits=bits,
                        group_size=64,
                    )
                    for _ in range(2)
                ]
                for count in (2, 3, 5, 9):
                    for limit in (None, 0.1):
                        row = x[:, :count]
                        gate, up = [
                            mx.concatenate(
                                [
                                    mx.quantized_matmul(
                                        row[:, j : j + 1],
                                        *w,
                                        transpose=True,
                                        bits=bits,
                                        group_size=64,
                                    )
                                    for j in range(count)
                                ],
                                1,
                            )
                            for w in weights
                        ]
                        actual = mlx_affine_swiglu_rows(
                            row, *weights, bits=bits, group_size=64, limit=limit
                        )
                        expected = _compiled_swiglu(gate, up, limit)
                        self.assertTrue(
                            mx.array_equal(actual, expected).item(),
                            (dtype, bits, count, limit),
                        )

    def test_mixed_metadata_preserves_native_promotion(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_mixed_rows

        mx.random.seed(712)
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            for bits in (4, 5, 6, 8):
                w, sc, bi = mx.quantize(
                    mx.random.normal((256, 512)).astype(mx.float16),
                    bits=bits,
                    group_size=64,
                )
                x = mx.random.normal((1, 3, 512)).astype(dtype)
                expected = mx.concatenate(
                    [
                        mx.quantized_matmul(
                            x[:, j : j + 1],
                            w,
                            sc,
                            bi,
                            bits=bits,
                            group_size=64,
                            transpose=True,
                        )
                        for j in range(3)
                    ],
                    1,
                )
                actual = mlx_affine_mixed_rows(x, w, sc, bi, bits=bits, group_size=64)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_routed_gate_up_epilogue_and_weight_replacement(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_affine_rows import mlx_affine_routed_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import clamped_swiglu

        mx.random.seed(713)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                x = mx.random.normal((1, 3, 512)).astype(dtype)
                gate = mx.quantize(
                    mx.random.normal((4, 256, 512)).astype(dtype),
                    bits=bits,
                    group_size=64,
                )
                indices = mx.array([[[1, 3, 3], [3, 1, 1], [0, 3, 1]]], mx.uint32)
                for _ in range(2):
                    up = mx.quantize(
                        mx.random.normal((4, 256, 512)).astype(dtype),
                        bits=bits,
                        group_size=64,
                    )

                    def project(arr, x=x, indices=indices, bits=bits):
                        return mx.gather_qmm(
                            x[..., None, None, :],
                            *arr,
                            rhs_indices=indices,
                            bits=bits,
                            group_size=64,
                            transpose=True,
                        ).squeeze(-2)

                    for limit in (None, 0.05):
                        expected = clamped_swiglu(project(gate), project(up), limit)
                        actual = mlx_affine_routed_swiglu(
                            x, gate, up, indices, bits=bits, group_size=64, limit=limit
                        )
                        self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
