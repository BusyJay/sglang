"""Paired expert QMM must preserve native packed-loader and T epilogue rounding."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=15, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxQmmSwiGLU(CustomTestCase):
    def test_sparse_experts_preserve_native_gemv_rounding(self):
        """Sparse routing must use affine GEMV's float accumulation equation."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import mlx_sorted_qmm_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(215)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for k, count in (
                    (256, 1),
                    (512, 3),
                    (512, 17),
                    (256, 65),
                    (512, 65),
                    (4096, 208),
                ):
                    experts, n = 80, 64
                    operands = []
                    for stride in (2, 3):
                        values = mx.quantize(
                            (mx.random.normal((experts, n, k)) * 0.02).astype(dtype),
                            group_size=64,
                            bits=bits,
                        )
                        views = []
                        for value in values:
                            storage = mx.zeros(
                                (experts, n * 2, value.shape[-1] * stride), value.dtype
                            )
                            storage[:, ::2, ::stride] = value
                            views.append(storage[:, ::2, ::stride])
                        operands.append(tuple(views))
                    gate, up = operands
                    x = mx.random.normal((count, 1, k)).astype(dtype)
                    ids = mx.sort(mx.random.randint(0, experts, (count,))).astype(
                        mx.int32
                    )
                    g, u = [
                        mx.gather_qmm(
                            x,
                            *v,
                            rhs_indices=ids,
                            transpose=True,
                            group_size=64,
                            bits=bits,
                            sorted_indices=True,
                        )
                        for v in (gate, up)
                    ]
                    for limit in (None, 1.3):
                        actual = mlx_sorted_qmm_swiglu(
                            x, ids, gate, up, bits=bits, group_size=64, limit=limit
                        )
                        self.assertIsNotNone(actual)
                        self.assertTrue(
                            mx.array_equal(
                                actual, _compiled_swiglu(g, u, limit)
                            ).item(),
                            (dtype, bits, k, count, limit),
                        )

    def test_sparse_group_and_output_tails_match_native(self):
        """Partial output tiles and non-fast K blocks preserve affine group boundaries."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import mlx_sorted_qmm_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(216)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for group, k, n, count in (
                    (32, 96, 17, 3),
                    (64, 192, 31, 17),
                    (128, 384, 48, 65),
                    (128, 128, 65, 1),
                ):
                    operands = [
                        mx.quantize(
                            (mx.random.normal((80, n, k)) * 0.02).astype(dtype),
                            group_size=group,
                            bits=bits,
                        )
                        for _ in range(2)
                    ]
                    gate, up = operands
                    x = mx.random.normal((count, 1, k)).astype(dtype)
                    ids = mx.sort(mx.random.randint(0, 80, (count,))).astype(mx.int32)
                    g, u = [
                        mx.gather_qmm(
                            x,
                            *v,
                            rhs_indices=ids,
                            transpose=True,
                            group_size=group,
                            bits=bits,
                            sorted_indices=True,
                        )
                        for v in operands
                    ]
                    for limit in (None, 0.0, 1.3):
                        actual = mlx_sorted_qmm_swiglu(
                            x, ids, gate, up, bits=bits, group_size=group, limit=limit
                        )
                        self.assertIsNotNone(actual)
                        self.assertTrue(
                            mx.array_equal(
                                actual, _compiled_swiglu(g, u, limit)
                            ).item(),
                            (dtype, bits, group, k, n, count, limit),
                        )

    def test_strided_inputs_and_independent_expert_layouts(self):
        """Strided activations/ids and packed row views must read logical values."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import mlx_sorted_qmm_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(411)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                for experts, count in ((4, 65), (8, 65), (4, 257)):
                    operands = []
                    for stride in (2, 3):
                        packed = mx.quantize(
                            (mx.random.normal((experts, 128, 256)) * 0.02).astype(
                                dtype
                            ),
                            group_size=64,
                            bits=bits,
                        )
                        views = []
                        for value in packed:
                            storage = mx.zeros(
                                (experts * stride, 256, value.shape[-1] * stride),
                                value.dtype,
                            )
                            storage[::stride, ::2, ::stride] = value
                            views.append(storage[::stride, ::2, ::stride])
                        operands.append(tuple(views))
                    gate, up = operands
                    x = mx.random.normal((count * 2, 1, 512)).astype(dtype)[::2, :, ::2]
                    ids = mx.sort(mx.random.randint(0, experts, (count,))).astype(
                        mx.int32
                    )
                    table = mx.zeros((count * 2,), mx.int32)
                    table[::2] = ids
                    ids = table[::2]
                    g = mx.gather_qmm(
                        x,
                        *gate,
                        rhs_indices=ids,
                        transpose=True,
                        group_size=64,
                        bits=bits,
                        sorted_indices=True,
                    )
                    u = mx.gather_qmm(
                        x,
                        *up,
                        rhs_indices=ids,
                        transpose=True,
                        group_size=64,
                        bits=bits,
                        sorted_indices=True,
                    )
                    expected = _compiled_swiglu(g, u, 1.3)
                    for value in (x, mx.contiguous(x)):
                        actual = mlx_sorted_qmm_swiglu(
                            value, ids, gate, up, bits=bits, group_size=64, limit=1.3
                        )
                        self.assertIsNotNone(actual)
                        self.assertTrue(
                            mx.array_equal(actual, expected).item(),
                            (dtype, bits, count),
                        )

    def test_prepared_expert_views_preserve_strides_and_rounding(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import mlx_sorted_qmm_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                joined = mx.quantize(
                    mx.random.normal((8, 256, 256)).astype(dtype),
                    bits=bits,
                    group_size=64,
                )
                gate = tuple(v[:, :128] for v in joined)
                up = tuple(v[:, 128:] for v in joined)
                x = mx.random.normal((129, 1, 256)).astype(dtype)
                ids = mx.sort(mx.random.randint(0, 8, (129,))).astype(mx.int32)
                values = mx.gather_qmm(
                    x,
                    *joined,
                    rhs_indices=ids,
                    transpose=True,
                    group_size=64,
                    bits=bits,
                    sorted_indices=True,
                )
                g, u = mx.split(values, 2, -1)
                actual = mlx_sorted_qmm_swiglu(
                    x, ids, gate, up, bits=bits, group_size=64, limit=1.3
                )
                self.assertTrue(
                    mx.array_equal(actual, _compiled_swiglu(g, u, 1.3)).item()
                )

    def test_compact_worklist_preserves_expert_row_order_and_capacity(self):
        """Every sorted row belongs to exactly one bounded tile, including tails."""
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import _expert_worklist

        for rows, experts, bm in ((26, 288, 32), (2048, 288, 32), (257, 8, 64)):
            for active in (1, 8, experts):
                ids = mx.sort(mx.arange(rows) % active).astype(mx.int32)
                work, count = _expert_worklist(ids, experts, rows, bm)
                mx.eval(work, count)
                self.assertLessEqual(count.item(), (rows + bm - 1) // bm + experts - 1)
                covered = []
                values = ids.tolist()
                for expert, start, end in work[:count.item()].tolist():
                    stop = min(start + bm, end)
                    self.assertGreater(stop, start)
                    self.assertTrue(all(values[row] == expert for row in range(start, stop)))
                    covered.extend(range(start, stop))
                self.assertEqual(covered, list(range(rows)))

    def test_sorted_paired_qmm_matches_native_epilogue(self):
        import mlx.core as mx

        from sglang.kernels.ops.gemm.mlx_qmm_swiglu import mlx_sorted_qmm_swiglu
        from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_swiglu

        mx.random.seed(91)
        for dtype in (mx.float16, mx.bfloat16):
            for bits in (4, 5, 6, 8):
                gate = mx.quantize(
                    (mx.random.normal((8, 128, 256)) * 0.02).astype(dtype),
                    bits=bits,
                    group_size=64,
                )
                up = mx.quantize(
                    (mx.random.normal((8, 128, 256)) * 0.02).astype(dtype),
                    bits=bits,
                    group_size=64,
                )
                joined = [mx.concatenate([a, b], 1) for a, b in zip(gate, up)]
                for count in (65, 129, 257):
                    x = mx.random.normal((count, 1, 256)).astype(dtype)
                    ids = mx.sort(mx.random.randint(0, 8, (count,))).astype(mx.int32)
                    values = mx.gather_qmm(
                        x,
                        *joined,
                        rhs_indices=ids,
                        transpose=True,
                        group_size=64,
                        bits=bits,
                        sorted_indices=True,
                    )
                    g, u = mx.split(values, 2, -1)
                    for limit in (None, 1.3):
                        actual = mlx_sorted_qmm_swiglu(
                            x, ids, gate, up, bits=bits, group_size=64, limit=limit
                        )
                        self.assertIsNotNone(actual)
                        expected = _compiled_swiglu(g, u, limit)
                        self.assertTrue(
                            mx.array_equal(actual, expected).item(),
                            (dtype, bits, count, limit),
                        )


if __name__ == "__main__":
    unittest.main()
