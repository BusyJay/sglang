# SPDX-License-Identifier: Apache-2.0
"""Mixed expert dispatch preserves original checkpoint arithmetic and pick order."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxMixedMoE(CustomTestCase):
    def test_router_compaction_and_mixed_projection(self):
        import mlx.core as mx
        from sglang.kernels.ops.gemm.mlx_block_fp8 import mlx_block_fp8_pair
        from sglang.kernels.ops.gemm.mlx_fp8_rows import (
            mlx_fp8_dynamic_32,
            mlx_fp8_rows,
        )
        from sglang.kernels.ops.gemm.mlx_mixed_moe import (
            mlx_mixed_fp8_nvfp4_down,
            mlx_mixed_fp8_nvfp4_swiglu,
        )
        from sglang.kernels.ops.gemm.mlx_nvfp4_gather import mlx_nvfp4_gather_rows
        from sglang.kernels.ops.gemm.mlx_nvfp4_swiglu import mlx_nvfp4_swiglu
        from sglang.kernels.ops.moe.mlx_glm import mlx_glm_router_select
        from sglang.kernels.ops.moe.mlx_router_pairs import mlx_glm_router_select_pairs
        from sglang.srt.hardware_backend.mlx.models.glm5_next import clamped_swiglu

        for dtype in (mx.bfloat16, mx.float32):
            for rows in (1, 4, 8, 17):
                with self.subTest(dtype=dtype, rows=rows):
                    mx.random.seed(rows)
                    k, n, experts, topk = 512, 512, 16, 4
                    x = mx.random.normal((1, rows, k)).astype(dtype)
                    logits = mx.random.normal((1, rows, experts))
                    if rows == 4:
                        # Expert 0 occurs three times: a pair plus a singleton
                        # must coexist with independent experts in one launch.
                        logits = mx.full((1, rows, experts), -5.0, mx.float32)
                        for token, picks in enumerate(
                            ((0, 1, 2, 3), (0, 4, 5, 6), (0, 7, 8, 9), (10, 11, 12, 13))
                        ):
                            for expert in picks:
                                logits[0, token, expert] = 5.0
                    bias = mx.zeros((experts,), mx.float32)
                    selected = mlx_glm_router_select_pairs(
                        logits, bias, top_k=topk, scaling=2.5, renormalize=True
                    )
                    ordinary = mlx_glm_router_select(
                        logits, bias, top_k=topk, scaling=2.5, renormalize=True
                    )
                    mx.eval(selected, ordinary)
                    for got, want in zip(selected[:2], ordinary):
                        self.assertTrue(mx.array_equal(got, want).item())
                    ids, work = selected[0], selected[2:]
                    owners, partners, counts, total = work
                    if rows == 4:
                        self.assertIn(1, counts.tolist())
                        self.assertIn(2, counts.tolist())
                    visited = []
                    for a, b, count in zip(
                        owners.tolist()[: int(total.item())],
                        partners.tolist(),
                        counts.tolist(),
                    ):
                        visited.append(a)
                        if count == 2:
                            visited.append(b)
                            self.assertEqual(
                                ids.reshape(-1)[a].item(), ids.reshape(-1)[b].item()
                            )
                    self.assertEqual(sorted(visited), list(range(ids.size)))
                    gate = (
                        mx.random.randint(0, 256, (experts, n, k // 2), dtype=mx.uint8),
                        mx.full((experts, n, k // 16), 48, mx.uint8),
                        mx.full((experts,), 2688.0, mx.float32),
                    )
                    up = (
                        mx.random.randint(0, 256, (experts, n, k // 2), dtype=mx.uint8),
                        gate[1],
                        gate[2],
                    )
                    shared_gate = (
                        mx.random.randint(0, 127, (n, k), dtype=mx.uint8),
                        mx.full((n // 128, k // 128), 0.01, mx.float32),
                    )
                    shared_up = (
                        mx.random.randint(0, 127, (n, k), dtype=mx.uint8),
                        shared_gate[1],
                    )
                    quantized = mlx_fp8_dynamic_32(x)
                    for limit in (None, 0.25):
                        routed, shared = mlx_mixed_fp8_nvfp4_swiglu(
                            x,
                            gate,
                            up,
                            ids,
                            work,
                            shared_gate,
                            shared_up,
                            quantized,
                            limit=limit,
                        )
                        reference = mlx_nvfp4_swiglu(x, *gate, *up, ids, limit=limit)
                        sg, su = mlx_block_fp8_pair(
                            x, *shared_gate, *shared_up, quantized
                        )
                        expected_shared = clamped_swiglu(sg, su, limit)
                        mx.eval(routed, shared, reference, expected_shared)
                        self.assertTrue(mx.array_equal(routed, reference).item())
                        self.assertTrue(mx.array_equal(shared, expected_shared).item())
                        # Compact pair down must preserve original pick ordering.
                        want = mx.gather_qmm(
                            routed,
                            gate[0].view(mx.uint32),
                            gate[1],
                            rhs_indices=ids,
                            mode="nvfp4",
                            group_size=16,
                            bits=4,
                            global_scale=gate[2],
                        )
                        got = mlx_nvfp4_gather_rows(
                            routed, *gate, ids, worklist=work, output_rows=1
                        )
                        mx.eval(want, got)
                        self.assertTrue(mx.array_equal(got, want).item())
                        shared_quantized = mlx_fp8_dynamic_32(shared)
                        mixed_down, shared_down = mlx_mixed_fp8_nvfp4_down(
                            routed,
                            shared,
                            gate,
                            ids,
                            work,
                            shared_gate,
                            shared_quantized,
                        )
                        ordinary_shared = mlx_fp8_rows(
                            shared,
                            *shared_gate,
                            block_size=128,
                            quantized_input=shared_quantized,
                        )
                        mx.eval(mixed_down, shared_down, ordinary_shared)
                        self.assertTrue(mx.array_equal(mixed_down, want).item())
                        self.assertTrue(
                            mx.array_equal(shared_down, ordinary_shared).item()
                        )


if __name__ == "__main__":
    unittest.main()
