"""Prepared launch must rebind changing cache storage and dynamic route inputs.

A stale pointer cache or incorrect vendor argument order silently computes using
old experts. CUDA compile/execute is mocked; descriptor selection is real.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.quantization.qwen38_cached_b12x import _PreparedSmallMLaunch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPreparedLaunch(CustomTestCase):
    def test_dynamic_inputs_cache_replacement_and_bounded_bindings(self):
        calls, pointers = [], {}

        def make_ptr(dtype, address, space, assumed_align):
            return address

        def execute(*args):
            # Assert external ABI positions and compute using current inputs.
            self.assertEqual(args[3], args[4])
            self.assertEqual(args[5], args[9])
            self.assertEqual(pointers[args[1]].dtype, torch.uint8)
            self.assertEqual(pointers[args[10]].dtype, torch.int64)
            pointers[args[12]].copy_(pointers[args[0]] + pointers[args[11]].sum())
            calls.append(args)

        kernel = SimpleNamespace(
            make_ptr=make_ptr,
            cute=SimpleNamespace(AddressSpace=SimpleNamespace(gmem=0)),
            cutlass=SimpleNamespace(
                Uint8=0, Float32=1, Uint32=2, Int64=3, Int32=4, BFloat16=5
            ),
            Int32=int,
            cuda=SimpleNamespace(CUstream=int),
            _small_m_direct_host_barrier_reset_enabled=lambda: True,
            _compile_w4a16_small_m_direct=Mock(
                return_value=SimpleNamespace(compiled=execute, grid_x=10)
            ),
        )
        workspace = torch.ones(10, dtype=torch.int32)
        prepared = _PreparedSmallMLaunch(
            kernel, workspace, torch.empty(4096, dtype=torch.bfloat16)
        )
        weights = dict(
            w13_weight=torch.empty(2, 64, 64, dtype=torch.uint8),
            w2_weight=torch.empty(2, 128, 16, dtype=torch.uint8),
            w13_blockscale_swizzled=torch.empty(2, 128, 8, dtype=torch.uint8),
            w2_blockscale_swizzled=torch.empty(2, 128, 4, dtype=torch.uint8),
        )
        g1, g2 = torch.ones(2), torch.ones(2)
        held = []
        for i in range(12):
            if i > 1:
                held.extend(weights.values())  # Prevent allocator address reuse.
                weights = {name: value.clone() for name, value in weights.items()}
            x = torch.full((3, 128), float(i), dtype=torch.bfloat16)
            ids = torch.full((3, 10), i % 2, dtype=torch.int64)
            router = torch.full((3, 10), float(i))
            output = torch.empty_like(x)
            for value in (*weights.values(), g1, g2, x, ids, router, output):
                pointers[value.data_ptr()] = value
            actual = prepared.run(
                x, ids, router, weights, g1, g2, output, SimpleNamespace(cuda_stream=7)
            )
            torch.testing.assert_close(actual, torch.full_like(x, 31.0 * i))
            self.assertEqual(calls[-1][1], weights["w13_weight"].data_ptr())
            self.assertEqual(calls[-1][10], ids.data_ptr())
            self.assertEqual(workspace[-2:].tolist(), [0, 0])
        self.assertEqual(len(prepared.bindings), 8)
        # First two calls share weights/geometry; subsequent replacements rebind.
        self.assertEqual(kernel._compile_w4a16_small_m_direct.call_count, 11)
        for launch, fixed, tail in prepared.bindings.values():
            self.assertFalse(any(isinstance(value, torch.Tensor) for value in fixed))

    def test_geometry_and_id_dtype_change_use_distinct_launch(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.override(True):
            self.assertTrue(envs.SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH.get())
        # The keys include M, top-k, dtype and capacity; exercise actual rebinding.
        seen = []
        k = SimpleNamespace(
            make_ptr=lambda dtype, address, *args, **kw: address,
            cute=SimpleNamespace(AddressSpace=SimpleNamespace(gmem=0)),
            cutlass=SimpleNamespace(
                Uint8=0, Float32=1, Uint32=2, Int64=3, Int32=4, BFloat16=5
            ),
            Int32=int,
            cuda=SimpleNamespace(CUstream=int),
            _small_m_direct_host_barrier_reset_enabled=lambda: True,
        )

        def compile(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(compiled=lambda *args: None, grid_x=1)

        k._compile_w4a16_small_m_direct = compile
        runner = _PreparedSmallMLaunch(
            k,
            torch.zeros(10, dtype=torch.int32),
            torch.empty(4096, dtype=torch.bfloat16),
        )
        weights = {
            name: torch.empty(4, 128, 64, dtype=torch.uint8)
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_blockscale_swizzled",
                "w2_blockscale_swizzled",
            )
        }
        g1, g2 = torch.ones(4), torch.ones(4)
        for m, topk, dtype, cap in (
            (1, 10, torch.int32, 2),
            (3, 10, torch.int32, 2),
            (3, 10, torch.int64, 2),
            (3, 4, torch.int64, 4),
        ):
            t = {name: value[:cap] for name, value in weights.items()}
            x = torch.empty(m, 128, dtype=torch.bfloat16)
            runner.run(
                x,
                torch.empty(m, topk, dtype=dtype),
                torch.empty(m, topk),
                t,
                g1[:cap],
                g2[:cap],
                torch.empty_like(x),
                SimpleNamespace(cuda_stream=0),
            )
        self.assertEqual(
            [(s["m"], s["topk"], s["topk_ids_dtype"], s["num_experts"]) for s in seen],
            [
                (1, 10, torch.int32, 2),
                (3, 10, torch.int32, 2),
                (3, 10, torch.int64, 2),
                (3, 4, torch.int64, 4),
            ],
        )


if __name__ == "__main__":
    unittest.main()
