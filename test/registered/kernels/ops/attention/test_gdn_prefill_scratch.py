# SPDX-License-Identifier: Apache-2.0
"""Borrowed GDN prefill storage preserves outputs, chunk states and final states."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGdnPrefillScratch(CustomTestCase):
    def compare(self, tokens, hq, hv, kdim, vdim, split, *, small=False, alias=False):
        torch.manual_seed(872)
        q = torch.randn(1, tokens, hq, kdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, hv, vdim, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, tokens, hv, device="cuda") * 0.1
        beta = torch.rand(1, tokens, hv, device="cuda", dtype=torch.bfloat16)
        states = (torch.randn(4, hv, vdim, kdim, device="cuda") * 0.01).bfloat16()
        cu = torch.tensor([0, split, tokens], dtype=torch.int32, device="cuda")
        indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
        chunks = (split + 63) // 64 + (tokens - split + 63) // 64
        size = chunks * hv * vdim * kdim + tokens * hv * kdim
        scratch = torch.empty(1 if small else size, device="cuda", dtype=torch.bfloat16)

        def run(borrow):
            inputs = [t.clone() for t in (q, k, v)]
            if alias:
                inputs[1] = inputs[0]
            state = states.clone()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output, _, chunk_states = chunk_gated_delta_rule(
                *inputs,
                g,
                beta,
                initial_state=state,
                initial_state_indices=indices,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                prefill_scratch=scratch if borrow else None,
            )
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            return output, chunk_states, state, peak, inputs

        # Warm compilation before measuring allocator peaks.
        run(False)
        run(True)
        reference = run(False)
        actual = run(True)
        for got, expected in zip(actual[:3], reference[:3]):
            self.assertTrue(torch.isfinite(got).all().item())
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        if small:
            self.assertNotEqual(actual[1].data_ptr(), scratch.data_ptr())
            torch.testing.assert_close(actual[4][0], q, rtol=0, atol=0)
            torch.testing.assert_close(actual[4][1], k, rtol=0, atol=0)
        else:
            self.assertEqual(actual[1].data_ptr(), scratch.data_ptr())
            if kdim >= vdim:
                self.assertGreaterEqual(actual[0].data_ptr(), scratch.data_ptr())
                self.assertLess(
                    actual[0].data_ptr(), scratch.data_ptr() + scratch.nbytes
                )
        if alias:
            torch.testing.assert_close(actual[4][0], q, rtol=0, atol=0)
        if tokens == 8192:
            self.assertLess(actual[3], reference[3] - (256 << 20))
            print(
                "GDN_PREFILL_PEAK",
                {
                    "reference": reference[3],
                    "borrowed": actual[3],
                    "saved": reference[3] - actual[3],
                },
                flush=True,
            )

    def test_rectangular_varlen_and_fallback(self):
        self.compare(257, 2, 6, 64, 128, 127)
        self.compare(257, 2, 6, 64, 128, 127, small=True)

    def test_production_chunk(self):
        self.compare(8192, 16, 48, 128, 128, 4096)

    def test_aliased_query_key(self):
        self.compare(257, 2, 6, 64, 128, 127, alias=True)

    def test_graph_reuses_verify_storage(self):
        torch.manual_seed(571)
        q, k, v = [
            torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        g = -torch.rand(1, 256, 2, device="cuda") * 0.1
        beta = torch.rand(1, 256, 2, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(2, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
        state = torch.empty_like(initial)
        inputs = [torch.empty_like(t) for t in (q, k, v)]
        cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        indices = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        scratch = torch.empty(
            4 * 2 * 64 * 64 + 256 * 2 * 64, device="cuda", dtype=torch.bfloat16
        )

        def execute():
            state.copy_(initial)
            inputs[0].copy_(q)
            inputs[1].copy_(k)
            inputs[2].copy_(v)
            return chunk_gated_delta_rule(
                *inputs,
                g,
                beta,
                initial_state=state,
                initial_state_indices=indices,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                prefill_scratch=scratch,
            )

        execute()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, _, h = execute()
        for _ in range(3):
            # Simulate verify overwriting the shared buffer before prefill.
            scratch.fill_(float("nan"))
            graph.replay()
            reference_state = initial.clone()
            expected, _, expected_h = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=reference_state,
                initial_state_indices=indices,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
            torch.testing.assert_close(state, reference_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
