# SPDX-License-Identifier: Apache-2.0
"""TRTLLM scratch reuse across sequential target/draft shapes, including graphs."""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires SM12x TRTLLM-gen",
)
class TestQsaWorkspaceGpu(CustomTestCase):
    def test_interleaved_shapes_and_graph_replay(self):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        torch.manual_seed(191)
        calls = []
        for batch in (3, 1, 3):
            q = torch.randn(batch, 24, 256, device="cuda", dtype=torch.bfloat16)
            kv = tuple(
                torch.randn(batch * 33, 2, 64, 256, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            )
            table = torch.arange(batch * 33, device="cuda", dtype=torch.int32).reshape(
                batch, 33
            )
            lens = torch.tensor(
                ([17, 2048, 2051] if batch == 3 else [2051]),
                device="cuda",
                dtype=torch.int32,
            )
            counters = torch.zeros(
                get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    batch, 24, torch.cuda.get_device_properties(0).multi_processor_count
                ),
                device="cuda",
                dtype=torch.uint8,
            )
            calls.append((q, kv, table, lens, counters))

        def run(call, workspace):
            q, kv, table, lens, counters = call
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv,
                workspace_buffer=workspace,
                block_tables=table,
                seq_lens=lens,
                max_seq_len=2112,
                bmm1_scale=1 / math.sqrt(256),
                multi_ctas_kv_counter_buffer=counters,
            )

        private = [
            torch.zeros(128 << 20, device="cuda", dtype=torch.uint8) for _ in calls
        ]
        expected = [
            run(call, workspace).clone() for call, workspace in zip(calls, private)
        ]
        shared = torch.zeros_like(private[0])
        for _ in range(3):
            for call, reference in zip(calls, expected):
                actual = run(call, shared)
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [run(call, shared) for call in calls]
        for _ in range(3):
            graph.replay()
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
