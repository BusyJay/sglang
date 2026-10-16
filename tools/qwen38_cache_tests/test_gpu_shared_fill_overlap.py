"""Parity of shared GEMV/direct fills across streams and breakable replay."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestGpuSharedFillOverlap(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_parallel

        self.enterContext(get_parallel().override(tp_size=1))

    def _fixture(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        from .test_cpu_runtime import TestSynchronousRuntime

        TestSynchronousRuntime.setUpClass()
        fixture = TestSynchronousRuntime()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        runtime = Qwen38ThreadedExpertRuntime(
            fixture.pack,
            "cuda",
            slots_per_layer=2,
            max_pending=8,
            adaptive_cache=True,
            adaptive_cache_window=16,
            adaptive_cache_interval=16,
            adaptive_cache_mode="sample",
        )
        runtime.reserve()
        self.addCleanup(runtime.close)
        return fixture, runtime

    def _block(self, runtime):
        from sglang.srt.layers.moe.qwen38_cache_overlap import current_fill_overlap
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            _capture_expert_output,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        weight = (
            torch.arange(64, device="cuda", dtype=torch.float32)
            .reshape(8, 8)
            .to(torch.bfloat16)
            / 64
        )
        block = SimpleNamespace(
            ids=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
            layer=0,
            shared_expert_gate=None,
            shared_calls=0,
        )

        def shared(hidden, apply_gate):
            block.shared_calls += 1
            return F.linear(hidden, weight)

        block._forward_shared_experts = shared

        @eager_on_graph(True, capture_stub=_capture_expert_output)
        def experts(_block, hidden, ids, *, shared_call, shared_input):
            return runtime.run(
                block.layer,
                ids,
                lambda prepared: (
                    prepared.tensors["w13_weight"][prepared.topk_ids.long()]
                    .to(torch.bfloat16)
                    .sum(dim=1)
                ),
                execution_phase="decode",
                during_fill=lambda: shared_call.run(shared_input),
            )

        def routed(hidden):
            call, shared_input = current_fill_overlap()
            return experts(
                block,
                hidden,
                block.ids,
                shared_call=call,
                shared_input=shared_input,
            )

        block._forward_router_experts = routed
        return block, weight

    def test_shared_gemv_fill_cross_stream_bytes_and_sampling(self):
        from sglang.srt.layers.moe.qwen38_cache_overlap import run_shared_overlap

        fixture, runtime = self._fixture()
        block, weight = self._block(runtime)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        outputs = []
        for step in range(24):
            pair = [(0, 1), (2, 3), (1, 3), (2, 2)][step % 4]
            block.layer = step % 2
            with torch.cuda.stream(streams[step % 2]):
                block.ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                hidden = torch.full(
                    (1, 8), step / 8, dtype=torch.bfloat16, device="cuda"
                )
                output = run_shared_overlap(block, hidden, "decode")
                reference = (
                    fixture.resident[block.layer]
                    .w13_weight[torch.tensor([pair])]
                    .to("cuda", dtype=torch.bfloat16)
                    .sum(dim=1)
                )
                reference.add_(F.linear(hidden, weight))
                outputs.append((output, reference))
        runtime.synchronize()
        for stream in streams:
            stream.synchronize()
        self.assertGreater(runtime.miss_mailbox.overlap_invocations, 0)
        for actual, expected in outputs:
            self.assertTrue(torch.equal(actual, expected))

    def test_shared_fill_breakable_replay_fresh_routes_and_activations(self):
        from sglang.srt.layers.moe.qwen38_cache_overlap import run_shared_overlap
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
        )

        fixture, runtime = self._fixture()
        block, weight = self._block(runtime)
        hidden = torch.ones((1, 8), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_shared_overlap(block, hidden, "decode")
        runtime.synchronize()
        graph = BreakableCUDAGraph()
        before = runtime.run_invocations
        with BreakableCUDAGraphCapture(
            graph, stream=stream, pool=torch.cuda.graph_pool_handle()
        ):
            result = run_shared_overlap(block, hidden * 1.0, "decode") * 1.0
        self.assertEqual(runtime.run_invocations, before)
        recorded_shared_calls = block.shared_calls
        for step, pair in enumerate([(0, 1), (2, 3), (1, 3), (2, 2)]):
            hidden.fill_(step / 8)
            block.ids.copy_(torch.tensor([pair], dtype=torch.int32, device="cuda"))
            graph.replay()
            runtime.synchronize()
            torch.cuda.synchronize()
            reference = (
                fixture.resident[0]
                .w13_weight[torch.tensor([pair])]
                .to("cuda", dtype=torch.bfloat16)
                .sum(dim=1)
            )
            reference.add_(F.linear(hidden, weight))
            self.assertTrue(torch.equal(result, reference))
            self.assertEqual(block.shared_calls, recorded_shared_calls)

    def test_failed_shared_branch_leaves_fill_ordered_for_next_reader(self):
        fixture, runtime = self._fixture()
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        runtime.run(0, ids, lambda prepared: None, execution_phase="decode")

        def fail():
            raise RuntimeError("injected shared failure")

        with self.assertRaisesRegex(RuntimeError, "injected shared failure"):
            runtime.run(
                0,
                torch.tensor([[2, 3]], dtype=torch.int32, device="cuda"),
                lambda prepared: None,
                during_fill=fail,
            )
        actual = runtime.run(
            0,
            ids,
            lambda prepared: prepared.tensors["w13_weight"][
                prepared.topk_ids.long()
            ].clone(),
        )
        runtime.synchronize()
        self.assertTrue(
            torch.equal(
                actual.cpu(), fixture.resident[0].w13_weight[torch.tensor([[0, 1]])]
            )
        )
