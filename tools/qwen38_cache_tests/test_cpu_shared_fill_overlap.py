import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from . import test_cpu_decode_graph as graph_test
from ._load_source import load_source


class Value(graph_test.Tensor):
    def add_(self, other):
        self.value += other.value
        return self


class TestSharedFillOverlap(unittest.TestCase):
    def setUp(self):
        replay = graph_test.TestEagerBreakReplay()
        replay.setUp()
        self.graph = replay.graph
        self.trace = []

        @contextlib.contextmanager
        def stream_context(stream):
            self.trace.append("enter_fill")
            try:
                yield
            finally:
                self.trace.append("exit_fill")

        with mock.patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(
                    empty_like=lambda x: Value(0),
                    cuda=SimpleNamespace(stream=stream_context),
                ),
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": self.graph,
                "sglang.srt.layers.moe": SimpleNamespace(
                    get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: True)
                ),
            },
        ):
            self.module = load_source(
                "shared_fill_overlap_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_overlap.py",
            )
        self.compute = SimpleNamespace(
            wait_event=lambda event: self.trace.append("compute_wait")
        )
        self.fill = SimpleNamespace(
            wait_event=lambda event: self.trace.append("fill_wait")
        )
        self.ready = SimpleNamespace(record=lambda stream: self.trace.append("ready"))
        self.filled = SimpleNamespace(record=lambda stream: self.trace.append("filled"))

    def test_fill_and_shared_join_before_routed_execution(self):
        self.module.fill_with_overlap(
            self.compute,
            self.fill,
            self.ready,
            self.filled,
            lambda: self.trace.append("fill"),
            lambda: self.trace.append("shared"),
        )
        self.trace.append("routed")
        self.assertEqual(
            self.trace,
            [
                "ready",
                "fill_wait",
                "enter_fill",
                "fill",
                "filled",
                "exit_fill",
                "shared",
                "compute_wait",
                "routed",
            ],
        )

    def test_failed_shared_still_joins_outstanding_fill(self):
        def fail():
            raise RuntimeError("shared failed")

        with self.assertRaisesRegex(RuntimeError, "shared failed"):
            self.module.fill_with_overlap(
                self.compute, self.fill, self.ready, self.filled, lambda: None, fail
            )
        self.assertEqual(self.trace[-1], "compute_wait")

    def test_failed_fill_still_records_completion_and_skips_shared(self):
        shared = mock.Mock()

        def fail():
            raise RuntimeError("fill failed")

        with self.assertRaisesRegex(RuntimeError, "fill failed"):
            self.module.fill_with_overlap(
                self.compute, self.fill, self.ready, self.filled, fail, shared
            )
        self.assertEqual(self.trace[-3:], ["filled", "exit_fill", "compute_wait"])
        shared.assert_not_called()

    def test_decode_only_eligibility_and_noninplace_guard(self):
        config = SimpleNamespace(inplace=False, apply_router_weight_on_input=False)
        runtime = SimpleNamespace(
            supports_fill_overlap=True, miss_mailbox=object(), fused_decode_plan=True
        )
        method = SimpleNamespace(runtime=runtime, moe_runner_config=config)
        block = SimpleNamespace(
            shared_expert=object(), enable_shared_expert_fusion=False, tp_size=1
        )
        hidden = SimpleNamespace(is_cuda=True, shape=(1, 8))
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.layers.moe": SimpleNamespace(
                    get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: True)
                )
            },
        ):
            self.assertTrue(
                self.module.can_overlap_shared(method, block, hidden, "decode")
            )
            for phase in ("prefill", "verify", None):
                self.assertFalse(
                    self.module.can_overlap_shared(method, block, hidden, phase)
                )
            for obj, field, value in [
                (hidden, "is_cuda", False),
                (hidden, "shape", (2, 8)),
                (block, "enable_shared_expert_fusion", True),
                (config, "inplace", True),
                (runtime, "miss_mailbox", None),
            ]:
                with mock.patch.object(obj, field, value):
                    self.assertFalse(
                        self.module.can_overlap_shared(method, block, hidden, "decode")
                    )

    def test_eager_shared_uses_fresh_inputs_and_leaves_no_callback(self):
        hidden, route = Value(10), Value(0)
        calls = []

        def shared(x, apply_gate):
            self.assertFalse(apply_gate)
            calls.append(x.value)
            return Value(x.value * 2)

        def routed(x):
            call, original = self.module.current_fill_overlap()
            self.assertIs(original, x)
            call.run(original)
            return Value(x.value * 3 + route.value)

        block = SimpleNamespace(
            _forward_router_experts=routed,
            _forward_shared_experts=shared,
            shared_expert_gate=None,
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.models.qwen2_moe": SimpleNamespace(
                    fused_gate_sigmoid_mul_add=None, reduce_moe_output=lambda x: x
                ),
            },
        ):
            for value in (1, 3, 2):
                hidden.value, route.value = value, value + 1
                output = self.module.run_shared_overlap(block, hidden, "decode")
                self.assertEqual(output.value, 5 * value + value + 1)
                self.assertIsNone(self.module.current_fill_overlap())
        self.assertEqual(calls, [1, 3, 2])

    def test_shared_graph_records_at_capture_and_only_replays_later(self):
        block = SimpleNamespace(
            _forward_shared_experts=mock.Mock(return_value=Value(7))
        )
        call = self.module.SharedExpertCall(block)
        hidden = Value(2)
        graph = SimpleNamespace(replay=mock.Mock())
        cuda = SimpleNamespace(
            CUDAGraph=mock.Mock(return_value=graph),
            current_stream=lambda: self.compute,
            graph=mock.Mock(return_value=contextlib.nullcontext()),
        )
        with mock.patch.object(self.module.torch, "cuda", cuda):
            call.capture(hidden)
        cuda.graph.assert_called_once_with(graph, pool=None, stream=self.compute)
        self.assertEqual(call.output.value, 7)
        self.assertIsNotNone(call.output)
        graph.replay.assert_not_called()
        for _ in range(3):
            call.run(hidden)
        self.assertEqual(graph.replay.call_count, 3)
        block._forward_shared_experts.assert_called_once_with(hidden, apply_gate=False)
        self.assertNotIn(hidden, vars(call).values())
        with self.assertRaisesRegex(RuntimeError, "already captured"):
            call.capture(hidden)

    def test_failed_shared_capture_does_not_install_partial_graph(self):
        block = SimpleNamespace(
            _forward_shared_experts=mock.Mock(
                side_effect=RuntimeError("capture failed")
            )
        )
        call = self.module.SharedExpertCall(block)
        cuda = SimpleNamespace(
            CUDAGraph=lambda: object(),
            current_stream=lambda: self.compute,
            graph=lambda *a, **kw: contextlib.nullcontext(),
        )
        with mock.patch.object(self.module.torch, "cuda", cuda):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                call.capture(Value(1))
        self.assertIsNone(call.graph)
        self.assertIsNone(call.output)

    def test_shared_graph_reuses_outer_capture_pool(self):
        pool = (19, 5)
        block = SimpleNamespace(_forward_shared_experts=lambda *a, **kw: Value(1))
        cuda = SimpleNamespace(
            CUDAGraph=lambda: object(),
            current_stream=lambda: self.compute,
            graph=mock.Mock(side_effect=lambda *a, **kw: contextlib.nullcontext()),
        )
        with (
            mock.patch.object(self.module.torch, "cuda", cuda),
            mock.patch.object(
                self.module, "get_current_capture_pool", return_value=pool
            ),
        ):
            for _ in range(2):
                self.module.SharedExpertCall(block).capture(Value(0))
        self.assertEqual(cuda.graph.call_count, 2)
        for call in cuda.graph.call_args_list:
            self.assertIs(call.kwargs["pool"], pool)

    def test_failed_shared_callback_does_not_leak_to_later_expert_calls(self):
        def fail(*args, **kwargs):
            raise RuntimeError("shared failed")

        def routed(x):
            call, hidden = self.module.current_fill_overlap()
            call.run(hidden)

        block = SimpleNamespace(
            _forward_router_experts=routed,
            _forward_shared_experts=fail,
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.models.qwen2_moe": SimpleNamespace(
                    fused_gate_sigmoid_mul_add=None, reduce_moe_output=None
                ),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "shared failed"):
                self.module.run_shared_overlap(block, Value(1), "decode")
        self.assertIsNone(self.module.current_fill_overlap())
