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
                    zeros_like=lambda x: Value(0),
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

    def test_whole_moe_capture_stub_and_replay_use_fresh_inputs(self):
        phase = SimpleNamespace(value=None)
        calls, breaks = [], []

        @contextlib.contextmanager
        def scoped(**kw):
            old = phase.value
            phase.value = kw["qwen38_execution_phase"]
            try:
                yield
            finally:
                phase.value = old

        hidden, route = Value(10), Value(0)

        def shared(x, apply_gate):
            self.assertFalse(apply_gate)
            calls.append(("shared", x.value, phase.value))
            return Value(x.value * 2)

        def routed(x):
            calls.append(("routed", x.value, phase.value))
            self.module.current_fill_overlap()()
            return Value(x.value * 3 + route.value)

        def merge(x, weight, shared, routed):
            routed.value += shared.value / 2

        block = SimpleNamespace(
            _forward_router_experts=routed,
            _forward_shared_experts=shared,
            shared_expert_gate=SimpleNamespace(
                weight=SimpleNamespace(squeeze=lambda: None)
            ),
        )
        capture = SimpleNamespace(
            _barrier_fn=None,
            _end_current_segment=lambda: None,
            _begin_new_segment=lambda: None,
            cuda_graph=SimpleNamespace(_break_fns=breaks),
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.models.qwen2_moe": SimpleNamespace(
                    fused_gate_sigmoid_mul_add=merge, reduce_moe_output=lambda x: x
                ),
                "sglang.srt.runtime_context": SimpleNamespace(
                    get_forward=lambda: SimpleNamespace(scoped=scoped)
                ),
            },
        ):
            token = self.graph._current_capture_var.set(capture)
            try:
                with mock.patch.object(
                    self.graph, "_weak_ref_if_tensor", side_effect=lambda x: x
                ):
                    output = self.module.run_shared_overlap(block, hidden, "decode")
            finally:
                self.graph._current_capture_var.reset(token)
            self.assertEqual(calls, [])
            for value in (1, 3, 2):
                hidden.value, route.value = value, value + 1
                breaks[0]()
                self.assertEqual(output.value, 4 * value + value + 1)
                self.assertIsNone(self.module.current_fill_overlap())
                self.assertIsNone(phase.value)
            self.assertEqual([item[0] for item in calls], ["routed", "shared"] * 3)
            self.assertTrue(all(item[2] == "decode" for item in calls))

    def test_failed_shared_callback_does_not_leak_to_later_expert_calls(self):
        def fail(*args, **kwargs):
            raise RuntimeError("shared failed")

        block = SimpleNamespace(
            _forward_router_experts=lambda x: self.module.current_fill_overlap()(),
            _forward_shared_experts=fail,
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "sglang.srt.models.qwen2_moe": SimpleNamespace(
                    fused_gate_sigmoid_mul_add=None, reduce_moe_output=None
                ),
                "sglang.srt.runtime_context": SimpleNamespace(
                    get_forward=lambda: SimpleNamespace(
                        scoped=lambda **kw: contextlib.nullcontext()
                    )
                ),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "shared failed"):
                self.module.run_shared_overlap(block, Value(1), "decode")
        self.assertIsNone(self.module.current_fill_overlap())
