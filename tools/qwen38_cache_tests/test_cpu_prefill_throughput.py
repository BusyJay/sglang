import ast
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT


class TestPrefillThroughput(unittest.TestCase):
    def setUp(self):
        source = ast.parse(
            (
                REPO_ROOT
                / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
            ).read_text()
        )
        method = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_prefill_report_latency"
        )
        self.clock = SimpleNamespace(monotonic=mock.Mock())
        namespace = {"time": self.clock}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                "actual-prefill-latency",
                "exec",
            ),
            namespace,
        )
        self.latency = lambda batch: namespace["_prefill_report_latency"](None, batch)

    def batch(self, start, split=False):
        return SimpleNamespace(
            launch_ts=start,
            split_prefill_start=None,
            forward_mode=SimpleNamespace(is_split_prefill=lambda: split),
        )

    def test_delayed_log_then_tail_does_not_inflate_throughput(self):
        # Previous log is delayed until time 20. The tail was launched at 10
        # and finishes at 21: 1024/1 (inter-log) is false; 1024/11 is its span.
        self.clock.monotonic.return_value = 20.0
        previous_latency = self.latency(self.batch(0.0))
        self.clock.monotonic.return_value = 21.0
        tail_latency = self.latency(self.batch(10.0))
        self.assertEqual(previous_latency, 20.0)
        self.assertEqual(tail_latency, 11.0)
        self.assertAlmostEqual(1024 / tail_latency, 93.090909, places=6)

    def test_idle_before_launch_is_excluded(self):
        self.clock.monotonic.return_value = 105.0
        self.assertEqual(self.latency(self.batch(100.0)), 5.0)

    def test_split_prefill_uses_first_split_start(self):
        batch = self.batch(8.0, split=True)
        batch.split_prefill_start = (1, 2.0)
        self.clock.monotonic.return_value = 10.0
        self.assertEqual(self.latency(batch), 8.0)

    def test_missing_launch_does_not_report_spurious_rate(self):
        self.assertEqual(self.latency(None), 0.0)
        self.assertEqual(self.latency(self.batch(None)), 0.0)
