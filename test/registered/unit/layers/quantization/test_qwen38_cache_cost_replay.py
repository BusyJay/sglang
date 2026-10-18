"""Cache cost must sum per-call prices, preserve warm state and reject gaps."""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from tools.qwen38_cache_tests.cache_cost_replay import (
    FillCost,
    read_sweeps,
    replay_layer,
)
from tools.qwen38_cache_tests.cache_route_record import RouteRecorder
from tools.qwen38_cache_tests.replay_cache import allocate

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCacheCostReplay(CustomTestCase):
    def test_recorder_publishes_complete_sweeps_and_refuses_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tape.jsonl"
            recorder = RouteRecorder(path, 2, 1)
            recorder.append(0, [[3, 1]])
            self.assertEqual(path.read_text(), "")
            recorder.append(1, [[2, 0]])
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual([r["layer"] for r in rows], [0, 1])
            self.assertEqual(rows[0]["routes"], [[3, 1]])
            recorder.append(0, [[0]])
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                RouteRecorder(path, 2, 1)

    def cost(self):
        return FillCost(
            dict(
                reports=[
                    dict(variant="fill", misses=n, ms=v)
                    for n, v in [(0, 0.1), (1, 2), (2, 3), (4, 4)]
                ]
            ),
            "fill",
            "ms",
        )

    def test_per_call_cost_and_warm_prefix(self):
        events = [(x, 1, None) for x in [[0], [1], [0, 1], [2, 3]]]
        row = replay_layer(events, 2, 4, "lru", 0, self.cost(), 2, 4)
        self.assertEqual(row["misses"], 2)
        self.assertEqual(row["miss_histogram"], {0: 1, 2: 1})
        self.assertAlmostEqual(row["estimated_fill_ms"], 3.1)
        self.assertEqual(row["all_hit_calls"], 1)

    def test_fewer_misses_can_cost_more_under_sublinear_price(self):
        # Give the extra slot to layer 0: one fewer miss, but two separate
        # transactions instead of one three-miss call on the alternative.
        miss_layout = allocate([{1: 2, 2: 0}, {1: 3, 2: 0}], 3)
        cost = self.cost()
        time_layout = allocate([{1: 2 * cost(1), 2: 0}, {1: cost(3), 2: 0}], 3)
        self.assertEqual(miss_layout["layer_cache_slots"], [1, 2])
        self.assertEqual(time_layout["layer_cache_slots"], [2, 1])
        self.assertEqual(cost.interpolated, {3})
        with self.assertRaises(ValueError):
            cost(5)

    def test_trace_rejects_gap_and_cross_request_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"

            def write(rows):
                path.write_text("\n".join(json.dumps(x) for x in rows))

            write([dict(layer=0, routes=[[0]]), dict(layer=0, routes=[[1]])])
            with self.assertRaises(ValueError):
                read_sweeps(path, 2, 4)
            rows = [
                dict(event="start", case="a"),
                dict(layer=0, routes=[[0]]),
                dict(event="end"),
                dict(event="start", case="b"),
                dict(layer=0, routes=[[1]]),
                dict(event="end"),
            ]
            write(rows)
            with self.assertRaises(ValueError):
                read_sweeps(path, 1, 4)
            self.assertEqual(len(read_sweeps(path, 1, 4, "a")), 1)


if __name__ == "__main__":
    unittest.main()
