import itertools
import json
import tempfile
import unittest
from pathlib import Path

from ._load_source import load_source
from .replay_cache import allocate, miss_curve, read_trace

capacity = load_source("sglang.srt.layers.moe.qwen38_cache_capacity",
    "python/sglang/srt/layers/moe/qwen38_cache_capacity.py")


class TestCapacity(unittest.TestCase):
    def test_total_budget_is_exact_and_legacy_compatible(self):
        for total in (480, 3840, 3841, 5040, 24576):
            layout = capacity.resolve_cache_slots({"cache_slots_total": total}, 48, 512)
            self.assertEqual(sum(layout), total)
            self.assertLessEqual(max(layout) - min(layout), 1)
        self.assertEqual(capacity.resolve_cache_slots({"cache_slots_per_layer": 105}, 48, 512), [105] * 48)

    def test_invalid_total_budgets_and_ambiguous_configs(self):
        for total in (False, 479, 24577, 3840.0, "3840"):
            with self.assertRaises(ValueError):
                capacity.resolve_cache_slots({"cache_slots_total": total}, 48, 512)
        for key, value in (("cache_slots_per_layer", 80), ("layer_cache_slots", [80] * 48)):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                capacity.resolve_cache_slots({"cache_slots_total": 3840, key: value}, 48, 512)
    def test_defaults_and_explicit_layout(self):
        self.assertEqual(capacity.normalize_layer_slots(2, None, 3, 4), [2, 2, 2])
        self.assertEqual(capacity.normalize_layer_slots(2, [1, 4, 2], 3, 4), [1, 4, 2])

    def test_invalid_layouts(self):
        for layout in ([1], [1, 0], [1, 5], [True, 2], {"0": 2}):
            with self.assertRaises(ValueError):
                capacity.normalize_layer_slots(2, layout, 2, 4)

    def test_replay_hit_and_cold_miss(self):
        self.assertEqual(miss_curve([[0], [1], [0], [1]], [1, 2], 4), {1: 4, 2: 2})

    def test_budget_allocation_matches_exhaustive_search(self):
        curves = [{1: 10, 2: 10, 3: 1}, {1: 8, 2: 4, 3: 3}, {1: 7, 2: 3, 3: 1}]
        for budget in range(3, 10):
            reference = min((sum(curve[c] for curve, c in zip(curves, choice)), sum(choice), list(choice))
                for choice in itertools.product([1, 2, 3], repeat=3) if sum(choice) <= budget)
            actual = allocate(curves, budget)
            self.assertEqual((actual["misses"], actual["used_slots"], actual["layer_cache_slots"]), reference)

    def test_trace_filters_prefill_and_validates_layers(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in [
                {"layer": 0, "routes": [[2, 1, -1]]},
                {"layer": 0, "routes": [[0], [1]]},
                {"layer": 1, "routes": [[3]]},
            ]))
            layers, skipped = read_trace(path, 2, 4)
            self.assertEqual(layers, [[[1, 2]], [[3]]])
            self.assertEqual(skipped, 1)
            with self.assertRaises(ValueError):
                read_trace(path, 3, 4)
