"""Lossless cache-stat log serialization regressions."""

import json
import random
import unittest
from ._load_source import load_source

stats = load_source(
    "qwen38_stats_format_test",
    "python/sglang/srt/layers/moe/qwen38_cache_stats_format.py",
)


class CompactStatsTests(unittest.TestCase):
    def test_equal_numeric_values_keep_original_json_types(self):
        original = [
            {"value": 1, "zero": -0.0, "nested": {"items": [True, 0]}},
            {"value": 1.0, "zero": 0.0, "nested": {"items": [1, False]}},
            {"value": True, "zero": 0.0, "nested": {"items": [1.0, 0.0]}},
        ]
        expanded = stats.expand_stats(stats.compact_stats(original))
        self.assertEqual(
            json.dumps(expanded, sort_keys=True), json.dumps(original, sort_keys=True)
        )

    def test_lossless_columns_preserve_all_layers_and_fields(self):
        rng = random.Random(81)
        original = {
            "hits": 123,
            "layer_cache_slots": [144] * 48,
            "prefill_copy_stats": [
                {
                    "dense": 1,
                    "missing": rng.randrange(1000),
                    "nested": {"a": None, "b": [0, 1]},
                    "flag": False,
                }
                for _ in range(48)
            ],
            "mixed": [{"a": 1}, {"b": 2}],
            "empty": [],
        }
        encoded = stats.compact_stats(original)
        self.assertEqual(stats.expand_stats(json.loads(json.dumps(encoded))), original)
        self.assertEqual(encoded["prefill_copy_stats"]["common"]["dense"], 1)
        self.assertEqual(len(encoded["prefill_copy_stats"]["columns"]["missing"]), 48)
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(original)))
        self.assertEqual(stats.expand_stats(original), original)

    def test_identical_empty_dictionaries_keep_row_count(self):
        for original in ([{}] * 48, [{"a": 0}] * 48):
            self.assertEqual(
                stats.expand_stats(stats.compact_stats(original)), original
            )

    def test_canary_expands_compact_payload(self):
        import tempfile
        from pathlib import Path
        from .test_cpu_cache_stats import canary_module

        original = {
            "hits": 12,
            "prefill_copy_stats": [
                {"dense": 1, "missing": 0},
                {"dense": 1, "missing": 3},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "service.log"
            log.write_text(
                "INFO Qwen38ExpertCacheStats[periodic] "
                + json.dumps(stats.compact_stats(original))
                + "\n"
            )
            self.assertEqual(canary_module._extract_cache_stats(log), original)


if __name__ == "__main__":
    unittest.main()
