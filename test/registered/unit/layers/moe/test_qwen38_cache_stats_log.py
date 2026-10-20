"""The periodic Qwen38ExpertCacheStats line must stay column-encoded.

The lossless encoder shares key names across the 48 per-layer rows only when
*every* row carries the same keys, so one counter created on first use costs a
key name per row: ``auto_split_runs`` once took the line from 5.6 KB to 15.7 KB
in production. The encoder cannot defend against that itself, so the guard runs
against the producer: the counters the threaded runtime increments have to be
the ones it seeds.
"""

import ast
import json
import unittest
from pathlib import Path

import sglang.srt.layers.moe.qwen38_cache_overlap_policy as policy_module
import sglang.srt.layers.moe.qwen38_cache_threaded as threaded_module
from sglang.srt.layers.moe.qwen38_cache_stats_format import compact_stats, expand_stats
from sglang.srt.layers.moe.qwen38_cache_threaded import PREFILL_COPY_COUNTERS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

NUM_LAYERS = 48


def _tree(module):
    return ast.parse(Path(module.__file__).read_text())


def _policy_paths():
    """Every path the selector can return, so counter names cannot drift."""
    returned = {
        node.value.value
        for node in ast.walk(_tree(policy_module))
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert returned <= set(policy_module.PREFILL_CACHE_PATHS), returned
    return tuple(policy_module.PREFILL_CACHE_PATHS)


def _incremented_counters():
    """Counter names the runtime writes into a per-layer row, literal or f-string.

    ``stats[counter] = ...`` with ``counter = f"auto_{path}_runs"`` can only be
    enumerated through the selector's path list, so expand those templates rather
    than guessing from any f-string in the file.
    """
    tree = _tree(threaded_module)
    counters, counter_names = set(), set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "stats"
        ):
            continue
        if isinstance(node.slice, ast.Name):
            counter_names.add(node.slice.id)
        for key in ast.walk(node.slice):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                counters.add(key.value)
    templates = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.JoinedStr)
            and any(
                isinstance(target, ast.Name) and target.id in counter_names
                for target in node.targets
            )
        ):
            continue
        parts = node.value.values
        assert (
            sum(isinstance(part, ast.FormattedValue) for part in parts) == 1
        ), "a counter template must interpolate exactly once"
        template = tuple(part.value for part in parts if isinstance(part, ast.Constant))
        assert len(template) == 2, "a counter template must be prefix + path + suffix"
        templates.add(template)
    assert templates, "the auto path counter disappeared"
    for prefix, suffix in templates:
        for path in _policy_paths():
            counters.add(f"{prefix}{path}{suffix}")
    return counters


class TestCacheStatsLogEncoding(CustomTestCase):
    def test_every_incremented_counter_is_seeded(self):
        unseeded = sorted(_incremented_counters() - set(PREFILL_COPY_COUNTERS))
        self.assertEqual(
            unseeded,
            [],
            f"counters created on first use break the column encoding: {unseeded}",
        )

    def test_seeded_counters_are_exactly_the_incremented_set(self):
        # A seeded counter nobody writes is harmless for the encoding but it is
        # still paid for in every log line, so the two sets must agree.
        self.assertEqual(set(PREFILL_COPY_COUNTERS), _incremented_counters())

    def test_periodic_rows_pack_into_shared_columns(self):
        rows = [dict.fromkeys(PREFILL_COPY_COUNTERS, 0) for _ in range(NUM_LAYERS)]
        for row, layer in zip(rows, range(NUM_LAYERS)):
            for counter in row:
                row[counter] = layer
        encoded = compact_stats(rows)
        self.assertIn("__qwen38_columns_v1__", encoded)
        self.assertEqual(expand_stats(encoded), rows)
        # Key names are paid once, not once per layer row.
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(rows)) // 2)

    def test_one_unseeded_counter_is_the_regression(self):
        rows = [dict.fromkeys(PREFILL_COPY_COUNTERS, 0) for _ in range(NUM_LAYERS)]
        rows[-1].pop(PREFILL_COPY_COUNTERS[-1])
        self.assertNotIn("__qwen38_columns_v1__", compact_stats(rows))


if __name__ == "__main__":
    unittest.main()
