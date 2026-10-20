"""Per-layer cache counters must stay column-encodable in the periodic log.

The compact encoder only shares key names across the ``prefill_copy_stats``
rows when *every* row carries the same keys. One layer that created a counter
on first use (``auto_split_runs`` did) pushed all 48 rows back to named dicts
and tripled the periodic ``Qwen38ExpertCacheStats`` line, from 5.6 KB to 15.7
KB in production. That is a producer bug the encoder cannot defend against, so
check it at the source: every counter the threaded runtime can increment must
be seeded, and the packed line must keep the column form.
"""

import ast
import json
import unittest

from ._load_source import REPO_ROOT, load_source

stats_format = load_source(
    "qwen38_counter_seeding_format",
    "python/sglang/srt/layers/moe/qwen38_cache_stats_format.py",
)
THREADED = "python/sglang/srt/layers/moe/qwen38_cache_threaded.py"
POLICY = "python/sglang/srt/layers/moe/qwen38_cache_overlap_policy.py"


def _tree(relative):
    return ast.parse((REPO_ROOT / relative).read_text())


def _assigned_tuple(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    raise AssertionError(f"{name} is no longer assigned")


def _policy_paths():
    tree = _tree(POLICY)
    paths = ast.literal_eval(_assigned_tuple(tree, "PREFILL_CACHE_PATHS").value)
    # Every path the selector can return has to be one of the seeded names.
    returned = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert returned <= set(paths), (returned, paths)
    return tuple(paths)


def _seeded_counters(paths):
    """Evaluate the real PREFILL_COPY_COUNTERS assignment without importing torch."""
    tree = _tree(THREADED)
    node = _assigned_tuple(tree, "PREFILL_COPY_COUNTERS")
    namespace = {"PREFILL_CACHE_PATHS": paths}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<counters>", "exec"), namespace)
    return tuple(namespace["PREFILL_COPY_COUNTERS"])


def _incremented_counters():
    """Keys the runtime writes into a per-layer stats row, literal or templated.

    ``stats[counter] = ...`` with ``counter = f"auto_{path}_runs"`` can only be
    enumerated through the selector's path list, so return those halves instead
    of guessing from any f-string in the file.
    """
    tree = _tree(THREADED)
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
        template = tuple(
            part.value for part in parts if isinstance(part, ast.Constant)
        )
        assert len(template) == 2, "a counter template must be prefix + path + suffix"
        templates.add(template)
    return counters, templates


def seeded_counters():
    """Counter names every per-layer row is created with."""
    return tuple(_seeded_counters(_policy_paths()))


def incremented_counters():
    """Every counter name the runtime can write into a per-layer row."""
    counters, templates = _incremented_counters()
    assert templates, "the auto path counter disappeared"
    for prefix, suffix in templates:
        for path in _policy_paths():
            counters.add(f"{prefix}{path}{suffix}")
    return counters


class CacheStatsCounterSeedingTests(unittest.TestCase):
    def setUp(self):
        self.paths = _policy_paths()
        self.seeded = _seeded_counters(self.paths)

    def test_every_incremented_counter_is_seeded(self):
        missing = sorted(incremented_counters() - set(self.seeded))
        self.assertEqual(
            missing,
            [],
            f"counters created on first use break the column encoding: {missing}",
        )

    def test_seeded_rows_pack_into_shared_columns(self):
        rows = [dict.fromkeys(self.seeded, index) for index in range(48)]
        encoded = stats_format.compact_stats(rows)
        self.assertIn("__qwen38_columns_v1__", encoded)
        self.assertEqual(stats_format.expand_stats(encoded), rows)
        # Key names must be paid once, not once per layer row.
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(rows)) // 2)

    def test_one_unseeded_counter_is_the_regression(self):
        rows = [dict.fromkeys(self.seeded, 0) for _ in range(48)]
        rows[-1].pop(self.seeded[-1])
        encoded = stats_format.compact_stats(rows)
        self.assertNotIn("__qwen38_columns_v1__", encoded)
        self.assertGreater(
            len(json.dumps(encoded)),
            len(json.dumps(stats_format.compact_stats(list(rows[:-1])))),
        )


if __name__ == "__main__":
    unittest.main()
