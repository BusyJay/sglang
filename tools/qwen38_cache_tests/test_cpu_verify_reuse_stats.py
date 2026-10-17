"""Route reuse counts must distinguish token sharing and intra-token duplicates."""

import json
import tempfile
import unittest
from pathlib import Path

from ._load_source import load_source

stats = load_source(
    "verify_reuse_stats",
    "tools/qwen38_cache_tests/verify_experiments_20261018/analyze_verify_reuse.py",
)


class ReuseStatsTests(unittest.TestCase):
    def analyze(self, routes):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        dict(event="start", case="warmup"),
                        dict(layer=0, routes=routes),
                        dict(event="end", case="warmup"),
                        dict(event="start", case="short"),
                        dict(layer=0, routes=routes),
                        dict(event="end", case="short"),
                        dict(layer=0, routes=routes),
                    )
                )
                + "\n"
            )
            return stats.analyze(path)["overall"]

    def test_full_overlap_retains_padded_arithmetic(self):
        row = self.analyze([[0, 1]] * 3)
        self.assertEqual(row["rounds"], 1)
        self.assertAlmostEqual(row["cross_token_reuse_fraction"], 2 / 3)
        self.assertAlmostEqual(row["padded_arithmetic_factor"], 4 / 3)
        self.assertEqual(row["expert_token_multiplicity"], {3: 2})

    def test_disjoint_routes_multiply_arithmetic_by_four(self):
        row = self.analyze([[0, 1], [2, 3], [4, 5]])
        self.assertEqual(row["cross_token_reuse_fraction"], 0)
        self.assertEqual(row["padded_arithmetic_factor"], 4)
        self.assertEqual(row["no_cross_token_reuse_fraction"], 1)

    def test_duplicate_and_invalid_routes_do_not_inflate_cross_token_reuse(self):
        row = self.analyze([[0, 0, 1, -1], [0, 2, 2, 512]])
        self.assertEqual(row["within_token_duplicates"], 2)
        self.assertAlmostEqual(row["cross_token_reuse_fraction"], 0.25)
        self.assertAlmostEqual(row["total_route_reuse_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
