"""Guard against silently omitting changed CTA buckets from overlap timing."""

import ast
import types
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestFillBenchmarkCoverage(CustomTestCase):
    def test_every_candidate_bucket_gets_overlap_comparison(self):
        root = Path(__file__).resolve().parents[4]
        source = root / "test/manual/kernels/moe/qwen38_fill_ab.py"
        tree = ast.parse(source.read_text())
        condition = next(
            node.test
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "copy_stream is not None" in ast.unparse(node.test)
            and "len(pairs)" in ast.unparse(node.test)
        )
        expression = compile(ast.Expression(condition), str(source), "eval")
        for full_sweep in (False, True):
            counts = range(31) if full_sweep else (3, 4, 20, 30)
            for count in counts:
                for variant in ("mapped_adaptive", "mapped_candidate"):
                    with self.subTest(
                        full_sweep=full_sweep, count=count, variant=variant
                    ):
                        enabled = eval(
                            expression,
                            {
                                "copy_stream": object(),
                                "pairs": [None] * count,
                                "name": variant,
                                "args": types.SimpleNamespace(
                                    full_miss_sweep=full_sweep
                                ),
                            },
                        )
                        self.assertTrue(enabled)


if __name__ == "__main__":
    unittest.main()
