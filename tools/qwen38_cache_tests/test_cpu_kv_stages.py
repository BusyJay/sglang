"""Run the production stage policy tests on hosts without torch installed."""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ._load_source import REPO_ROOT, load_source


class TestStagePolicyWithoutCUDA(unittest.TestCase):
    def test_production_policy_cases(self):
        source = load_source(
            "qwen38_kv_stages_cpu", "python/sglang/srt/mem_cache/qwen38_kv_stages.py"
        )
        test_path = (
            REPO_ROOT / "test/registered/unit/mem_cache/test_qwen38_kv_stages.py"
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.qwen38_kv_stages": source,
                "sglang.test.ci.ci_register": SimpleNamespace(
                    register_cpu_ci=lambda **kw: None
                ),
                "sglang.test.test_utils": SimpleNamespace(
                    CustomTestCase=unittest.TestCase
                ),
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "stage_policy_tests", test_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            names = [
                n
                for n in unittest.defaultTestLoader.getTestCaseNames(
                    module.TestKVStages
                )
                if n
                not in (
                    "test_real_allocator_returns_at_existing_result_boundary",
                    "test_cpu_replay_does_not_charge_or_retain_gpu_embeddings",
                )
            ]
            result = unittest.TestResult()
            unittest.TestSuite(module.TestKVStages(n) for n in names).run(result)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
        self.assertEqual(result.testsRun, 9)


if __name__ == "__main__":
    unittest.main()
