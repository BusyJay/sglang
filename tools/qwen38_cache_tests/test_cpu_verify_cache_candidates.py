"""Fixed-budget hotspot allocation and lossless compact-log regressions."""

import json
import os
import subprocess
import sys
import unittest

from ._load_source import REPO_ROOT
from .test_cpu_adaptive_cache import adaptive


class VerifyExperimentsTests(unittest.TestCase):
    def test_worker_default_extensions_and_explicit_rollback(self):
        request = dict(
            current=[133, 134],
            budget=267,
            min_slots=30,
            num_experts=512,
            top_k=1,
            policy="lfu",
            protected_fraction=0.7,
            routes=[[[i % 200] for i in range(64)], [[511]] * 64],
        )
        worker = (
            REPO_ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
        )
        for setting, maximum in ((None, 256), ("", 144), ("192", 192)):
            env = os.environ.copy()
            env.pop("SGLANG_QWEN38_ADAPTIVE_CANDIDATE_MAX", None)
            if setting is not None:
                env["SGLANG_QWEN38_ADAPTIVE_CANDIDATE_MAX"] = setting
            result = subprocess.run(
                [sys.executable, str(worker)],
                input=json.dumps(request) + "\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
                env=env,
            )
            reply = json.loads(result.stdout)
            self.assertEqual(max(reply["candidates"]), maximum)
            if reply["target"] is not None:
                self.assertEqual(sum(reply["target"]), 267)

    def test_coarse_candidates_remove_ceiling_without_changing_budget(self):
        allocator = adaptive.AdaptiveLayerAllocator(
            [133, 134], 512, min_slots=30, candidate_max=256
        )
        self.assertEqual(allocator.budget, 267)
        self.assertEqual(
            allocator.candidates, (80, 96, 112, 128, 133, 134, 144, 160, 192, 224, 256)
        )
        migrated = adaptive.AdaptiveLayerAllocator([256, 80], 512, candidate_max=256)
        self.assertEqual(max(migrated.candidates), 256)
        for invalid in (133, 513, "256"):
            with self.assertRaises(ValueError):
                adaptive.AdaptiveLayerAllocator([134, 134], 512, candidate_max=invalid)

    def test_worker_hotspot_crosses_old_ceiling_with_fixed_budget(self):
        request = dict(
            current=[64, 64],
            budget=128,
            min_slots=1,
            num_experts=128,
            top_k=1,
            policy="lfu",
            protected_fraction=0.7,
            candidate_max=128,
            routes=[[[i % 90] for i in range(256)], [[127]] * 256],
        )
        worker = (
            REPO_ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
        )
        result = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
        )
        reply = json.loads(result.stdout)
        self.assertEqual(sum(reply["target"]), 128)
        self.assertGreater(reply["target"][0], 80)
        self.assertIn(96, reply["candidates"])


if __name__ == "__main__":
    unittest.main()
