"""Budget tiers, persistent reuse and garbage collection in a dedicated tmpfs."""

import gc
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

policy = load_source(
    "startup_cache_policy_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/startup_cache.py",
)


class TestStartupCachePolicy(unittest.TestCase):
    def test_tier_boundaries_preserve_requested_priority(self):
        for budget, expected in (
            (132, (True, True)),
            (131, (True, False)),
            (96, (True, False)),
            (95, (False, True)),
            (36, (False, True)),
            (35, (False, False)),
            (0, (False, False)),
        ):
            with self.subTest(budget=budget):
                self.assertEqual(policy.choose_cache_tier(96, 36, budget), expected)
        self.assertEqual(policy.choose_cache_tier(96, 0, 96), (True, False))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = self.root / "experts.bin"
        self.payload.write_bytes(bytes(range(96)))
        self.manifest = SimpleNamespace(payload_path=self.payload, payload_nbytes=96)
        self.module = SimpleNamespace(
            _fp4_ple=True,
            _backend="pinned",
            weight=SimpleNamespace(shape=(4, 8)),
            embedding_dim=16,
        )
        self.model = SimpleNamespace(named_modules=lambda: [("ple", self.module)])
        self.cache_dir = self.root / "cache"
        self.patch = mock.patch.dict(
            os.environ,
            {
                "SGLANG_QWEN38_SOURCE_CACHE_DIR": str(self.cache_dir),
                "SGLANG_QWEN38_STARTUP_CACHE_RESERVE_GIB": "0",
            },
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def select(self, budget, available=1 << 30, cgroup=None):
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN38_STARTUP_CACHE_MAX_GIB": str(budget / (1 << 30))}
        ), mock.patch.object(
            policy, "memory_headroom", return_value=(1 << 30, available, cgroup)
        ):
            mode = policy.prepare_startup_cache(
                self.model, self.manifest, [str(self.payload)]
            )
        return mode, self.model._qwen38_startup_cache_plan

    def test_all_four_tiers_wire_components(self):
        for budget, tier, mode, ple in (
            (132, "both", "shared_thp", True),
            (96, "experts", "shared_thp", False),
            (36, "ple", "anonymous_thp", True),
            (0, "none", "anonymous_thp", False),
        ):
            with self.subTest(tier=tier):
                selected, report = self.select(budget)
                self.assertEqual(selected, mode)
                self.assertEqual(report["tier"], tier)
                self.assertEqual(
                    bool(getattr(self.module, "_shared_cache_spec", None)), ple
                )

    def test_cgroup_headroom_limits_host_budget(self):
        mode, report = self.select(132, cgroup=40)
        self.assertEqual((mode, report["tier"]), ("anonymous_thp", "ple"))

    def test_existing_data_is_credited_and_unselected_data_reclaimed(self):
        import threading
        from sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache import (
            map_shared_source,
        )

        with self.payload.open("rb") as source:
            mapping, _ = map_shared_source(
                source.fileno(), self.cache_dir, threading.Event()
            )
        mapping.close()
        del mapping
        gc.collect()
        path = next(self.cache_dir.glob("*.bin"))
        mode, report = self.select(132, available=16, cgroup=16)
        self.assertEqual((mode, report["tier"]), ("shared_thp", "experts"))
        self.assertEqual(report["reusable_bytes"], 96)
        self.assertTrue(path.exists())
        mode, report = self.select(36)
        self.assertEqual((mode, report["tier"]), ("anonymous_thp", "ple"))
        self.assertFalse(path.exists())

    def test_garbage_is_freed_before_selection_and_reusable_file_is_kept(self):
        self.cache_dir.mkdir()
        garbage = self.cache_dir / "old-image-data"
        garbage.write_bytes(b"not current data")
        _, report = self.select(132)
        self.assertEqual(report["cleanup"]["removed_files"], 1)
        self.assertFalse(garbage.exists())

    def test_cache_directory_cannot_delete_source_model_files(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN38_SOURCE_CACHE_DIR": str(self.root)}
        ):
            mode, report = self.select(132)
        self.assertEqual((mode, report["tier"]), ("anonymous_thp", "none"))
        self.assertIn("source model files", report["reason"])
        self.assertEqual(self.payload.read_bytes(), bytes(range(96)))

    def test_registration_failure_reclaims_published_cache_before_fallback(self):
        from contextlib import nullcontext
        import torch
        from sglang.kernels.ops.moe import qwen38_expert_cache
        from sglang.srt.models.qwen38_shared_ple import try_shared_ple

        def reject_registration(raw):
            # This frame deliberately retains the tensor in the traceback, just
            # like a Python wrapper around a failing CUDA registration call.
            raise RuntimeError("register expert mmap: simulated unsupported mapping")

        stat = self.payload.stat()
        identity = [
            (
                str(self.payload),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        ]
        fake = SimpleNamespace(mailbox_register=reject_registration)
        with mock.patch.object(
            torch.cuda, "device", return_value=nullcontext()
        ), mock.patch.object(qwen38_expert_cache, "_module", return_value=fake):
            self.assertIsNone(
                try_shared_ple(
                    self.payload, (str(self.cache_dir), "d" * 64, identity), "cuda:0"
                )
            )
        self.assertEqual(list(self.cache_dir.glob("*.bin")), [])
        self.assertEqual(self.payload.read_bytes(), bytes(range(96)))


if __name__ == "__main__":
    unittest.main()
