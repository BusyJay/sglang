"""Real safetensors row reads, payload filtering, SLRU and byte parity."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ._load_source import load_source

try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch/numpy required")
class TestLazyPle(unittest.TestCase):
    def setUp(self):
        self.module = load_source(
            "ple_lazy_under_test", "python/sglang/srt/models/qwen4_exp_ple_lazy.py"
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, dtype="BF16", rows=8, dim=4, shard=0):
        name = f"model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        itemsize = 2 if dtype == "BF16" else 1
        raw = bytes((i % 127 for i in range(rows * dim * itemsize)))
        header = json.dumps(
            {
                name: {
                    "dtype": dtype,
                    "shape": [rows, dim],
                    "data_offsets": [0, len(raw)],
                }
            }
        ).encode()
        path = Path(self.tmp.name) / f"shard{shard}.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
        return path, name, raw

    def cache(self, shards, row_bytes=8, capacity=4):
        cache = self.module.PleRowCache(
            shards, row_bytes=row_bytes, capacity_rows=capacity, pin_memory=False
        )
        self.addCleanup(cache.close)
        return cache

    def fill(self, cache, rows):
        out = torch.empty((len(rows), cache.row_bytes), dtype=torch.uint8)
        cache.fill(rows, out)
        return out

    def test_index_only_and_filter_never_gets_ple_payload(self):
        path, name, _ = self.fixture()
        with mock.patch.object(
            self.module.os, "preadv", side_effect=AssertionError("payload read")
        ):
            desc = self.module.read_ple_checkpoint_index(str(path))[name]
        self.assertEqual(desc.shape, (8, 4))
        common = load_source(
            "ple_common_test", "python/sglang/srt/model_loader/qwen38_nvfp4_common.py"
        )
        from .test_cpu_loader_filter import _FakeSafeOpen

        handle = _FakeSafeOpen([name, "model.language_model.norm.weight"])
        with mock.patch.dict(
            "sys.modules", {"sglang.srt.models.qwen4_exp_ple_lazy": self.module}
        ), mock.patch("safetensors.safe_open", return_value=handle):
            result = list(
                common.filtered_safetensors_weights_iterator([str(path)], lazy_ple=True)
            )
        self.assertIsInstance(result[0][1], self.module.PleCheckpointShard)
        self.assertEqual(handle.requested, ["model.language_model.norm.weight"])

    def test_cross_shard_and_fp8_byte_parity(self):
        for dtype in ("BF16", "F8_E4M3"):
            paths = [self.fixture(dtype=dtype, shard=i) for i in range(2)]
            shards = [
                (i * 8, (i + 1) * 8, self.module.read_ple_checkpoint_index(str(p))[n])
                for i, (p, n, _) in enumerate(paths)
            ]
            row_bytes = 8 if dtype == "BF16" else 4
            cache = self.cache(shards, row_bytes=row_bytes)
            actual = self.fill(cache, [7, 8, 3, 15])
            raw = np.frombuffer(paths[0][2], dtype=np.uint8).reshape(8, row_bytes)
            np.testing.assert_array_equal(actual.numpy(), raw[[7, 0, 3, 7]])
            # An all-hit lookup does no I/O and preserves bytes exactly.
            with mock.patch.object(
                self.module.os, "preadv", side_effect=AssertionError("cache hit read")
            ):
                np.testing.assert_array_equal(
                    self.fill(cache, [15, 3]).numpy(), raw[[7, 3]]
                )

    def test_scan_does_not_evict_protected_hot_rows(self):
        path, name, raw = self.fixture(rows=32)
        cache = self.cache(
            [(0, 32, self.module.read_ple_checkpoint_index(str(path))[name])]
        )
        self.fill(cache, [0, 1, 2, 3])
        self.fill(cache, [0, 1])
        # Large one-time scans are chunked to fit the cache. Hot rows survive.
        for rows in ([4, 5], [6, 7], [8, 9]):
            self.fill(cache, rows)
        self.assertIn(0, cache.protected)
        self.assertIn(1, cache.protected)
        before = cache.misses
        self.fill(cache, [0, 1])
        self.assertEqual(cache.misses, before)
        self.assertLessEqual(
            len(cache.probation) + len(cache.protected), cache.capacity
        )
        # A batch that evicts its own old hits still returns correct staging.
        actual = self.fill(cache, [0, 10, 11, 12])
        expected = np.frombuffer(raw, dtype=np.uint8).reshape(32, 8)[[0, 10, 11, 12]]
        np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual(
            len(set(cache.probation.values()) | set(cache.protected.values())), 4
        )

    def test_short_read_and_checkpoint_identity_fail(self):
        path, name, _ = self.fixture()
        desc = self.module.read_ple_checkpoint_index(str(path))[name]
        cache = self.cache([(0, 8, desc)])
        with mock.patch.object(self.module.os, "preadv", return_value=0):
            with self.assertRaisesRegex(OSError, "short"):
                self.fill(cache, [3])
        self.assertFalse(cache.probation)
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.cache([(0, 8, desc)])

    def test_truncated_header_and_offsets_rejected(self):
        path, name, _ = self.fixture()
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "offsets"):
            self.module.read_ple_checkpoint_index(str(path))
        path.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.module.read_ple_checkpoint_index(str(path))

    def test_parallel_reads_and_partial_pread(self):
        path, name, raw = self.fixture(rows=20000)
        desc = self.module.read_ple_checkpoint_index(str(path))[name]
        cache = self.cache([(0, 20000, desc)], capacity=16)
        rows = list(range(0, 16000, 1000))
        original = self.module.os.preadv

        def partial(fd, buffers, offset):
            return original(fd, [buffers[0][:3]], offset)

        with mock.patch.object(self.module.os, "preadv", side_effect=partial):
            out = self.fill(cache, rows)
        np.testing.assert_array_equal(
            out.numpy(), np.frombuffer(raw, dtype=np.uint8).reshape(20000, 8)[rows]
        )
        self.assertEqual(cache.read_bytes, 16 * 8)
        self.assertEqual(cache.read_calls, 16 * 3)
