"""Persistent PLE cache publication/reuse and direct buffered read correctness."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source

cache = load_source(
    "packed_ple_cache_test", "python/sglang/srt/models/qwen38_ple_packed_cache.py"
)


class PackedPleCacheTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / "source.bin"
        path.write_bytes(b"0123456789")
        st = path.stat()
        desc = SimpleNamespace(
            path=str(path),
            shape=(2, 8),
            dtype="u8",
            offset=0,
            identity=(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns),
        )
        return [(0, 2, desc)]

    def test_build_reuse_and_direct_read(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.fixture(directory)

            def writer(view):
                view[:] = bytes(range(18))

            path, hit, _ = cache.ensure_packed_ple_cache(
                directory, shards, shards, 2, 16, 0, writer
            )
            self.assertFalse(hit)
            fail = mock.Mock(side_effect=AssertionError("writer must not run"))
            _, hit, _ = cache.ensure_packed_ple_cache(
                directory, shards, shards, 2, 16, 0, fail
            )
            self.assertTrue(hit)
            destination = bytearray(18)
            cache.read_packed_ple_cache(path, destination, chunk_bytes=5)
            self.assertEqual(destination, bytes(range(18)))
            with self.assertRaises(ValueError):
                cache.read_packed_ple_cache(path, bytearray(17))

    def test_concurrent_builders_publish_once(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.fixture(directory)
            writer = mock.Mock(
                side_effect=lambda view: view.__setitem__(slice(None), bytes(18))
            )

            def build():
                return cache.ensure_packed_ple_cache(
                    directory, shards, shards, 2, 16, 0, writer
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: build(), range(2)))
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(sorted(hit for _, hit, _ in results), [False, True])

    def test_failed_build_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.fixture(directory)

            def fail(view):
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                cache.ensure_packed_ple_cache(directory, shards, shards, 2, 16, 0, fail)
            self.assertEqual(list(Path(directory).glob("fp4-ple*.bin")), [])
            cache.ensure_packed_ple_cache(
                directory,
                shards,
                shards,
                2,
                16,
                0,
                lambda view: view.__setitem__(slice(None), bytes(18)),
            )

    def test_truncation_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.fixture(directory)
            writer = lambda view: view.__setitem__(slice(None), bytes(18))
            path, _, _ = cache.ensure_packed_ple_cache(
                directory, shards, shards, 2, 16, 0, writer
            )
            path.write_bytes(b"bad")
            _, hit, _ = cache.ensure_packed_ple_cache(
                directory, shards, shards, 2, 16, 0, writer
            )
            self.assertFalse(hit)
            Path(shards[0][2].path).write_bytes(b"changed size")
            with self.assertRaises(ValueError):
                cache.ensure_packed_ple_cache(
                    directory, shards, shards, 2, 16, 0, writer
                )


if __name__ == "__main__":
    unittest.main()
