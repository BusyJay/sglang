"""Shared source publication, invalidation and failure recovery on real tmpfs."""

import concurrent.futures
import gc
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ._load_source import load_source

cache = load_source(
    "qwen38_source_cache_under_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/source_cache.py",
)


@unittest.skipUnless(Path("/dev/shm").is_dir(), "Linux tmpfs required")
class TestSharedSourceCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.bin"
        self.data = bytes(range(251)) * 35001  # Cross the 8 MiB read boundary.
        self.source.write_bytes(self.data)
        self.stream = self.source.open("r+b", buffering=0)
        self.addCleanup(self.stream.close)
        self.stop = threading.Event()
        self.destination = self.root / "cache"

    def read(self):
        return cache.map_shared_source(
            self.stream.fileno(), self.destination, self.stop
        )

    def test_reopen_reuses_bytes_without_source_io(self):
        mapping, hit = self.read()
        self.assertFalse(hit)
        self.assertEqual(mapping[:], self.data)
        mapping.close()
        with mock.patch.object(
            cache.os, "preadv", side_effect=AssertionError("cache miss")
        ):
            mapping, hit = self.read()
        try:
            self.assertTrue(hit)
            self.assertEqual(mapping[:], self.data)
        finally:
            mapping.close()
        self.assertEqual(self.source.read_bytes(), self.data)

    def test_changed_source_cannot_reuse_old_cache(self):
        mapping, _ = self.read()
        mapping.close()
        self.stream.seek(0)
        self.stream.write(b"changed!")
        mapping, hit = self.read()
        try:
            self.assertFalse(hit)
            self.assertEqual(mapping[:], self.source.read_bytes())
        finally:
            mapping.close()

    def test_truncated_and_abandoned_files_are_rebuilt(self):
        mapping, _ = self.read()
        mapping.close()
        path = next(self.destination.glob("*.bin"))
        path.write_bytes(b"short")
        abandoned = path.with_name(path.name + ".old.tmp")
        abandoned.write_bytes(b"abandoned")
        mapping, hit = self.read()
        try:
            self.assertFalse(hit)
            self.assertEqual(mapping[:], self.data)
            self.assertFalse(abandoned.exists())
        finally:
            mapping.close()

    def test_failed_copy_is_not_published_and_can_retry(self):
        with mock.patch.object(cache.os, "preadv", return_value=0):
            with self.assertRaisesRegex(OSError, "short expert source read"):
                self.read()
        self.assertEqual(list(self.destination.glob("*.bin")), [])
        self.assertEqual(list(self.destination.glob("*.tmp")), [])
        mapping, hit = self.read()
        try:
            self.assertFalse(hit)
            self.assertEqual(mapping[:], self.data)
        finally:
            mapping.close()

    def test_concurrent_readers_publish_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.read(), range(2)))
        try:
            self.assertEqual(sorted(hit for _, hit in results), [False, True])
            for mapping, _ in results:
                self.assertEqual(mapping[:], self.data)
        finally:
            for mapping, _ in results:
                mapping.close()

    def test_cancelled_copy_is_not_published(self):
        original = cache.os.preadv

        def cancel_after_first_chunk(*args):
            result = original(*args)
            self.stop.set()
            return result

        with mock.patch.object(
            cache.os, "preadv", side_effect=cancel_after_first_chunk
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.read()
        self.assertEqual(list(self.destination.glob("*.bin")), [])
        self.assertEqual(list(self.destination.glob("*.tmp")), [])

    def test_disk_cache_rejected_before_mapping(self):
        with self.assertRaisesRegex(ValueError, "must be on tmpfs"):
            cache.map_shared_source(self.stream.fileno(), "/", self.stop)

    def test_source_changed_during_copy_is_not_published(self):
        original = cache.os.preadv

        def change_source(*args):
            count = original(*args)
            self.stream.seek(0)
            self.stream.write(b"changed!")
            return count

        with mock.patch.object(cache.os, "preadv", side_effect=change_source):
            with self.assertRaisesRegex(RuntimeError, "Expert pack changed"):
                self.read()
        self.assertEqual(list(self.destination.glob("*.bin")), [])
        self.assertEqual(list(self.destination.glob("*.tmp")), [])

    def test_insufficient_space_fails_before_allocating(self):
        space = mock.Mock(f_bavail=0, f_frsize=4096)
        with mock.patch.object(cache.os, "statvfs", return_value=space):
            with self.assertRaisesRegex(OSError, "Not enough tmpfs space"):
                self.read()
        self.assertEqual(list(self.destination.glob("*.bin")), [])
        self.assertEqual(list(self.destination.glob("*.tmp")), [])

    def test_directory_cleanup_preserves_matching_data_and_reuses_it(self):
        mapping, _ = self.read()
        mapping.close()
        del mapping
        gc.collect()
        path = next(self.destination.glob("*.bin"))
        garbage = self.destination / "arbitrary-old-container-data"
        garbage.write_bytes(b"garbage")
        abandoned = path.with_name(path.name + ".interrupted.tmp")
        abandoned.write_bytes(b"unfinished")
        report = cache.clean_cache_directory(
            self.destination, {path.name: len(self.data)}
        )
        self.assertEqual(report["removed_files"], 2)
        self.assertTrue(path.exists())
        self.assertFalse(garbage.exists())
        self.assertFalse(abandoned.exists())
        with mock.patch.object(
            cache.os, "preadv", side_effect=AssertionError("must reuse")
        ):
            mapping, hit = self.read()
        try:
            self.assertTrue(hit)
            self.assertEqual(mapping[:], self.data)
        finally:
            mapping.close()

    def test_cleanup_never_removes_mapped_source_then_reclaims_after_exit(self):
        mapping, _ = self.read()
        path = next(self.destination.glob("*.bin"))
        report = cache.clean_cache_directory(self.destination, {})
        self.assertEqual(report["in_use"], 1)
        self.assertTrue(path.exists())
        self.assertEqual(mapping[:], self.data)
        mapping.close()
        del mapping
        gc.collect()
        report = cache.clean_cache_directory(self.destination, {})
        self.assertEqual(report["removed_files"], 1)
        self.assertFalse(path.exists())

    def test_cleanup_does_not_interrupt_active_writer(self):
        import fcntl

        self.destination.mkdir()
        path = self.destination / "source.bin"
        abandoned = path.with_name(path.name + ".writing.tmp")
        abandoned.write_bytes(b"in progress")
        with path.with_suffix(".lock").open("a+b") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX)
            report = cache.clean_cache_directory(self.destination, {})
            self.assertEqual(report["in_use"], 1)
            self.assertTrue(abandoned.exists())
        cache.clean_cache_directory(self.destination, {})
        self.assertFalse(abandoned.exists())


if __name__ == "__main__":
    unittest.main()
