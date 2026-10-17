from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ._load_source import load_source

format_module = load_source(
    "qwen38_pack_format_under_test",
    "python/sglang/srt/model_loader/qwen38_nvfp4_pack/format.py",
)
FORMAT_NAME = format_module.FORMAT_NAME
REQUIRED_TENSORS = format_module.PACK_TENSORS
Qwen38ExpertPack = format_module.Qwen38ExpertPack
load_manifest = format_module.load_manifest


def make_pack(root: Path, *, overlap: bool = False) -> Path:
    payload = root / "experts.bin"
    payload.write_bytes(bytes(range(96)))
    offset = 0
    layers = {}
    records = {}
    for index, name in enumerate(REQUIRED_TENSORS):
        if overlap and index == 1:
            current = 0
        else:
            current = offset
        records[name] = {
            "offset": current,
            "nbytes": 4 if name in format_module.LAYER_TENSORS else 8,
            "shape": [1] if name in format_module.LAYER_TENSORS else [2, 4],
            "dtype": "float32" if name in format_module.LAYER_TENSORS else "uint8",
        }
        offset += 8
    layers["0"] = records
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": FORMAT_NAME,
                "version": format_module.FORMAT_VERSION,
                "payload": payload.name,
                "checkpoint_revision": "fixture-revision",
                "sglang_commit": "e00930c5489053f26d86b179cee0d087f846acbb",
                "flashinfer_backend": "flashinfer_cutlass",
                "compute_capability": "sm120",
                "model": {
                    "num_layers": 1,
                    "num_experts": 2,
                    "top_k": 2,
                    "hidden_size": 2560,
                    "intermediate_size": 640,
                },
                "layers": layers,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class TestPackFormat(unittest.TestCase):
    def test_manifest_and_read_only_mmap(self):
        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            manifest.assert_runtime_compatible(
                checkpoint_revision="fixture-revision",
                sglang_commit="e00930c5489053f26d86b179cee0d087f846acbb",
            )
            with Qwen38ExpertPack(manifest) as pack:
                self.assertEqual(
                    bytes(pack.tensor_bytes(0, "w13_weight")), bytes(range(8))
                )
                self.assertEqual(
                    bytes(pack.expert_tensor_bytes(0, 1, "w13_weight")),
                    bytes(range(4, 8)),
                )
                with self.assertRaises(TypeError):
                    pack.tensor_bytes(0, "w13_weight")[0] = 9

    def test_background_prefetch_finishes_and_pack_remains_readable(self):
        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            with Qwen38ExpertPack(manifest) as pack:
                pack.start_prefetch()
                pack.start_prefetch()
                pack.wait_prefetch()
                self.assertIsNone(pack._prefetch_error)
                self.assertEqual(
                    bytes(pack.tensor_bytes(0, "w13_weight")), bytes(range(8))
                )

    @mock.patch.dict(os.environ, {"SGLANG_QWEN38_SOURCE_MODE": "file_cow"})
    def test_cuda_source_replaces_mapping_and_preserves_read_only_views(self):
        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            original = manifest.payload_path.read_bytes()
            with Qwen38ExpertPack(manifest) as pack:
                old = pack._mapping
                shared = pack.cuda_source_mapping()
                self.assertTrue(old.closed)
                self.assertIs(shared, pack._mapping)
                self.assertIs(shared, pack.cuda_source_mapping())
                for view in (
                    pack.tensor_bytes(0, "w13_weight"),
                    pack.expert_tensor_bytes(0, 1, "w13_weight"),
                ):
                    try:
                        self.assertTrue(view.readonly)
                        with self.assertRaises(TypeError):
                            view[0] = 9
                    finally:
                        view.release()
                self.assertEqual(manifest.payload_path.read_bytes(), original)

    @mock.patch.dict(os.environ, {"SGLANG_QWEN38_SOURCE_MODE": "file_cow"})
    def test_cuda_source_promotion_keeps_existing_views_valid_on_failure(self):
        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            with Qwen38ExpertPack(manifest) as pack:
                old = pack._mapping
                view = pack.tensor_bytes(0, "w13_weight")
                try:
                    with self.assertRaises(BufferError):
                        pack.cuda_source_mapping()
                    self.assertIs(pack._mapping, old)
                    self.assertFalse(old.closed)
                    self.assertEqual(bytes(view), bytes(range(8)))
                finally:
                    view.release()
                self.assertIs(pack.cuda_source_mapping(), pack._mapping)

    def test_prepared_sources_are_readable_before_registration(self):
        for mode in ("cow_prefault", "anonymous", "anonymous_thp"):
            for background in (False, True):
                with self.subTest(
                    mode=mode, background=background
                ), tempfile.TemporaryDirectory() as value:
                    manifest = load_manifest(make_pack(Path(value)))
                    original = manifest.payload_path.read_bytes()
                    with mock.patch.dict(
                        os.environ, {"SGLANG_QWEN38_SOURCE_MODE": mode}
                    ), Qwen38ExpertPack(manifest) as pack:
                        if background:
                            pack.start_prefetch()
                        # CPU scale/weight reads may precede runtime registration.
                        view = pack.tensor_bytes(0, "w13_weight")
                        try:
                            self.assertTrue(view.readonly)
                            self.assertEqual(bytes(view), bytes(range(8)))
                        finally:
                            view.release()
                        mapping = pack.cuda_source_mapping()
                        self.assertEqual(bytes(mapping), original)
                        self.assertIs(mapping, pack.cuda_source_mapping())
                    self.assertEqual(manifest.payload_path.read_bytes(), original)

    def test_shared_source_first_cpu_read_and_reopen(self):
        with tempfile.TemporaryDirectory() as value, tempfile.TemporaryDirectory(
            dir="/dev/shm"
        ) as cache:
            manifest = load_manifest(make_pack(Path(value)))
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_QWEN38_SOURCE_MODE": "shared_thp",
                    "SGLANG_QWEN38_SOURCE_CACHE_DIR": cache,
                },
            ):
                for background in (False, True):
                    with Qwen38ExpertPack(manifest) as pack:
                        if background:
                            pack.start_prefetch()
                        view = pack.tensor_bytes(0, "w13_weight")
                        try:
                            self.assertTrue(view.readonly)
                            self.assertEqual(bytes(view), bytes(range(8)))
                        finally:
                            view.release()
                        self.assertEqual(pack._source_cache_hit, background)
                        self.assertIs(pack.cuda_source_mapping(), pack._mapping)
                        self.assertEqual(pack._mapping[:], bytes(range(96)))

    def test_auto_shared_source_resource_failure_loads_original_bytes(self):
        from sglang.srt.model_loader.qwen38_nvfp4_pack import source_cache

        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            with mock.patch.object(
                source_cache,
                "map_shared_source",
                side_effect=OSError(errno.ENOSPC, "injected exhaustion"),
            ):
                with Qwen38ExpertPack(
                    manifest, source_mode="shared_thp", allow_shared_fallback=True
                ) as pack:
                    self.assertEqual(pack.cuda_source_mapping()[:], bytes(range(96)))
                    self.assertEqual(pack._source_mode, "anonymous_thp")

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_manifest(make_pack(Path(value), overlap=True))

    def test_abi_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            manifest = load_manifest(make_pack(Path(value)))
            with self.assertRaisesRegex(ValueError, "ABI mismatch"):
                manifest.assert_runtime_compatible(sglang_commit="different")


if __name__ == "__main__":
    unittest.main()
