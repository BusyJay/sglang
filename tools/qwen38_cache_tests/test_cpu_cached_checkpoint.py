# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from .test_cpu_loader_filter import common_module as common


class LocalEntryNotFoundError(Exception):
    pass


class TestCachedCheckpoint(unittest.TestCase):
    def test_directory_skips_hub_entirely(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict("sys.modules", {"huggingface_hub": None}):
                self.assertEqual(common.resolve_cached_checkpoint(root), Path(root))

    def test_repo_resolution_is_local_only_and_preserves_revision(self):
        snapshot = mock.Mock(return_value="/cache/snapshots/abc")
        with mock.patch.dict(
            "sys.modules",
            {
                "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
                "huggingface_hub.errors": SimpleNamespace(
                    LocalEntryNotFoundError=LocalEntryNotFoundError
                ),
            },
        ):
            self.assertEqual(
                common.resolve_cached_checkpoint(
                    "nvidia/model", revision="release", cache_dir="/cache"
                ),
                Path("/cache/snapshots/abc"),
            )
        snapshot.assert_called_once_with(
            repo_id="nvidia/model",
            revision="release",
            cache_dir="/cache",
            local_files_only=True,
        )

    def test_missing_cache_errors_without_retry_or_online_fallback(self):
        snapshot = mock.Mock(side_effect=LocalEntryNotFoundError())
        with mock.patch.dict(
            "sys.modules",
            {
                "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
                "huggingface_hub.errors": SimpleNamespace(
                    LocalEntryNotFoundError=LocalEntryNotFoundError
                ),
            },
        ):
            with self.assertRaisesRegex(ValueError, "No download was attempted"):
                common.resolve_cached_checkpoint("nvidia/model")
        self.assertEqual(snapshot.call_count, 1)

    def test_index_missing_shard_and_broken_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}})
            )
            (path / "a.safetensors").write_bytes(b"fixture")
            (path / "b.safetensors").symlink_to(path / "absent-blob")
            with self.assertRaisesRegex(ValueError, "missing shard"):
                common.cached_checkpoint_weight_files(path)
            (path / "absent-blob").write_bytes(b"fixture")
            self.assertEqual(
                common.cached_checkpoint_weight_files(path),
                [str(path / "a.safetensors"), str(path / "b.safetensors")],
            )

    def test_unindexed_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with self.assertRaisesRegex(ValueError, "no safetensors"):
                common.cached_checkpoint_weight_files(path)
            (path / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(
                common.cached_checkpoint_weight_files(path),
                [str(path / "model.safetensors")],
            )

    def test_nested_shards_allow_hf_blob_links_but_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "snapshot"
            (path / "tensors").mkdir(parents=True)
            blob = Path(root) / "blob"
            blob.write_bytes(b"fixture")
            shard = path / "tensors" / "model.safetensors"
            shard.symlink_to(blob)
            index = path / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"x": "tensors/model.safetensors"}})
            )
            self.assertEqual(common.cached_checkpoint_weight_files(path), [str(shard)])
            for name in (
                "../blob",
                str(blob),
                "tensors/../model.safetensors",
                "",
                "tensors",
            ):
                with self.subTest(name=name):
                    index.write_text(json.dumps({"weight_map": {"x": name}}))
                    with self.assertRaisesRegex(
                        ValueError, "invalid safetensors shard name"
                    ):
                        common.cached_checkpoint_weight_files(path)

    def test_snapshot_config_and_pack_must_agree(self):
        revision = "a" * 40
        config = SimpleNamespace(_commit_hash=revision)
        common.validate_cached_checkpoint_revision(
            f"/cache/snapshots/{revision}", config, revision
        )
        for path, cfg, pack in (
            (f"/cache/snapshots/{'b' * 40}", config, revision),
            (
                f"/cache/snapshots/{revision}",
                SimpleNamespace(_commit_hash="b" * 40),
                revision,
            ),
            ("/local/model", config, "b" * 40),
        ):
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                common.validate_cached_checkpoint_revision(path, cfg, pack)
        common.validate_cached_checkpoint_revision(
            "/local/model", SimpleNamespace(), "fixture"
        )
