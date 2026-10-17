"""Shared packed PLE: exact gather, graph replay, reuse and safe cleanup."""

import errno
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSharedPle(unittest.TestCase):
    def fixture(self, directory):
        from safetensors.torch import save_file
        from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index

        path = Path(directory) / "checkpoint.safetensors"
        prefix = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0."
        generator = torch.Generator().manual_seed(20261018)
        save_file(
            {
                prefix
                + "weight": torch.randint(
                    0, 256, (128, 80), dtype=torch.uint8, generator=generator
                ),
                prefix
                + "weight_scale": torch.full((128, 10), 0.5, dtype=torch.float8_e4m3fn),
            },
            str(path),
        )
        desc = read_ple_checkpoint_index(str(path))
        return path, desc[prefix + "weight"], desc[prefix + "weight_scale"]

    def embedding(self, directory, descriptors, shared=None):
        from sglang.srt.models.qwen38_fp4_ple import Qwen38Fp4PleEmbedding

        path, weights, scales = descriptors
        e = Qwen38Fp4PleEmbedding.__new__(Qwen38Fp4PleEmbedding)
        torch.nn.Module.__init__(e)
        e.embedding_dim = 160
        e.org_vocab_size = e._checkpoint_rows = 128
        e._backend = "pinned"
        e._table_dir = str(Path(directory) / "packed")
        e._lazy_shards, e._scale_shards = [(0, 128, weights)], [(0, 128, scales)]
        e._lazy, e._packed_ready, e._fp4_ple = False, False, True
        e._shared_ple = None
        e._shared_cache_spec = None
        if shared is not None:
            st = path.stat()
            checkpoint = [
                (
                    str(path),
                    st.st_dev,
                    st.st_ino,
                    st.st_size,
                    st.st_mtime_ns,
                    st.st_ctime_ns,
                )
            ]
            e._shared_cache_spec = (str(shared), "c" * 64, checkpoint)
        e._block_d, e.tp_size = 256, 1
        e.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=128
        )
        e.weight = torch.nn.Parameter(
            torch.empty(128, 80, device="meta", dtype=torch.uint8), requires_grad=False
        )
        e.register_buffer("fp4_global_scale", torch.tensor([0.0123], device="cuda"))
        e.finalize_lazy()
        self.addCleanup(e.close)
        return e

    def test_gather_graph_reopen_and_active_lease(self):
        from sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache import (
            clean_cache_directory,
        )

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory(
            dir="/dev/shm"
        ) as shared:
            descriptors = self.fixture(directory)
            baseline = self.embedding(directory, descriptors)
            e = self.embedding(directory, descriptors, shared)
            self.assertEqual(e.weight.data_ptr(), e._shared_ple.raw.data_ptr())
            ids = torch.tensor([0, 127, 42, -1, 128, 0], device="cuda")
            reference = baseline.gather(ids)
            result = e.gather(ids)
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                cross = e.gather(ids.flip(0))
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(cross, reference.flip(0), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = e.gather(ids)
            ids.copy_(ids.flip(0))
            graph.replay()
            torch.testing.assert_close(captured, reference.flip(0), rtol=0, atol=0)
            report = clean_cache_directory(shared, {})
            self.assertEqual(report["in_use"], 1)
            self.assertEqual(len(list(Path(shared).glob("*.bin"))), 1)
            e.close()
            with mock.patch(
                "sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache.os.preadv",
                side_effect=AssertionError("reopen must not reread"),
            ):
                reopened = self.embedding(directory, descriptors, shared)
            torch.testing.assert_close(
                reopened.gather(ids), reference.flip(0), rtol=0, atol=0
            )
            reopened.close()
            gc.collect()
            clean_cache_directory(shared, {})
            self.assertEqual(list(Path(shared).glob("*.bin")), [])

    def test_resource_failure_falls_back_to_original_pinned_table(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory(
            dir="/dev/shm"
        ) as shared:
            descriptors = self.fixture(directory)
            baseline = self.embedding(directory, descriptors)
            with mock.patch(
                "sglang.srt.model_loader.qwen38_nvfp4_pack.source_cache.map_shared_source",
                side_effect=OSError(errno.ENOSPC, "injected tmpfs exhaustion"),
            ):
                fallback = self.embedding(directory, descriptors, shared)
            self.assertIsNone(fallback._shared_ple)
            self.assertTrue(fallback.weight.is_pinned())
            ids = torch.tensor([3, 127, -1, 128], device="cuda")
            torch.testing.assert_close(
                fallback.gather(ids), baseline.gather(ids), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
