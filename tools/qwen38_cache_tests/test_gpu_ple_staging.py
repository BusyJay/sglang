"""Lazy PLE staging must remain valid across calls, chunks, and CUDA streams."""

import os
import tempfile
import unittest
from unittest import mock

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPleStaging(unittest.TestCase):
    def test_double_staging_across_streams_and_chunk_tails(self):
        import importlib.util
        from pathlib import Path

        from safetensors.torch import save_file

        from sglang.srt.models import qwen4_exp as q
        from sglang.srt.models.qwen4_exp_ple_lazy import read_ple_checkpoint_index
        from sglang.srt.runtime_context import get_parallel

        # Python's standard-library ``test`` package can shadow this checkout.
        fixture_path = Path(__file__).resolve().parents[2] / (
            "test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ple_staging_fixture", fixture_path
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        _make_source_embedding = fixture._make_source_embedding

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                rows = (torch.arange(32 * 160).reshape(32, 160).float() / 64).to(dtype)
                name = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
                path = os.path.join(tmp, "ple.safetensors")
                save_file({name: rows}, path)
                with get_parallel().override(tp_size=1):
                    source = _make_source_embedding(
                        dtype=dtype, embedding_dim=160, org_vocab_size=32, vocab_end=32
                    )
                    e = q.Qwen4ExpPinnedHostEmbedding(source, backend="lazy")
                e.bind_lazy_shard(read_ple_checkpoint_index(path)[name], 0, 32)
                with (
                    mock.patch.dict(
                        os.environ, {"SGLANG_QWEN38_PLE_DOUBLE_STAGING": "1"}
                    ),
                    mock.patch.object(
                        q.envs.SGLANG_QWEN4_PLE_LAZY_CACHE_MB, "get", return_value=1
                    ),
                    mock.patch.object(
                        q.envs.SGLANG_QWEN4_PLE_LAZY_BATCH_ROWS, "get", return_value=3
                    ),
                ):
                    e.finalize_lazy()
                outputs = []
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                try:
                    for step in range(16):
                        values = [(step * 7 + i) % 32 for i in range(8)] + [-1, 32]
                        with torch.cuda.stream(streams[step % 2]):
                            ids = torch.tensor(values, device="cuda")
                            outputs.append((e.gather(ids), values))
                    torch.cuda.synchronize()
                    for actual, values in outputs:
                        expected = torch.zeros(10, 160, dtype=torch.bfloat16)
                        for i, row in enumerate(values):
                            if 0 <= row < 32:
                                expected[i] = rows[row].to(torch.bfloat16)
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                finally:
                    torch.cuda.synchronize()
                    e._lazy_cache.close()
