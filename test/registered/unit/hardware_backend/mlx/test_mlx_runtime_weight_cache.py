"""Prepared-weight cache must restore decoded operands and reject torn entries."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlxRuntimeWeightCache(CustomTestCase):
    def test_decoded_projection_survives_roundtrip(self):
        """Dropping KV-B's decoded operand makes attention consume packed codes."""
        import mlx.core as mx
        import mlx.nn as nn

        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear
        from sglang.srt.hardware_backend.mlx.models.mlx_fp4_cache import (
            RuntimeWeightCache,
        )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            cache = object.__new__(RuntimeWeightCache)
            cache.path, cache.identity = Path(directory), "regression"
            source = nn.Module()
            source.self_attn = nn.Module()
            source.self_attn.kv_b_proj = FP8Linear(
                mx.full((128, 128), 56, mx.uint8),
                mx.full((128, 4), 127, mx.uint8),
                block_size=32,
                activation_mode="a16",
            )
            expected = mx.arange(128 * 128).reshape(128, 128).astype(mx.bfloat16)
            source.self_attn.kv_b_proj.decoded_weight = expected
            cache.save("layer", source)
            restored = nn.Module()
            restored.self_attn = nn.Module()
            restored.self_attn.kv_b_proj = nn.Linear(128, 128, bias=False)
            self.assertTrue(cache.load("layer", restored))
            projection = restored.self_attn.kv_b_proj
            self.assertIn("decoded_weight", projection)
            self.assertEqual(projection.decoded_weight.dtype, expected.dtype)
            self.assertTrue(mx.array_equal(projection.decoded_weight, expected).item())
            self.assertTrue(
                mx.array_equal(
                    projection.weight, source.self_attn.kv_b_proj.weight
                ).item()
            )

    def test_torn_manifest_and_corrupt_file_are_misses(self):
        import mlx.core as mx
        import mlx.nn as nn

        from sglang.srt.hardware_backend.mlx.models.mlx_fp4_cache import (
            RuntimeWeightCache,
        )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            cache = object.__new__(RuntimeWeightCache)
            cache.path, cache.identity = Path(directory), "regression"
            source = nn.Module()
            source.weight = mx.arange(32).astype(mx.float32)
            cache.save("layer", source)
            meta_file = cache.path / "layer.json"
            meta = json.loads(meta_file.read_text())
            meta["generation"] = "different-writer"
            meta_file.write_text(json.dumps(meta))
            target = nn.Module()
            target.weight = mx.zeros_like(source.weight)
            self.assertFalse(cache.load("layer", target))
            self.assertTrue(mx.all(target.weight == 0).item())
            cache.save("layer", source)
            file = cache.path / "layer.safetensors"
            file.write_bytes(b"x" * file.stat().st_size)
            self.assertFalse(cache.load("layer", target))
            self.assertTrue(mx.all(target.weight == 0).item())


if __name__ == "__main__":
    unittest.main()
