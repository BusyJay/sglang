# SPDX-License-Identifier: Apache-2.0
"""Vision patch/frame order and absolute embedding spans are model contracts."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")


class TestVisionMlx(CustomTestCase):
    def test_absolute_spans_cross_chunks_without_sentinel_lookup(self):
        import mlx.core as mx
        from mlx import nn
        from sglang.srt.hardware_backend.mlx.vision import prepare_vision_prompt

        values = mx.arange(24).reshape(6, 4).astype(mx.float32)
        mm = SimpleNamespace(
            mm_items=[
                SimpleNamespace(offsets=[(2, 5), (8, 9)], precomputed_embeddings=values)
            ]
        )
        prompt = prepare_vision_prompt(mm, None)
        embedding = nn.Embedding(16, 4)
        ids = [1, 2, 2**50, 2**50, 2**50, 2**50, 3, 4, 2**50, 2**50, 5]
        whole = prompt.embed(embedding, mx.array([ids], mx.int64), 0)
        chunks = []
        for start, end in [(0, 3), (3, 7), (7, 10), (10, 11)]:
            inputs = mx.array([ids[start:end]], mx.int64)
            got = prompt.embed(embedding, inputs, start)
            chunks.append(embedding(inputs) if got is None else got)
        mx.eval(whole, chunks)
        self.assertTrue(mx.array_equal(whole, mx.concatenate(chunks, 1)).item())
        self.assertTrue(mx.array_equal(whole[:, 2:6], values[:4][None]).item())
        self.assertTrue(mx.array_equal(whole[:, 8:10], values[4:][None]).item())
        self.assertIsNone(prompt.embed(embedding, mx.array([[3, 4]]), 6))

    def test_span_validation_rejects_mismatched_or_overlapping_embeddings(self):
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.vision import prepare_vision_prompt

        for offsets, count in [([(1, 3)], 2), ([(1, 3), (3, 4)], 5), ([(-1, 1)], 3)]:
            mm = SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        offsets=offsets, precomputed_embeddings=mx.ones((count, 4))
                    )
                ]
            )
            with self.assertRaises(ValueError):
                prepare_vision_prompt(mm, None)

    def test_patch_and_merger_convolutions_match_cpu_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        import torch.nn.functional as F
        from sglang.srt.hardware_backend.mlx.models.glm5_vision import Glm5Vision

        cfg = {
            "hidden_size": 8,
            "num_heads": 2,
            "attention_bias": True,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 4,
            "depth": 0,
            "rms_norm_eps": 1e-5,
            "intermediate_size": 12,
            "projection_intermediate_size": 6,
            "swiglu_limit": 10.0,
        }
        mx.random.seed(91)
        m = Glm5Vision(cfg)
        x = mx.random.normal((16, 24))
        cpu = lambda a: torch.from_numpy(np.array(a))
        patches = F.conv3d(
            cpu(x).reshape(-1, 3, 2, 2, 2),
            cpu(m.patch_embed.weight).reshape(8, 3, 2, 2, 2),
            cpu(m.patch_embed.bias),
        ).reshape(-1, 8)
        np.testing.assert_allclose(
            np.array(m.patch_embed(x)), patches.numpy(), rtol=1e-5, atol=1e-5
        )
        normed = m.post_layernorm(m.patch_embed(x))
        arranged = normed.reshape(-1, 4, 8)
        actual = m.downsample(arranged.transpose(0, 2, 1).reshape(-1, 32))
        expected = F.conv2d(
            cpu(normed).reshape(-1, 2, 2, 8).permute(0, 3, 1, 2),
            cpu(m.downsample.weight).reshape(4, 8, 2, 2),
            cpu(m.downsample.bias),
        ).reshape(-1, 4)
        np.testing.assert_allclose(
            np.array(actual), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_packed_frames_equal_separate_encoder_calls(self):
        import mlx.core as mx
        import numpy as np
        from sglang.srt.hardware_backend.mlx.models.glm5_vision import Glm5Vision

        cfg = {
            "hidden_size": 32,
            "num_heads": 4,
            "attention_bias": True,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 16,
            "depth": 2,
            "rms_norm_eps": 1e-5,
            "intermediate_size": 48,
            "projection_intermediate_size": 24,
            "swiglu_limit": 10.0,
        }
        mx.random.seed(819)
        model = Glm5Vision(cfg)
        a, b = mx.random.normal((16, 24)), mx.random.normal((24, 24))
        packed = model(mx.concatenate([a, b]), np.array([[1, 4, 4], [1, 4, 6]]))
        independent = mx.concatenate(
            [model(a, np.array([[1, 4, 4]])), model(b, np.array([[1, 4, 6]]))]
        )
        mx.eval(packed, independent)
        np.testing.assert_allclose(
            np.array(packed), np.array(independent), rtol=1e-5, atol=1e-5
        )
        # Two temporal frames must also avoid attending across the boundary.
        video = model(mx.concatenate([a, a]), np.array([[2, 4, 4]]))
        np.testing.assert_allclose(
            np.array(video),
            np.concatenate([np.array(model(a, np.array([[1, 4, 4]])))] * 2),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
