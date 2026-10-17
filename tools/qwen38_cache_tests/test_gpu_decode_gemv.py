"""Numerical and graph coverage for the measured NVFP4 BF16 projections."""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestDecodeGemv(unittest.TestCase):
    @mock.patch.dict(os.environ, {"SGLANG_QWEN38_DECODE_GEMV": "1"})
    def test_measured_shapes_and_graph_replay(self):
        from sglang.srt.layers.quantization.qwen38_decode_gemv import (
            applicable,
            decode_gemv,
        )

        torch.manual_seed(43)
        for n, k in (
            (16480, 2560),
            (2560, 6144),
            (13312, 2560),
            (2560, 2560),
            (1280, 2560),
            (2560, 640),
        ):
            with self.subTest(n=n, k=k):
                x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(applicable(x, weight, None))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    decode_gemv(x, weight)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = decode_gemv(x, weight)
                for _ in range(3):
                    x.normal_()
                    graph.replay()
                    reference = F.linear(x.float(), weight.float())
                    torch.testing.assert_close(
                        actual.float(), reference, rtol=0.02, atol=0.05 * k**0.5
                    )
                    relative_rmse = (
                        (actual.float() - reference).square().mean()
                        / reference.square().mean()
                    ).sqrt()
                    self.assertLess(float(relative_rmse), 0.003)

    @mock.patch.dict(os.environ, {"SGLANG_QWEN38_DECODE_GEMV": "1"})
    def test_unmeasured_shapes_and_disable_switch_fall_back(self):
        from sglang.srt.layers.quantization.qwen38_decode_gemv import applicable

        x = torch.zeros(1, 2560, device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros(1280, 2560, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(applicable(x, weight, None))
        self.assertFalse(applicable(x, weight[:512], None))
        self.assertFalse(applicable(x.expand(2, -1), weight, None))
        self.assertFalse(applicable(x, weight, torch.zeros(1280, device="cuda")))
        with mock.patch.dict(os.environ, {"SGLANG_QWEN38_DECODE_GEMV": "0"}):
            self.assertFalse(applicable(x, weight, None))
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_QWEN38_DECODE_GEMV", None)
            self.assertFalse(applicable(x, weight, None))
