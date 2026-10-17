"""Long draft extend must not allocate route scratch for the full prefill."""

import json
import unittest

import torch

from sglang.srt.layers.quantization.qwen38_fp4_mtp import fp4_mtp_moe


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMtpPrefillMemory(unittest.TestCase):
    def fixture(self, tokens):
        torch.manual_seed(20261017)
        e, h, n, k = 4, 2560, 640, 10
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = torch.randint(-1, e + 1, (tokens, k), device="cuda", dtype=torch.int32)
        router = torch.rand(tokens, k, device="cuda") / k
        w13 = torch.randint(256, (e, 2 * n, h // 2), device="cuda", dtype=torch.uint8)
        w2 = torch.randint(256, (e, h, n // 2), device="cuda", dtype=torch.uint8)
        s13 = torch.full((e, 2 * n, h // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
        s2 = torch.full((e, h, n // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
        g13 = torch.tensor([[0.03, 0.05]] * e, device="cuda")
        g2 = torch.full((e,), 0.03, device="cuda")
        return x, ids, router, w13, w2, s13, s2, g13, g2

    def test_full_8192_token_parity_and_peak(self):
        args = self.fixture(8192)
        # Same original kernels/reduction, one whole tile reproduces old scratch.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        reference = fp4_mtp_moe(*args, chunk_tokens=8192).cpu()
        original_peak = torch.cuda.max_memory_allocated() - before
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fp4_mtp_moe(*args)
        torch.cuda.synchronize()
        bounded_peak = torch.cuda.max_memory_allocated() - before
        torch.testing.assert_close(result.cpu(), reference, atol=0, rtol=0)
        self.assertLess(bounded_peak, 64 << 20)  # includes the 40-MiB output
        self.assertGreater(original_peak - bounded_peak, 800 << 20)
        print(json.dumps({"original_peak_bytes": original_peak,
                          "bounded_peak_bytes": bounded_peak,
                          "saved_peak_bytes": original_peak - bounded_peak}), flush=True)

    def test_partial_tile_and_graph_replay(self):
        args = self.fixture(129)
        reference = fp4_mtp_moe(*args, chunk_tokens=129)
        torch.testing.assert_close(fp4_mtp_moe(*args), reference, atol=0, rtol=0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fp4_mtp_moe(*args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fp4_mtp_moe(*args)
        args[0].mul_(0.75)
        args[1].copy_(args[1].roll(1, 1))
        graph.replay()
        torch.testing.assert_close(captured, fp4_mtp_moe(*args), atol=0, rtol=0)

    def test_adaptive_capacity_uses_same_splitk_kernels(self):
        from sglang.srt.layers.quantization import qwen38_cached_w4a16 as cached

        if not hasattr(cached, "_cache_project_splitk"):
            self.skipTest("Optional W4A16 split-K patch is not installed")
        from sglang.srt.layers.quantization.qwen38_cached_w4a16 import (
            _cache_project_splitk,
            cached_w4a16_moe,
        )
        from tools.qwen38_cache_tests.w4a16_verify_gpu_check import swizzle

        e, h, n = 8, 2560, 640
        x = torch.randn(3, h, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = torch.tensor([[0, 2], [3, -1], [1, 0]], device="cuda", dtype=torch.int32)
        router = torch.full((3, 2), 0.5, device="cuda")
        tensors = {}
        for prefix, rows, cols, alpha in (("w13", 2 * n, h, "g1_alphas"), ("w2", h, n, "g2_alphas")):
            tensors[prefix + "_weight"] = torch.randint(256, (e, rows, cols // 2), device="cuda", dtype=torch.uint8)
            scales = torch.full((e, rows, cols // 16), 0.5, dtype=torch.float8_e4m3fn)
            tensors[prefix + "_blockscale_swizzled"] = swizzle(scales)
            tensors[alpha] = torch.full((e,), 0.03, device="cuda")
            tensors[prefix + "_input_scale_quant"] = torch.ones((), device="cuda")
        count = None
        for capacity in (2, 3, 4, 8):
            views = {name: t[:capacity] if t.ndim else t for name, t in tensors.items()}
            actual = cached_w4a16_moe(x, ids, router, views, split_k=1)
            reference = cached_w4a16_moe(x, ids, router, views, split_k=0)
            torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.02)
            compiled = len(_cache_project_splitk.device_caches[torch.cuda.current_device()][0])
            if count is not None:
                self.assertEqual(compiled, count)
            count = compiled
        print(json.dumps({"capacities": [2, 3, 4, 8], "splitk_kernel_variants": count}), flush=True)


if __name__ == "__main__":
    unittest.main()
