# SPDX-License-Identifier: Apache-2.0
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestPackedScatterGPU(unittest.TestCase):
    def test_unaligned_tail_and_multiple_pages(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import packed_scatter
        sizes = (13, 4096, 65539)
        outputs = [torch.zeros(size + 2, dtype=torch.uint8, device="cuda") for size in sizes]
        payload = torch.arange(sum(sizes), device="cuda").to(torch.uint8)
        descriptors = []
        offset = 0
        for out, size in zip(outputs, sizes):
            descriptors.append([out.data_ptr() + 1, offset, size])
            offset += size
        packed_scatter(payload, torch.tensor(descriptors, dtype=torch.int64, device="cuda"))
        torch.cuda.synchronize()
        offset = 0
        for out, size in zip(outputs, sizes):
            self.assertTrue(torch.equal(out[1:-1], payload[offset:offset + size]))
            self.assertEqual(int(out[0]), 0)
            self.assertEqual(int(out[-1]), 0)
            offset += size

    def test_aligned_vector_copy_and_descriptor_reuse(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import packed_scatter
        scratch = torch.empty(8192, dtype=torch.uint8, device="cuda")
        output = torch.zeros_like(scratch)
        descriptor = torch.tensor([[output.data_ptr(), 0, 8192]], dtype=torch.int64, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for value in range(20):
                scratch.fill_(value)
                packed_scatter(scratch, descriptor)
        stream.synchronize()
        self.assertTrue(torch.all(output == 19).item())
