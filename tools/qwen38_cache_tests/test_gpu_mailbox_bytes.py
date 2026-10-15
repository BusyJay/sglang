"""Validate transport bytes, including multi-tile tails, without a checkpoint."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestMailboxBytes(unittest.TestCase):
    def test_direct_sources_and_tail_bytes(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import _module
        from cuda.bindings import driver
        from sglang.srt.utils.cuda_vmm_utils import check_drv

        def address(tensor):
            return int(
                check_drv(
                    driver.cuMemHostGetDevicePointer(tensor.data_ptr(), 0),
                    "test mapped pointer",
                )
            )

        for sizes in ((65536, 65552, 147472), (65537, 19, 8193)):
            sources = [
                torch.arange(4 * size, dtype=torch.int64).to(torch.uint8).pin_memory()
                for size in sizes
            ]
            targets = [
                torch.full((4, size), 255, dtype=torch.uint8, device="cuda")
                for size in sizes
            ]
            layout = torch.tensor(
                [
                    [dst.data_ptr(), size, address(src)]
                    for src, dst, size in zip(sources, targets, sizes)
                ],
                dtype=torch.int64,
                device="cuda",
            )
            failed = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            ready = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            epoch, top_k = 7, 2
            for pairs in (((3, 0),), ((1, 2), (3, 0))):
                cpu_plan = torch.zeros(4 + 3 * 4, dtype=torch.int32, pin_memory=True)
                cpu_plan[:4] = torch.tensor([0, len(pairs), len(pairs), 0])
                for m, (expert, slot) in enumerate(pairs):
                    cpu_plan[4 + 2 * m : 6 + 2 * m] = torch.tensor([expert, slot])
                plan = cpu_plan.to("cuda")
                ready[0] = epoch * (top_k + 1) + len(pairs)
                _module().mailbox_fill(
                    plan,
                    layout,
                    address(sources[0]),
                    sum(sizes),
                    top_k,
                    address(failed),
                    address(ready),
                    epoch,
                    1000000000,
                )
                torch.cuda.synchronize()
                for expert, slot in pairs:
                    for src, dst, size in zip(sources, targets, sizes):
                        self.assertTrue(
                            torch.equal(
                                dst[slot].cpu(),
                                src[expert * size : (expert + 1) * size],
                            ),
                            (size, expert, slot),
                        )
