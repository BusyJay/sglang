# SPDX-License-Identifier: Apache-2.0
"""Borrowed expert slots must restore GPU addresses as well as source bytes."""

import queue
import struct
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.qwen38_borrowed_arena import BorrowedArenaMixin
from sglang.srt.layers.moe.qwen38_csf_banks import CsfBanks
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCsfArenaRestore(CustomTestCase):
    def test_threaded_fill_binds_headers_before_hit(self):
        """A short prefill miss must publish bound headers to later cache hits."""
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
            TransferJob,
        )

        name = "w13_blockscale_swizzled"
        source = torch.zeros((3, 1040), dtype=torch.uint8, pin_memory=True)
        source[1, 976:984] = torch.tensor(list(struct.pack("<Q", 4096)))
        source[1:, 1024:] = torch.arange(16, dtype=torch.uint8)
        pack = SimpleNamespace(
            manifest=SimpleNamespace(
                scale_encoding={
                    name: {
                        "encoding": "b12x_csf_tiered_v1",
                        "overflow": {"0": {"offset": 4096, "nbytes": 16}},
                    }
                }
            ),
            cuda_source_mapping=lambda: bytearray(4096) + bytearray(range(16)),
        )
        banks = CsfBanks(pack, torch.device("cuda"))

        class Runtime(Qwen38ThreadedExpertRuntime):
            # Substitute only source transport; run the real worker, readiness
            # publication, slot-reader tracking and bank binding.
            def _fill(self, layer, expert, tensors, slot, names):
                with torch.cuda.stream(self.transfer):
                    tensors[name][slot].copy_(source[expert], non_blocking=True)

        runtime = Runtime.__new__(Runtime)
        runtime.device = torch.device("cuda")
        runtime.manifest = SimpleNamespace(num_experts=3)
        runtime.layer_cache_slots = [3]
        runtime.jobs, runtime.retired = queue.Queue(), queue.Queue()
        runtime.transfer = torch.cuda.Stream()
        runtime.worker_error = None
        runtime.caches = {0: {name: torch.zeros_like(source, device="cuda")}}
        runtime.csf_banks = banks
        runtime.decode_copy_names = (name,)
        runtime.slot_last_use = {0: [None] * 3}
        runtime.resident_alphas = runtime.adaptive_cache = False
        runtime.misses = runtime.hits = 0
        runtime.layer_misses, runtime.layer_hits = {0: 0}, {0: 0}
        worker = threading.Thread(target=runtime._work, daemon=True)
        worker.start()
        snapshots = []
        try:
            for misses in (2, 0):
                planned, computed = torch.cuda.Event(), torch.cuda.Event()
                planned.record()
                job = TransferJob(
                    0,
                    False,
                    None,
                    planned,
                    computed,
                    threading.Event(),
                    plan_values=[0, misses, 2, 0, 1, 1, 2, 2, 0, 0, 1, 2, 0],
                )
                runtime.jobs.put(job)
                self.assertTrue(job.ready_set.wait(30))
                self.assertIsNone(job.error)
                if job.ready is not None:
                    torch.cuda.current_stream().wait_event(job.ready)
                snapshots.append(runtime.caches[0][name].clone())
                computed.record()
                job.launched.set()
        finally:
            runtime.jobs.put(None)
            worker.join(30)
        self.assertFalse(worker.is_alive())
        expected = source.clone()
        pointer = banks.banks[0, name][0].data_ptr()
        expected[1, 992:1000] = torch.tensor(list(struct.pack("<Q", pointer)))
        for snapshot in snapshots:
            torch.testing.assert_close(snapshot.cpu(), expected, rtol=0, atol=0)
        self.assertEqual(runtime.misses, 2)
        self.assertEqual(runtime.hits, 2)
        self.assertEqual(source[:, 992:1000].count_nonzero().item(), 0)

    def test_restored_hits_rebind_external_bank(self):
        from sglang.srt.model_loader.qwen38_nvfp4_pack.tiered import TIER_MAGIC

        name = "w13_blockscale_swizzled"
        source = torch.zeros((3, 1040), dtype=torch.uint8, pin_memory=True)
        # Slot 1 is an outlier, slot 2 has inline replacement records.
        for slot in (1, 2):
            header = bytearray(1040)
            struct.pack_into("<Q", header, 976, 4096 if slot == 1 else 0)
            struct.pack_into("<I", header, 1000, 16 if slot == 1 else 0)
            struct.pack_into("<2I", header, 1016, TIER_MAGIC, 1040)
            header[1024:] = bytes(range(16))
            source[slot].copy_(torch.tensor(list(header), dtype=torch.uint8))
        bank_bytes = bytearray(4096) + bytearray(range(16))
        pack = SimpleNamespace(
            manifest=SimpleNamespace(
                scale_encoding={
                    name: {
                        "encoding": "b12x_csf_tiered_v1",
                        "overflow": {"0": {"offset": 4096, "nbytes": 16}},
                    }
                }
            ),
            cuda_source_mapping=lambda: bank_bytes,
        )
        banks = CsfBanks(pack, torch.device("cuda"))

        class Runtime(BorrowedArenaMixin):
            # Only the source transport is substituted; lease snapshot, drain,
            # return, header binding and stream ordering are production logic.
            def _fill(self, layer, expert, tensors, slot):
                with torch.cuda.stream(self.transfer):
                    tensors[name][slot].copy_(source[expert], non_blocking=True)

        runtime = Runtime()
        runtime.borrowed_arena = True
        runtime.borrowed_records = None
        runtime.layer_cache_slots = [3]
        runtime.manifest = SimpleNamespace(num_experts=3)
        runtime.device_states = {0: {"keys": torch.tensor([-1, 1, 2], device="cuda")}}
        runtime.caches = {0: {name: source.cuda()}}
        runtime.csf_banks = banks
        runtime.jobs = queue.Queue()
        runtime.miss_mailbox = None
        runtime.transfer = torch.cuda.Stream()
        runtime.metadata_stream = torch.cuda.Stream()
        runtime.layer_computed = {}
        runtime.arena_last_use = None
        runtime.borrowed_restore_experts = runtime.borrowed_restore_runs = 0
        runtime.poisoned = False
        banks.patch(0, runtime.caches[0])
        torch.cuda.synchronize()
        runtime._borrow_arena()
        # Prefill overwrites these storage slots without changing the hit map.
        runtime.caches[0][name].fill_(255)
        torch.cuda.synchronize()
        runtime._return_arena()
        actual = runtime.caches[0][name].cpu()
        expected = source.clone()
        pointer = banks.banks[0, name][0].data_ptr()
        expected[1, 992:1000] = torch.tensor(
            list(struct.pack("<Q", pointer)), dtype=torch.uint8
        )
        torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
        self.assertEqual(runtime.device_states[0]["keys"].tolist(), [-1, 1, 2])
        self.assertEqual(runtime.borrowed_restore_experts, 2)
        self.assertEqual(runtime.borrowed_restore_runs, 1)
        self.assertIsNone(runtime.borrowed_records)
        self.assertEqual(source[:, 992:1000].count_nonzero().item(), 0)
        torch.testing.assert_close(
            banks.banks[0, name][0].cpu(), torch.arange(16, dtype=torch.uint8)
        )


if __name__ == "__main__":
    unittest.main()
