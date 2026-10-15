"""Exercise GeForce registration without changing the checkpoint file."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestMailboxRegistration(unittest.TestCase):
    def test_private_mapping_and_cpu_controls_with_cuda_default(self):
        from .test_cpu_runtime import TestSynchronousRuntime
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        TestSynchronousRuntime.setUpClass()
        fixture = TestSynchronousRuntime()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        original = fixture.pack._mapping[:]
        with torch.device("cuda"):
            runtime = Qwen38ThreadedExpertRuntime(
                fixture.pack, "cuda", slots_per_layer=2, decode_device_gate=True
            )
            runtime.reserve()
        mapping = runtime.miss_mailbox.pack_mapping
        try:
            self.assertEqual(runtime.miss_mailbox.heads.device.type, "cpu")
            self.assertEqual(runtime.miss_mailbox.pack_view.device.type, "cpu")
            mapping[0] ^= 1
            self.assertNotEqual(mapping[:], original)
            self.assertEqual(fixture.pack._mapping[:], original)
            self.assertEqual(fixture.pack.manifest.payload_path.read_bytes(), original)
        finally:
            runtime.close()
        self.assertTrue(mapping.closed)
