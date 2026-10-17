"""Exercise GeForce registration without changing the checkpoint file."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestMailboxRegistration(unittest.TestCase):
    def test_single_source_and_cpu_controls_with_cuda_default(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        from .test_cpu_runtime import TestSynchronousRuntime

        TestSynchronousRuntime.setUpClass()
        fixture = TestSynchronousRuntime()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        # Anonymous sources are populated on first public read/registration.
        original = fixture.pack.manifest.payload_path.read_bytes()
        with torch.device("cuda"):
            runtime = Qwen38ThreadedExpertRuntime(
                fixture.pack, "cuda", slots_per_layer=2, decode_device_gate=True
            )
            runtime.reserve()
        mapping = runtime.miss_mailbox.pack_mapping
        try:
            self.assertIs(mapping, fixture.pack._mapping)
            self.assertIs(mapping, fixture.pack.cuda_source_mapping())
            self.assertEqual(runtime.miss_mailbox.failed.device.type, "cpu")
            self.assertEqual(runtime.miss_mailbox.pack_view.device.type, "cpu")
            view = fixture.pack.expert_tensor_bytes(0, 0, "w13_weight")
            try:
                self.assertTrue(view.readonly)
                with self.assertRaises(TypeError):
                    view[0] = 1
            finally:
                view.release()
            ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
            runtime.run(0, ids, lambda prepared: prepared.tensors["w13_weight"].clone())
            runtime.synchronize()
            self.assertEqual(fixture.pack._mapping[:], original)
            mapping[0] ^= 1
            self.assertEqual(fixture.pack._mapping[0], original[0] ^ 1)
            self.assertEqual(fixture.pack.manifest.payload_path.read_bytes(), original)
            mapping[0] = original[0]
        finally:
            runtime.close()
        # The reader still owns the only mapping after mailbox unregister.
        self.assertFalse(mapping.closed)
        self.assertEqual(mapping[:], original)
        fixture.pack.close()
        self.assertTrue(mapping.closed)
