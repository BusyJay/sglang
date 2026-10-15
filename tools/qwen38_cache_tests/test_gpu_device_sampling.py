"""Decode sampling uses direct source fills and exports the original routes."""

import unittest
from unittest import mock

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestDeviceRouteSampling(unittest.TestCase):
    def test_direct_sampling_parity_routes_padding_and_cross_streams(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
            _sample_rows,
        )

        from .test_cpu_runtime import TestSynchronousRuntime

        TestSynchronousRuntime.setUpClass()
        fixture = TestSynchronousRuntime()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        runtime = Qwen38ThreadedExpertRuntime(
            fixture.pack,
            "cuda",
            slots_per_layer=2,
            max_pending=8,
            adaptive_cache=True,
            adaptive_cache_window=16,
            adaptive_cache_interval=0,
            adaptive_cache_mode="sample",
        )
        runtime.reserve()
        self.addCleanup(runtime.close)
        batches = []
        enqueue = runtime.adaptive_samples.put_nowait

        def observe(item):
            batches.append(item)
            enqueue(item)

        routes = [(0, 1), (0, 2), (3, 1), (2, 2), (-1, 0), (-1, -1)]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs, expected = [], {}
        with (
            mock.patch.object(
                runtime, "_host_plan", side_effect=AssertionError("sampling plan D2H")
            ),
            mock.patch.object(
                runtime.jobs,
                "put_nowait",
                side_effect=AssertionError("sampling staging job"),
            ),
            mock.patch.object(
                runtime.adaptive_samples, "put_nowait", side_effect=observe
            ),
        ):
            for step in range(40):
                pair = routes[step % len(routes)]
                for layer in range(2):
                    with torch.cuda.stream(streams[step % 2]):
                        ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                        result = runtime.run(
                            layer,
                            ids,
                            lambda prepared: prepared.tensors["w13_weight"][
                                prepared.topk_ids.clamp_min(0).long()
                            ].clone(),
                        )
                    outputs.append((layer, pair, result))
                    expected[layer, step // 16, step % 16] = sorted(set(pair) - {-1})
                if step % 16 == 15:
                    # Let the observer consume bank credits before testing
                    # another complete window on this deliberately tiny model.
                    runtime.synchronize()
                    runtime.adaptive_samples.join()
            runtime.synchronize()
        observed = {}
        for batch in batches:
            self.assertEqual(len(batch), 5)
            for layer, cycle, index, active in _sample_rows(batch):
                self.assertNotIn((layer, cycle, index), observed)
                observed[layer, cycle, index] = active
        self.assertEqual(observed, expected)
        self.assertEqual(runtime.decode_device_gate_invocations, 80)
        self.assertEqual(runtime.host_plan_allocations, 0)
        self.assertEqual(runtime.adaptive_sample_drops, 0)
        self.assertEqual(
            runtime.hits + runtime.misses, sum(map(len, expected.values()))
        )
        for layer, pair, actual in outputs:
            valid = torch.tensor([[expert >= 0 for expert in pair]])
            ids = torch.tensor([pair]).clamp_min(0)
            self.assertTrue(
                torch.equal(
                    actual.cpu()[valid], fixture.resident[layer].w13_weight[ids][valid]
                )
            )
