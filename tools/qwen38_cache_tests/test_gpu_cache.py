import unittest

import torch


class TestGpuCache(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < 1 << 30:
            self.skipTest("less than 1 GiB free")
        torch.cuda.reset_peak_memory_stats()
        # Reuse the CPU fixture's disk exporter and pack builder, not its tests.
        from .test_cpu_runtime import TestSynchronousRuntime

        TestSynchronousRuntime.setUpClass()
        self.fixture = TestSynchronousRuntime()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def _exercise(self, mode, staging_buffer_bytes=None, warm_decode_cache=False, layer_cache_slots=None):
        from sglang.srt.layers.moe.qwen38_cache_runtime import (
            Qwen38SynchronousExpertRuntime,
        )

        runtime_class = Qwen38SynchronousExpertRuntime
        if mode == "async":
            from sglang.srt.layers.moe.qwen38_cache_async import (
                Qwen38AsyncExpertRuntime,
            )

            runtime_class = Qwen38AsyncExpertRuntime
        if mode == "device":
            from sglang.srt.layers.moe.qwen38_cache_device import (
                Qwen38DeviceExpertRuntime,
            )

            runtime_class = Qwen38DeviceExpertRuntime
        if mode == "threaded":
            from sglang.srt.layers.moe.qwen38_cache_threaded import (
                Qwen38ThreadedExpertRuntime,
            )

            runtime_class = Qwen38ThreadedExpertRuntime
        options = {}
        if staging_buffer_bytes is not None:
            options["staging_buffer_bytes"] = staging_buffer_bytes
        if warm_decode_cache:
            options["warm_decode_cache"] = True
        if layer_cache_slots is not None:
            options["layer_cache_slots"] = layer_cache_slots
        runtime = runtime_class(self.fixture.pack, "cuda", slots_per_layer=2, **options)
        self.addCleanup(
            runtime.close
            if hasattr(runtime, "close")
            else (
                runtime.synchronize
                if hasattr(runtime, "synchronize")
                else lambda: torch.cuda.synchronize()
            )
        )
        runtime.reserve()
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS
        pointers = [
            runtime.caches[0]["w13_weight"].data_ptr(),
            runtime.arena["w13_weight"].data_ptr(),
        ]
        outputs = []
        expected = []
        # Runs from independent compute streams to exercise last-use protection.
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for step in range(20):
            layer_id = step % 2
            ids_cpu = (
                torch.tensor([[0, 3]]) if step % 3 else torch.tensor([[0, 1], [2, 3]])
            )
            with torch.cuda.stream(streams[step % 2]):
                ids = ids_cpu.to("cuda", dtype=torch.int32)

                def execute(prepared):
                    return {
                        name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                        for name in REQUIRED_TENSORS
                    }

                outputs.append(runtime.run(layer_id, ids, execute))
            expected.append({
                name: getattr(self.fixture.resident[layer_id], name)[ids_cpu]
                for name in REQUIRED_TENSORS
            })
        if hasattr(runtime, "close"):
            runtime.close()
        elif hasattr(runtime, "synchronize"):
            runtime.synchronize()
        torch.cuda.synchronize()
        for actual, ref in zip(outputs, expected):
            for name in ref:
                self.assertTrue(torch.equal(actual[name].cpu(), ref[name]), name)
        self.assertEqual(runtime.caches[0]["w13_weight"].data_ptr(), pointers[0])
        self.assertEqual(runtime.arena["w13_weight"].data_ptr(), pointers[1])
        self.assertLess(torch.cuda.max_memory_allocated(), 512 << 20)

    def test_sync_transfer_cache_arena(self):
        self._exercise("sync")

    def test_async_ring_event_cache_arena(self):
        self._exercise("async")

    def test_threaded_pipeline(self):
        self._exercise("threaded")

    def test_threaded_bulk_split_inside_float_tensor(self):
        # Odd byte boundaries deliberately split float32 alpha tensors too.
        self._exercise("threaded", staging_buffer_bytes=11)

    def test_bulk_arena_copy_count_and_byte_parity(self):
        from sglang.srt.layers.moe.qwen38_cache_device import Qwen38DeviceExpertRuntime
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38DeviceExpertRuntime(
            self.fixture.pack, "cuda", slots_per_layer=2, staging_buffer_bytes=64
        )
        runtime.reserve()
        self.addCleanup(runtime.synchronize)
        copies_before = runtime.h2d_copy_calls
        batches_before = runtime.staging_batches
        runtime._fill_arena(0)
        runtime.synchronize()
        # Four 32-byte tensors and two 16-byte alpha tensors: 160 total bytes.
        self.assertEqual(runtime.h2d_copy_calls - copies_before, 6)
        self.assertEqual(runtime.staging_batches - batches_before, 3)
        for name in REQUIRED_TENSORS:
            self.assertTrue(torch.equal(
                runtime.arena[name].cpu(), getattr(self.fixture.resident[0], name)
            ), name)

    def test_device_policy_slot_pipeline(self):
        self._exercise("device")

    def test_threaded_warm_cache_cross_stream_parity(self):
        self._exercise("threaded", warm_decode_cache=True)

    def test_device_warm_cache_cross_stream_parity(self):
        self._exercise("device", warm_decode_cache=True)

    def test_threaded_heterogeneous_capacity_with_warmup(self):
        self._exercise("threaded", warm_decode_cache=True, layer_cache_slots=[2, 3])

    def test_device_heterogeneous_capacity(self):
        self._exercise("device", layer_cache_slots=[2, 3])

    def test_warm_seed_decode_hit_avoids_h2d(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack, "cuda", slots_per_layer=2, warm_decode_cache=True
        )
        runtime.reserve()
        self.addCleanup(runtime.close)

        def execute(prepared):
            return {name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                    for name in REQUIRED_TENSORS}

        prefill = torch.tensor([[1, 2], [1, 2], [0, 3]], dtype=torch.int32, device="cuda")
        runtime.run(0, prefill, execute)
        runtime.synchronize()
        transferred = runtime.transferred_bytes
        misses = runtime.misses
        for layer in (1, 0):
            # Replace the shared arena, then return to layer 0's warmed cache.
            if layer == 1:
                runtime.run(1, prefill, execute)
                runtime.synchronize()
                transferred = runtime.transferred_bytes
                continue
            output = runtime.run(0, prefill[:1], execute)
            runtime.synchronize()
            self.assertEqual(runtime.transferred_bytes, transferred)
            self.assertEqual(runtime.misses, misses)
            for name in REQUIRED_TENSORS:
                expected = getattr(self.fixture.resident[0], name)[torch.tensor([[1, 2]])]
                self.assertTrue(torch.equal(output[name].cpu(), expected), name)
        self.assertGreater(runtime.warmed_experts, 0)

    def test_jit_remap_invalid_padding(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_route

        ids = torch.tensor([[1, 3], [-1, 9]], dtype=torch.int32, device="cuda")
        lookup = torch.tensor([4, 2, -1, 0], dtype=torch.int32, device="cuda")
        output = torch.empty_like(ids)
        counts = torch.empty(5, dtype=torch.int32, device="cuda")
        cache_route(ids, lookup, output, counts)
        self.assertEqual(output.cpu().tolist(), [[2, 0], [-1, -1]])
        self.assertEqual(counts.cpu().tolist(), [0, 1, 0, 1, 1])

    def test_cuda_policy_matches_decay_lfu(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_plan, cache_route
        from sglang.srt.layers.moe.qwen38_expert_cache import LayerExpertCachePolicy

        experts, slots = 16, 4
        lookup = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
        keys = torch.full((slots,), -1, dtype=torch.int32, device="cuda")
        frequency = torch.zeros(slots, device="cuda")
        ages = torch.zeros(slots, dtype=torch.int64, device="cuda")
        generation = ages.clone()
        clock = torch.zeros(1, dtype=torch.int64, device="cuda")
        counts = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
        plan = torch.empty(4 + 3 * slots, dtype=torch.int32, device="cuda")
        reference = LayerExpertCachePolicy(experts, slots)
        for routes in [[0, 1], [0, 1], [2, 3], [4, 5], [1, 2], [7, 9], [7, 9]]:
            ids = torch.tensor([routes], dtype=torch.int32, device="cuda")
            out = torch.empty_like(ids)
            cache_route(ids, lookup, out, counts)
            cache_plan(counts, lookup, keys, frequency, ages, generation, clock, plan)
            mapping, misses = reference.ensure(sorted(set(routes)))
            values = plan.cpu().tolist()
            self.assertEqual(values[0], 0)
            self.assertEqual(values[1], len(misses))
            self.assertEqual(
                keys.cpu().tolist(),
                [
                    slot.expert_id if slot.expert_id is not None else -1
                    for slot in reference.slots
                ],
            )
            self.assertEqual(
                generation.cpu().tolist(), [slot.generation for slot in reference.slots]
            )
            cache_route(ids, lookup, out, counts)
            self.assertEqual(out.cpu().tolist(), [[mapping[e] for e in routes]])
