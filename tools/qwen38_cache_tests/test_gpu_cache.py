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

    def _exercise(
        self,
        mode,
        staging_buffer_bytes=None,
        warm_decode_cache=False,
        layer_cache_slots=None,
        decode_device_gate=None,
    ):
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
        if decode_device_gate is not None:
            options["decode_device_gate"] = decode_device_gate
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
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
            expected.append(
                {
                    name: getattr(self.fixture.resident[layer_id], name)[ids_cpu]
                    for name in REQUIRED_TENSORS
                }
            )
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
        self.assertLess(torch.cuda.max_memory_allocated() - allocated_before, 512 << 20)

    def test_mailbox_transport_all_tensors_single_multi_duplicate_and_streams(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        for resident_alphas in (False, True):
            runtime = Qwen38ThreadedExpertRuntime(
                self.fixture.pack,
                "cuda",
                slots_per_layer=2,
                decode_device_gate=True,
                resident_alphas=resident_alphas,
            )
            runtime.reserve()
            try:
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                # Warm path, one miss, two misses, repeated ID, all hit.
                routes = [(0, 1), (0, 2), (1, 3), (2, 2), (2, 2), (0, 1)]
                results = []
                for step in range(48):
                    pair = routes[step % len(routes)]
                    layer = (step // len(routes)) % 2
                    with torch.cuda.stream(streams[step % 2]):
                        ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                        actual = runtime.run(
                            layer,
                            ids,
                            lambda prepared: {
                                name: prepared.tensors[name][
                                    prepared.topk_ids.long()
                                ].clone()
                                for name in REQUIRED_TENSORS
                            },
                        )
                    results.append((layer, pair, actual))
                runtime.synchronize()
                for layer, pair, actual in results:
                    for name, value in actual.items():
                        reference = getattr(self.fixture.resident[layer], name)[
                            torch.tensor([pair])
                        ]
                        self.assertTrue(
                            torch.equal(value.cpu(), reference), (layer, pair, name)
                        )
                self.assertFalse(hasattr(runtime.miss_mailbox, "payload"))
            finally:
                runtime.close()
        # Closing the runtime must release the frombuffer view so pack.close works.

    def test_sync_transfer_cache_arena(self):
        self._exercise("sync")

    def test_async_ring_event_cache_arena(self):
        self._exercise("async")

    def test_threaded_pipeline(self):
        self._exercise("threaded")

    def test_device_gate_pipeline_preserves_cross_stream_and_prefill_readers(self):
        self._exercise("threaded", decode_device_gate=True)

    def test_device_gate_warm_cache_and_heterogeneous_capacity(self):
        self._exercise(
            "threaded",
            decode_device_gate=True,
            warm_decode_cache=True,
            layer_cache_slots=[2, 3],
        )

    def test_device_gate_rewarms_after_adaptive_migration(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            adaptive_cache=True,
            adaptive_cache_mode="sample",
            decode_device_gate=True,
        )
        runtime.reserve()
        self.addCleanup(runtime.close)
        # Exercise the native path outside adaptive observation windows.
        runtime.adaptive_steps[0] = runtime.adaptive_cache_window

        def execute(prepared):
            return prepared.tensors["w13_weight"][prepared.topk_ids.long()].clone()

        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        runtime.run(0, ids, execute)
        runtime.run(0, ids, execute)
        runtime.synchronize()
        self.assertEqual(runtime.decode_device_gate_invocations, 1)
        runtime.adaptive_target = [4, 2]
        runtime._maybe_rebalance()
        self.assertFalse(runtime.gate_warmed_layers)
        runtime.run(0, ids, execute)
        self.assertEqual(runtime.decode_device_gate_invocations, 1)
        output = runtime.run(0, ids, execute)
        runtime.synchronize()
        self.assertEqual(runtime.decode_device_gate_invocations, 2)
        self.assertTrue(
            torch.equal(
                output.cpu(),
                self.fixture.resident[0].w13_weight[torch.tensor([[0, 1]])],
            )
        )

    def test_device_gate_timeout_fails_before_stale_weights_in_subprocess(self):
        import os
        import subprocess
        import sys
        import textwrap

        # The failure check deliberately invalidates the child's CUDA context.
        # Keep the parent context usable for the remaining parity tests.
        source = textwrap.dedent("""
            import os, time, torch
            from tools.qwen38_cache_tests.test_cpu_runtime import TestSynchronousRuntime
            from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
            TestSynchronousRuntime.setUpClass()
            fixture = TestSynchronousRuntime()
            fixture.setUp()
            runtime = Qwen38ThreadedExpertRuntime(
                fixture.pack, 'cuda', slots_per_layer=2, decode_device_gate=True
            )
            runtime.reserve()
            routes = [torch.tensor([pair], dtype=torch.int32, device='cuda')
                      for pair in [(0, 1), (2, 3)]]
            runtime.run(0, routes[0], lambda prepared: None)
            runtime.synchronize()
            stage = runtime.miss_mailbox.stage
            def delayed_stage(*args, **kwargs):
                time.sleep(0.3)
                return stage(*args, **kwargs)
            runtime.miss_mailbox.stage = delayed_stage
            os.environ['QWEN38_READY_TIMEOUT_SECONDS'] = '0.05'
            try:
                runtime.run(0, routes[1], lambda prepared: None)
                torch.cuda.synchronize()
            except torch.AcceleratorError:
                os._exit(0 if runtime.poisoned else 2)
            os._exit(3)
        """)
        child = subprocess.run(
            [sys.executable, "-c", source],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_device_gate_all_hit_skips_d2h_jobs_and_worker(self):
        from unittest import mock

        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack, "cuda", slots_per_layer=2
        )
        self.assertTrue(runtime.decode_device_gate)
        runtime.reserve()
        self.addCleanup(runtime.close)
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

        def execute(prepared):
            return prepared.tensors["w13_weight"][prepared.topk_ids.long()].clone()

        runtime.run(0, ids, execute)
        runtime.synchronize()
        heads = runtime.miss_mailbox.heads.clone()
        host_ready = runtime.miss_mailbox.ready.clone()
        plans = runtime.host_plan_allocations
        with (
            mock.patch.object(
                runtime, "_host_plan", side_effect=AssertionError("all-hit D2H")
            ),
            mock.patch.object(
                runtime.jobs, "put_nowait", side_effect=AssertionError("all-hit job")
            ),
            mock.patch.object(
                runtime, "_fill", side_effect=AssertionError("all-hit fill")
            ),
        ):
            for _ in range(32):
                output = runtime.run(0, ids, execute)
            runtime.synchronize()
        self.assertTrue(
            torch.equal(
                output.cpu(),
                self.fixture.resident[0].w13_weight[torch.tensor([[0, 1]])],
            )
        )
        self.assertTrue(torch.equal(runtime.miss_mailbox.heads, heads))
        self.assertTrue(torch.equal(runtime.miss_mailbox.ready, host_ready))
        self.assertEqual(runtime.host_plan_allocations, plans)
        self.assertEqual(runtime.jobs.unfinished_tasks, 0)
        self.assertEqual(runtime.decode_device_gate_hits, 32)
        self.assertEqual(runtime.hits, 64)

    def test_miss_mailbox_sampling_and_slru_preserve_queued_readers(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        for policy in ("lfu", "slru"):
            runtime = Qwen38ThreadedExpertRuntime(
                self.fixture.pack,
                "cuda",
                slots_per_layer=2,
                max_pending=8,
                decode_device_gate=True,
                adaptive_cache=True,
                adaptive_cache_mode="sample",
                adaptive_cache_window=16,
                adaptive_cache_interval=16,
                cache_policy=policy,
            )
            runtime.reserve()
            try:
                routes = [
                    torch.tensor([pair], dtype=torch.int32, device="cuda")
                    for pair in ((0, 1), (2, 3))
                ]
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                outputs = []
                for i in range(128):
                    route = (i // 2) % 2
                    with torch.cuda.stream(streams[i % 2]):
                        output = runtime.run(
                            0,
                            routes[route],
                            lambda prepared: prepared.tensors["w13_weight"][
                                prepared.topk_ids.long()
                            ].clone(),
                        )
                    outputs.append((route, output))
                runtime.synchronize()
                for route, output in outputs:
                    self.assertTrue(
                        torch.equal(
                            output.cpu(),
                            self.fixture.resident[0].w13_weight[
                                routes[route].cpu().long()
                            ],
                        )
                    )
                self.assertEqual(runtime.hits + runtime.misses, 256)
                self.assertEqual(runtime.decode_device_gate_invocations, 64)
                self.assertGreater(runtime.decode_device_gate_hits, 0)
                self.assertEqual(runtime.jobs.unfinished_tasks, 0)
            finally:
                runtime.close()

    def test_all_hit_decode_skips_transport_and_preserves_cross_stream_readers(self):
        from unittest import mock

        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        for enabled in (False, True):
            runtime = Qwen38ThreadedExpertRuntime(
                self.fixture.pack,
                "cuda",
                slots_per_layer=2,
                decode_hit_fast_path=enabled,
                decode_device_gate=False,
                max_pending=1,
            )
            runtime.reserve()
            try:

                def execute(prepared):
                    # Keep a reader in flight when the next stream evicts it.
                    torch.cuda._sleep(100000)
                    return {
                        name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                        for name in REQUIRED_TENSORS
                    }

                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                outputs = []
                with mock.patch.object(
                    runtime.jobs, "put_nowait", wraps=runtime.jobs.put_nowait
                ) as submit:
                    for index, pair in enumerate(
                        ((0, 1), (0, 1), (2, 3), (2, 3), (0, 1))
                    ):
                        with torch.cuda.stream(streams[index % 2]):
                            ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                            outputs.append((pair, runtime.run(0, ids, execute)))
                    runtime.synchronize()
                    torch.cuda.synchronize()
                    self.assertEqual(submit.call_count, 3 if enabled else 5)
                for pair, output in outputs:
                    for name in REQUIRED_TENSORS:
                        expected = getattr(self.fixture.resident[0], name)[
                            torch.tensor([pair])
                        ]
                        self.assertTrue(torch.equal(output[name].cpu(), expected), name)
                self.assertEqual(runtime.hits, 4)
                self.assertEqual(runtime.misses, 6)
                self.assertEqual(
                    runtime.decode_hit_fast_path_invocations, 2 if enabled else 0
                )
                self.assertEqual(runtime.jobs.unfinished_tasks, 0)
            finally:
                runtime.close()

    def test_all_hit_execute_failure_releases_plan_and_queue_capacity(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=2,
            max_pending=1,
            decode_device_gate=False,
        )
        runtime.reserve()
        self.addCleanup(runtime.close)
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        runtime.run(0, ids, lambda prepared: None)
        runtime.synchronize()

        def fail(prepared):
            raise RuntimeError("test execute failure")

        with self.assertRaisesRegex(RuntimeError, "test execute failure"):
            runtime.run(0, ids, fail)
        self.assertTrue(runtime.capacity.acquire(blocking=False))
        runtime.capacity.release()
        self.assertEqual(runtime.jobs.unfinished_tasks, 0)
        self.assertTrue(all(pool.qsize() > 0 for pool in runtime.host_plans.values()))
        runtime.run(0, ids, lambda prepared: None)
        self.assertEqual(runtime.decode_hit_fast_path_invocations, 2)

    def test_fused_decode_plan_matches_reference_state_and_plan(self):
        import random

        from sglang.kernels.ops.moe.qwen38_expert_cache import (
            cache_decode_plan,
            cache_plan,
            cache_route,
        )

        experts, slots, top_k = 512, 16, 10
        for policy in (0, 1):
            for adaptive in (False, True):
                with self.subTest(policy=policy, adaptive=adaptive):

                    def fresh():
                        return {
                            "lookup": torch.full(
                                (experts,), -1, dtype=torch.int32, device="cuda"
                            ),
                            "keys": torch.full(
                                (slots,), -1, dtype=torch.int32, device="cuda"
                            ),
                            "frequency": torch.zeros(slots, device="cuda"),
                            "ages": torch.zeros(
                                slots, dtype=torch.int64, device="cuda"
                            ),
                            "generation": torch.zeros(
                                slots, dtype=torch.int64, device="cuda"
                            ),
                            "clock": torch.zeros(1, dtype=torch.int64, device="cuda"),
                            "counts": torch.zeros(
                                experts + 1, dtype=torch.int32, device="cuda"
                            ),
                            "plan": torch.full(
                                (4 + (4 if adaptive else 3) * slots,),
                                -77,
                                dtype=torch.int32,
                                device="cuda",
                            ),
                        }

                    reference, fused = fresh(), fresh()
                    rng = random.Random(17)
                    rows = [
                        list(range(10)),
                        list(reversed(range(10))),
                        [3] * top_k,
                        [-1] * top_k,
                        [511, 0] * 5,
                    ]
                    rows += [rng.sample(range(48), top_k) for _ in range(80)]
                    rows += [[-2] + [1] * 9, [512] + [-1] * 9]
                    for row in rows:
                        ids = torch.tensor([row], dtype=torch.int32, device="cuda")
                        ref_out, fused_out = (
                            torch.empty_like(ids),
                            torch.empty_like(ids),
                        )
                        cache_route(
                            ids, reference["lookup"], ref_out, reference["counts"]
                        )
                        cache_plan(
                            *(
                                reference[name]
                                for name in (
                                    "counts",
                                    "lookup",
                                    "keys",
                                    "frequency",
                                    "ages",
                                    "generation",
                                    "clock",
                                    "plan",
                                )
                            ),
                            policy=policy,
                        )
                        cache_route(
                            ids, reference["lookup"], ref_out, reference["counts"]
                        )
                        cache_decode_plan(
                            ids,
                            fused_out,
                            *(
                                fused[name]
                                for name in (
                                    "counts",
                                    "lookup",
                                    "keys",
                                    "frequency",
                                    "ages",
                                    "generation",
                                    "clock",
                                    "plan",
                                )
                            ),
                            policy=policy,
                        )
                        for name in reference:
                            self.assertTrue(
                                torch.equal(reference[name], fused[name]),
                                (policy, adaptive, row, name),
                            )
                        self.assertTrue(torch.equal(ref_out, fused_out), row)

    def test_fused_decode_overflow_preserves_policy(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_decode_plan
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack, "cuda", slots_per_layer=2
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        state = runtime.device_states[0]
        before = {
            name: state[name].clone()
            for name in ("lookup", "keys", "frequency", "ages", "generation", "clock")
        }
        ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
        remapped = torch.empty_like(ids)
        cache_decode_plan(
            ids,
            remapped,
            *(
                state[name]
                for name in (
                    "counts",
                    "lookup",
                    "keys",
                    "frequency",
                    "ages",
                    "generation",
                    "clock",
                    "plan",
                )
            ),
        )
        self.assertEqual(state["plan"][:4].tolist(), [0, 0, 0, 1])
        for name in before:
            self.assertTrue(torch.equal(before[name], state[name]), name)

    def test_decode_transport_options_preserve_all_tensors_and_migration(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        for alphas in (False, True):
            with self.subTest(alphas=alphas):
                runtime = Qwen38ThreadedExpertRuntime(
                    self.fixture.pack,
                    "cuda",
                    slots_per_layer=2,
                    layer_cache_slots=[2, 4],
                    adaptive_cache=True,
                    adaptive_cache_mode="sample",
                    prefill_seed_cache=False,
                    staging_buffer_bytes=11,
                    resident_alphas=alphas,
                )
                self.addCleanup(runtime.close)
                runtime.reserve()
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                calls_before = runtime.h2d_copy_calls
                misses_before = runtime.misses
                bytes_before = runtime.transferred_bytes
                row_bytes = sum(
                    self.fixture.pack.manifest.tensors[0][name].nbytes // 4
                    for name in REQUIRED_TENSORS
                )
                # Two misses, all hits, eviction, duplicate and padding.
                for step, pair in enumerate(((0, 1), (0, 1), (2, 3), (3, 3), (-1, 3))):
                    with torch.cuda.stream(streams[step % 2]):
                        ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                        output = runtime.run(
                            0,
                            ids,
                            lambda prepared: {
                                name: prepared.tensors[name][
                                    prepared.topk_ids.long()
                                ].clone()
                                for name in REQUIRED_TENSORS
                            },
                        )
                    runtime.synchronize()
                    valid = torch.tensor(pair) >= 0
                    for name in REQUIRED_TENSORS:
                        expected = getattr(self.fixture.resident[0], name)[
                            torch.tensor(pair)[valid]
                        ]
                        self.assertTrue(
                            torch.equal(output[name].cpu()[0, valid], expected), name
                        )
                self.assertEqual(runtime.misses - misses_before, 4)
                self.assertEqual(
                    runtime.transferred_bytes - bytes_before,
                    4 * (row_bytes - (8 if alphas else 0)),
                )
                self.assertEqual(
                    runtime.decode_alpha_h2d_calls_avoided, 8 if alphas else 0
                )
                if alphas:
                    self.assertEqual(
                        runtime.memory_budget()["resident_alpha_bytes"], 64
                    )
                self.assertGreater(runtime.h2d_copy_calls, calls_before)
                runtime.adaptive_target = [4, 2]
                runtime._maybe_rebalance()
                for pair in ((2, 3), (0, 1)):
                    ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                    output = runtime.run(
                        0,
                        ids,
                        lambda prepared: {
                            name: prepared.tensors[name][
                                prepared.topk_ids.long()
                            ].clone()
                            for name in REQUIRED_TENSORS
                        },
                    )
                    runtime.synchronize()
                    for name in REQUIRED_TENSORS:
                        expected = getattr(self.fixture.resident[0], name)[ids.cpu()]
                        self.assertTrue(torch.equal(output[name].cpu(), expected), name)
                runtime.close()

    def test_alpha_scatter_updates_only_misses_and_rejects_invalid_plan(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_alpha_scatter

        for plan_size in (10, 12):  # ordinary and adaptive plan layouts
            g1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
            g2 = g1 + 10
            slot_g1 = torch.full((2,), -9.0, device="cuda")
            slot_g2 = torch.full((2,), -8.0, device="cuda")
            plan = torch.zeros(plan_size, dtype=torch.int32, device="cuda")
            plan[:6] = torch.tensor([0, 1, 1, 0, 3, 1], device="cuda")
            cache_alpha_scatter(g1, g2, slot_g1, slot_g2, plan)
            self.assertEqual(slot_g1.tolist(), [-9.0, 4.0])
            self.assertEqual(slot_g2.tolist(), [-8.0, 14.0])
            for status, arena in ((1, 0), (0, 1)):
                plan[0], plan[3] = status, arena
                plan[4] = 0
                cache_alpha_scatter(g1, g2, slot_g1, slot_g2, plan)
                self.assertEqual(slot_g1.tolist(), [-9.0, 4.0])
                self.assertEqual(slot_g2.tolist(), [-8.0, 14.0])

    def test_slru_planner_matches_cpu_on_evictions_and_exports_routes(self):
        import random

        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_plan, cache_route
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.layers.moe.qwen38_expert_cache import LayerExpertCachePolicy

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            cache_policy="slru",
            cache_protected_fraction=0.5,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        state = runtime.device_states[0]
        plan = torch.empty(4 + 4 * 3, dtype=torch.int32, device="cuda")
        reference = LayerExpertCachePolicy(4, 3, policy="slru", protected_fraction=0.5)
        rng = random.Random(42)
        for _ in range(64):
            active = sorted(rng.sample(range(4), 2))
            ids = torch.tensor([active], dtype=torch.int32, device="cuda")
            remapped = torch.empty_like(ids)
            cache_route(ids, state["lookup"], remapped, state["counts"])
            cache_plan(
                state["counts"],
                state["lookup"],
                state["keys"],
                state["frequency"],
                state["ages"],
                state["generation"],
                state["clock"],
                plan,
                policy=1,
                protected_fraction=0.5,
            )
            mapping, misses = reference.ensure(active)
            values = plan.cpu().tolist()
            self.assertEqual(values[:4], [0, len(misses), 2, 0])
            self.assertEqual(values[13:15], active)
            self.assertEqual(
                state["keys"].cpu().tolist(),
                [
                    -1 if slot.expert_id is None else slot.expert_id
                    for slot in reference.slots
                ],
            )
            self.assertEqual(
                state["frequency"].cpu().tolist(),
                [slot.frequency for slot in reference.slots],
            )

    def test_slru_threaded_weights_across_scan_and_rebalance(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            cache_policy="slru",
            adaptive_cache=True,
            adaptive_cache_mode="sample",
            adaptive_cache_window=16,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        for step in range(40):
            if step == 20:
                runtime.adaptive_target = [2, 4]
                runtime._maybe_rebalance()
            for layer in (0, 1):
                active = ((0, 1), (1, 2), (2, 3))[step % 3]
                ids = torch.tensor([active], dtype=torch.int32, device="cuda")
                output = runtime.run(
                    layer,
                    ids,
                    lambda prepared: {
                        name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                        for name in REQUIRED_TENSORS
                    },
                )
                for name, tensor in output.items():
                    expected = getattr(self.fixture.resident[layer], name)[ids.cpu()]
                    self.assertTrue(torch.equal(tensor.cpu(), expected), name)
        self.assertLess(runtime.host_plan_allocations, 20)

    def test_threaded_sparse_prefill_arena_loads_only_routed_experts(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=2,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        runtime.prefill_tokens_seen = 128
        one_expert_bytes = sum(
            self.fixture.pack.manifest.tensors[0][name].nbytes // 4
            for name in REQUIRED_TENSORS
        )
        for pair, expected_loaded in (((0, 1), 2), ((0, 1), 0), ((2, 3), 2)):
            ids = torch.tensor([pair, pair], dtype=torch.int32, device="cuda")
            before = runtime.transferred_bytes
            result = runtime.run(
                0,
                ids,
                lambda prepared: prepared.tensors["w13_weight"][
                    prepared.topk_ids.long()
                ].clone(),
            )
            runtime.synchronize()
            expected = self.fixture.resident[0].w13_weight[ids.cpu()]
            self.assertTrue(torch.equal(result.cpu(), expected))
            self.assertEqual(
                runtime.transferred_bytes - before,
                expected_loaded * one_expert_bytes,
            )

    def test_prefill_reuses_cache_across_layers_and_seeds_only_empty_layers(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        for adaptive in (False, True):
            runtime = Qwen38ThreadedExpertRuntime(
                self.fixture.pack,
                "cuda",
                slots_per_layer=2,
                adaptive_cache=adaptive,
                adaptive_cache_mode="sample",
                cache_policy="slru",
            )
            self.addCleanup(runtime.close)
            runtime.reserve()
            runtime.prefill_tokens_seen = 128
            one_expert = sum(
                self.fixture.pack.manifest.tensors[0][name].nbytes // 4
                for name in REQUIRED_TENSORS
            )
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            for step in range(3):
                for layer in (0, 1):
                    with torch.cuda.stream(streams[step % 2]):
                        ids = torch.tensor(
                            [[0, 1], [2, 3]], dtype=torch.int32, device="cuda"
                        )
                        before = runtime.transferred_bytes
                        output = runtime.run(
                            layer,
                            ids,
                            lambda prepared: {
                                name: prepared.tensors[name][
                                    prepared.topk_ids.long()
                                ].clone()
                                for name in REQUIRED_TENSORS
                            },
                        )
                    runtime.synchronize()
                    self.assertEqual(
                        runtime.transferred_bytes - before,
                        (4 if step == 0 else 2) * one_expert,
                    )
                    for name in REQUIRED_TENSORS:
                        expected = getattr(self.fixture.resident[layer], name)[
                            ids.cpu()
                        ]
                        self.assertTrue(torch.equal(output[name].cpu(), expected), name)
            self.assertEqual(runtime.prefill_seed_runs, 2)
            self.assertEqual(runtime.prefill_reused_experts, 8)
            self.assertEqual(runtime.prefill_reuse_d2d_bytes, 8 * one_expert)
            runtime.close()

    def test_prefill_cache_scatter_handles_vector_and_scalar_rows(self):
        from sglang.kernels.ops.moe.qwen38_expert_cache import cache_scatter

        for row_bytes in (7, 16, 1024):
            source = (
                torch.arange(3 * row_bytes, device="cuda")
                .to(torch.uint8)
                .reshape(3, row_bytes)
            )
            arena = torch.zeros((8, row_bytes), device="cuda", dtype=torch.uint8)
            keys = torch.tensor([5, -1, 2], device="cuda", dtype=torch.int32)
            counts = torch.zeros(9, device="cuda", dtype=torch.int32)
            counts[5] = counts[2] = 1
            cache_scatter(source, arena, keys, counts)
            self.assertTrue(torch.equal(arena[5], source[0]))
            self.assertTrue(torch.equal(arena[2], source[2]))
            self.assertEqual(arena[0].sum().item(), 0)

    def test_prefill_reuse_can_be_disabled_for_old_path_comparison(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=2,
            prefill_reuse_cache=False,
            prefill_seed_cache=False,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        for _ in range(2):
            for layer in (0, 1):
                ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
                before = runtime.transferred_bytes
                output = runtime.run(
                    layer,
                    ids,
                    lambda prepared: prepared.tensors["w13_weight"][
                        prepared.topk_ids.long()
                    ].clone(),
                )
                runtime.synchronize()
                self.assertGreater(runtime.transferred_bytes, before)
                self.assertTrue(
                    torch.equal(
                        output.cpu(), self.fixture.resident[layer].w13_weight[ids.cpu()]
                    )
                )
        self.assertEqual(runtime.prefill_seed_runs, 0)
        self.assertEqual(runtime.prefill_reused_experts, 0)

    def test_threaded_adaptive_repartition_preserves_budget_and_weights(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            adaptive_cache=True,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        pool_ptrs = {
            name: tensor.data_ptr() for name, tensor in runtime.cache_pool.items()
        }
        metadata_ptrs = [
            {name: tensor.data_ptr() for name, tensor in state.items()}
            for state in runtime.adaptive_metadata.device
        ]
        scratch_ptrs = {
            name: tensor.data_ptr()
            for name, tensor in runtime.migration_scratch.items()
        }
        for layout in ([3, 3], [2, 4], [4, 2]):
            if layout != [3, 3]:
                runtime.adaptive_target = layout
                runtime._maybe_rebalance()
            self.assertEqual(sum(runtime.layer_cache_slots), 6)
            self.assertEqual(
                {
                    name: tensor.data_ptr()
                    for name, tensor in runtime.cache_pool.items()
                },
                pool_ptrs,
            )
            self.assertEqual(
                [
                    {name: tensor.data_ptr() for name, tensor in state.items()}
                    for state in runtime.adaptive_metadata.device
                ],
                metadata_ptrs,
            )
            self.assertEqual(
                {
                    name: tensor.data_ptr()
                    for name, tensor in runtime.migration_scratch.items()
                },
                scratch_ptrs,
            )
            for layer in (0, 1):
                ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
                misses_before = runtime.misses
                output = runtime.run(
                    layer,
                    ids,
                    lambda prepared: {
                        name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                        for name in REQUIRED_TENSORS
                    },
                )
                runtime.synchronize()
                if layout != [3, 3]:
                    self.assertEqual(runtime.misses, misses_before)
                for name in REQUIRED_TENSORS:
                    expected = getattr(self.fixture.resident[layer], name)[ids.cpu()]
                    self.assertTrue(torch.equal(output[name].cpu(), expected), name)

    def test_threaded_breakable_verify_replays_after_adaptive_migration(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            adaptive_cache=True,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        pointers = {name: pool.data_ptr() for name, pool in runtime.cache_pool.items()}

        @eager_on_graph(
            True, capture_stub=lambda ids: torch.zeros_like(ids, dtype=torch.float32)
        )
        def experts(ids):
            return runtime.run(
                0,
                ids,
                lambda prepared: (
                    prepared.tensors["w13_weight"][prepared.topk_ids.long()]
                    .float()
                    .reshape(ids.numel(), -1)
                    .sum(-1)
                    .view_as(ids)
                ),
                execution_phase="verify",
            )

        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="cuda")
        experts(ids)
        runtime.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = BreakableCUDAGraph()
        with BreakableCUDAGraphCapture(graph, stream=stream):
            output = experts(ids + 0) * 1.0
        torch.cuda.current_stream().wait_stream(stream)
        for layout, pair in (([3, 3], [0, 1]), ([2, 4], [2, 3]), ([4, 2], [0, 3])):
            runtime.adaptive_target = layout
            ids.copy_(
                torch.tensor([pair, pair[::-1]], device="cuda", dtype=torch.int32)
            )
            graph.replay()
            runtime.synchronize()
            expected = self.fixture.resident[0].w13_weight[ids.cpu().long()]
            expected = (
                expected.float().reshape(ids.numel(), -1).sum(-1).view_as(ids.cpu())
            )
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            self.assertEqual(runtime.layer_cache_slots, layout)
            self.assertEqual(
                {name: pool.data_ptr() for name, pool in runtime.cache_pool.items()},
                pointers,
            )

    def test_threaded_adaptive_hotspot_changes_layout(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            adaptive_cache=True,
            adaptive_cache_window=16,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        for step in range(16):
            for layer, pair in (
                (0, ((0, 1), (1, 2), (2, 3))[step % 3]),
                (1, (0, 1)),
            ):
                ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
                runtime.run(
                    layer,
                    ids,
                    lambda prepared: prepared.tensors["w13_weight"][
                        prepared.topk_ids.long()
                    ].clone(),
                )
        runtime.jobs.join()
        runtime.adaptive_samples.join()
        self.assertEqual(runtime.adaptive_target, [4, 2])
        runtime._maybe_rebalance()
        self.assertEqual(runtime.layer_cache_slots, [4, 2])
        self.assertEqual(runtime.adaptive_rebalances, 1)

    def test_threaded_adaptive_retains_hot_nonprefix_slots(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack,
            "cuda",
            slots_per_layer=3,
            adaptive_cache=True,
        )
        self.addCleanup(runtime.close)
        runtime.reserve()
        for pair in ((0, 1), (2, 3), (2, 3), (2, 3)):
            ids = torch.tensor([pair], dtype=torch.int32, device="cuda")
            runtime.run(
                0,
                ids,
                lambda prepared: prepared.tensors["w13_weight"][
                    prepared.topk_ids.long()
                ].clone(),
            )
        runtime.synchronize()
        runtime.adaptive_target = [2, 4]
        runtime._maybe_rebalance()
        misses_before = runtime.misses
        ids = torch.tensor([[2, 3]], dtype=torch.int32, device="cuda")
        actual = runtime.run(
            0,
            ids,
            lambda prepared: prepared.tensors["w13_weight"][
                prepared.topk_ids.long()
            ].clone(),
        )
        runtime.synchronize()
        self.assertEqual(runtime.misses, misses_before)
        expected = self.fixture.resident[0].w13_weight[ids.cpu()]
        self.assertTrue(torch.equal(actual.cpu(), expected))

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
            self.assertTrue(
                torch.equal(
                    runtime.arena[name].cpu(), getattr(self.fixture.resident[0], name)
                ),
                name,
            )

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
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        runtime = Qwen38ThreadedExpertRuntime(
            self.fixture.pack, "cuda", slots_per_layer=2, warm_decode_cache=True
        )
        runtime.reserve()
        self.addCleanup(runtime.close)

        def execute(prepared):
            return {
                name: prepared.tensors[name][prepared.topk_ids.long()].clone()
                for name in REQUIRED_TENSORS
            }

        prefill = torch.tensor(
            [[1, 2], [1, 2], [0, 3]], dtype=torch.int32, device="cuda"
        )
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
                expected = getattr(self.fixture.resident[0], name)[
                    torch.tensor([[1, 2]])
                ]
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

    def test_packed_transport_alpha_ring_and_target_migration(self):
        from sglang.srt.layers.moe.qwen38_cache_threaded import (
            Qwen38ThreadedExpertRuntime,
        )
        from sglang.srt.model_loader.qwen38_nvfp4_pack.format import REQUIRED_TENSORS

        for alphas in (False, True):
            runtime = Qwen38ThreadedExpertRuntime(
                self.fixture.pack,
                "cuda",
                slots_per_layer=3,
                adaptive_cache=True,
                adaptive_cache_mode="sample",
                staging_buffers=2,
                staging_buffer_bytes=64,
                resident_alphas=alphas,
                prefill_copy_mode="sparse",
                prefill_reuse_cache=False,
                packed_decode="packed",
                packed_prefill="packed",
                packed_min_bytes=1,
                packed_max_bytes=1 << 20,
                packed_min_copies=2,
            )
            try:
                runtime.reserve()
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                runtime.prefill_tokens_seen = (
                    128  # explicitly test sparse, not startup dense
                )
                for step in range(24):
                    if step == 12:
                        with runtime.adaptive_lock:
                            runtime.adaptive_target = [2, 4]
                    layer = step % 2
                    cpu_ids = torch.tensor(
                        [[0, 2], [0, 2]]
                        if step % 3 == 0
                        else [[step % 4, (step + 1) % 4]]
                    )
                    with torch.cuda.stream(streams[step % 2]):
                        ids = cpu_ids.to("cuda", dtype=torch.int32)
                        actual = runtime.run(
                            layer,
                            ids,
                            lambda prepared: {
                                name: prepared.tensors[name][
                                    prepared.topk_ids.long()
                                ].clone()
                                for name in REQUIRED_TENSORS
                            },
                        )
                    runtime.synchronize()
                    for name in REQUIRED_TENSORS:
                        self.assertTrue(
                            torch.equal(
                                actual[name].cpu(),
                                getattr(self.fixture.resident[layer], name)[cpu_ids],
                            ),
                            name,
                        )
                self.assertGreater(
                    runtime.packed_transport.counters["scatter_calls"], 0
                )
                self.assertGreater(
                    runtime.packed_transport.counters["prefill_experts"], 0
                )
                self.assertGreaterEqual(
                    runtime.packed_transport.counters["target_refreshes"], 2
                )
                self.assertEqual(runtime.layer_cache_slots, [2, 4])
            finally:
                runtime.close()
