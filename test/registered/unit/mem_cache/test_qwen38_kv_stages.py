"""One-way KV loan boundaries, publication ordering and allocator reset safety."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.qwen38_kv_stages import (
    OneWayKVStages,
    StagedKVLoan,
    distribute_slots,
    page_capacities,
    validate_stages,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestKVStages(CustomTestCase):
    def test_decode_after_long_prefill_keeps_next_chunk_reserve(self):
        """74K cached tokens must not shrink to 80K before an 8192-token extend."""
        loan = StagedKVLoan.__new__(StagedKVLoan)
        loan.policy = OneWayKVStages([80000, 160000, 220032], lambda _: None)
        loan.policy.index = 1
        loan.runtime = SimpleNamespace(pending_kv_stage=None, kv_reborrow_probe=False)
        loan.reserve = 8192
        loan.used_tokens = lambda: 74048
        loan.before_forward(8192)
        self.assertFalse(loan.runtime.kv_reborrow_probe)
        self.assertEqual(loan.reserve, 8192)
        loan.used_tokens = lambda: 1024
        loan.before_forward(8192)
        self.assertTrue(loan.runtime.kv_reborrow_probe)

    def test_large_decode_reserve_drops_with_current_request_demand(self):
        loan = StagedKVLoan.__new__(StagedKVLoan)
        loan.policy = OneWayKVStages([80000, 160000, 220032], lambda _: None)
        loan.policy.index = 2
        loan.runtime = SimpleNamespace(pending_kv_stage=None, kv_reborrow_probe=False)
        loan.reserve = 0
        loan.used_tokens = lambda: 1024
        loan.before_forward(90000)
        self.assertFalse(loan.runtime.kv_reborrow_probe)
        loan.before_forward(8192)
        self.assertEqual(loan.reserve, 8192)
        self.assertTrue(loan.runtime.kv_reborrow_probe)

    def test_host_cache_reclaim_keeps_active_demand_and_pending_returns(self):
        from unittest.mock import Mock

        import torch

        loan = StagedKVLoan.__new__(StagedKVLoan)
        loan.policy = OneWayKVStages([80000, 160000, 220032], lambda _: None)
        loan.policy.index = 2
        loan.runtime = SimpleNamespace(pending_kv_stage=None)
        loan.allocator = SimpleNamespace(
            get_all_free_pages=lambda: torch.tensor([2, 1])
        )
        cache = Mock()
        cache.reclaim_device_tail.return_value = 0
        cache.full_protected_size.return_value = 74048
        loan.reclaim_host_cached_tail(cache, 8192)
        cache.reclaim_device_tail.assert_called_once_with(160000)
        self.assertEqual(loan.allocator.free_pages.tolist(), [1, 2])
        cache.reclaim_device_tail.reset_mock()
        cache.full_protected_size.return_value = 1024
        loan.reclaim_host_cached_tail(cache, 8192)
        cache.reclaim_device_tail.assert_called_once_with(80000)
        cache.reclaim_device_tail.reset_mock()
        loan.runtime.pending_kv_stage = 2
        loan.reclaim_host_cached_tail(cache, 8192)
        cache.reclaim_device_tail.assert_not_called()

    def test_cpu_replay_does_not_charge_or_retain_gpu_embeddings(self):
        """CPU replay's 100 MiB setting used to subtract from the 220K KV budget."""
        import torch

        from sglang.srt.disaggregation.encoder.replay_cache import (
            gpu_embedding_cache_size_mb,
        )
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.kv_cache_configurator import mm_runtime_reservation_gb
        from sglang.srt.mem_cache.multimodal_cache import (
            EmbeddingResult,
            MultiModalStaticCache,
        )
        from sglang.srt.runtime_context import get_context

        with (
            get_context().override_server_args(
                language_only=True, encoder_transfer_backend="zmq_to_scheduler"
            ),
            envs.SGLANG_VLM_CACHE_SIZE_MB.override(100),
        ):
            self.assertEqual(
                mm_runtime_reservation_gb(
                    is_multimodal=True, mm_feature_transport=None
                ),
                0,
            )
            cache = MultiModalStaticCache(gpu_embedding_cache_size_mb() * 1024 * 1024)
            self.assertFalse(cache.set(1, EmbeddingResult(embedding=torch.ones(8))))
            self.assertEqual(cache.current_size, 0)
        with (
            get_context().override_server_args(language_only=False),
            envs.SGLANG_VLM_CACHE_SIZE_MB.override(100),
        ):
            self.assertEqual(
                mm_runtime_reservation_gb(
                    is_multimodal=True, mm_feature_transport=None
                ),
                100 / 1024,
            )

    def test_prefill_equality_and_no_reborrow(self):
        returns = []
        stages = OneWayKVStages([80000, 160000, 220000], returns.append)
        self.assertEqual(stages.prepare(71807, 8192), 80000)
        self.assertEqual(stages.prepare(71808, 8192), 160000)
        self.assertEqual(returns, [1])
        self.assertEqual(stages.prepare(0, 8192), 160000)
        self.assertEqual(stages.prepare(151808, 8192), 220000)
        self.assertEqual(stages.prepare(0, 1), 220000)
        self.assertEqual(returns, [1, 2])

    def test_spec_candidates_and_multi_stage_jump(self):
        for candidates in (1, 2, 3, 4):
            with self.subTest(candidates=candidates):
                returned = []
                stages = OneWayKVStages([80000, 160000, 220000], returned.append)
                self.assertEqual(
                    stages.prepare(80000 - candidates - 1, candidates), 80000
                )
                self.assertEqual(stages.prepare(80000 - candidates, candidates), 160000)
        returned = []
        stages = OneWayKVStages([80000, 160000, 220000], returned.append)
        stages.prepare(0, 180000)
        self.assertEqual(returned, [1, 2])

    def test_failure_does_not_publish_capacity_or_retry_serving(self):
        observations = []

        def fail(index):
            observations.append((index, stages.capacity))
            raise RuntimeError("expert reader failed")

        stages = OneWayKVStages([80000, 160000, 220000], fail)
        with self.assertRaisesRegex(RuntimeError, "expert reader"):
            stages.prepare(71808, 8192)
        self.assertEqual(observations, [(1, 80000)])
        self.assertEqual(stages.capacity, 80000)
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            stages.prepare(0, 1)

    def test_validation_and_page_rounding(self):
        for invalid in ([], [1, 2], [True, 2, 3], [1, 1, 2], [0, 2, 3], [3, 2, 1]):
            with self.subTest(stages=invalid), self.assertRaises(ValueError):
                validate_stages(invalid)
        self.assertEqual(
            page_capacities([80000, 160000, 220000], 64), (80000, 160000, 220032)
        )
        with self.assertRaises(ValueError):
            page_capacities([1, 2, 3], 64)

    def test_resize_budget_bounds_and_hot_layer_preference(self):
        for budget in (12, 17, 40, 64):
            layout = distribute_slots([4, 20, 8, 4], budget, 3, 16)
            self.assertEqual(sum(layout), budget)
            self.assertTrue(all(3 <= n <= 16 for n in layout))
            self.assertGreaterEqual(layout[1], layout[0])
        with self.assertRaises(ValueError):
            distribute_slots([10, 10], 19, 10, 512)

    def test_real_allocator_returns_at_existing_result_boundary(self):
        import torch

        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        allocator = PagedTokenToKVPoolAllocator(
            256, 64, torch.float32, "cpu", None, False
        )
        operations = []
        runtime = SimpleNamespace(
            kv_cache_stages=[64, 128, 256],
            expert_loan_budgets=[30, 20, 10],
            pending_kv_stage=None,
            staged_trim_enqueued=False,
            staged_pool=SimpleNamespace(
                release_stage=lambda i: operations.append(("expert", i))
            ),
        )
        pool = SimpleNamespace(
            _post_capture_owner=SimpleNamespace(
                finalize=lambda n: operations.append(("kv", n))
            )
        )
        allocator.staged_loan = StagedKVLoan(allocator, pool, runtime)
        self.assertEqual(allocator.available_size(), 256)
        allocator.clear()
        self.assertEqual(allocator.free_pages.tolist(), [1])
        result = object()
        allocator.alloc(64)
        allocator.cache_maintenance_before_forward(64)
        self.assertEqual(runtime.pending_kv_stage, 2)
        self.assertTrue(allocator.cache_maintenance_needs_result())
        allocator.cache_maintenance_bind_result(result)
        allocator.cache_maintenance_after_result(object())
        self.assertEqual(operations, [])
        # A queued trim is not sufficient; this callback runs after copy_done.
        allocator.cache_maintenance_after_result(result)
        self.assertEqual(operations, [])
        runtime.staged_trim_enqueued = True
        allocator.cache_maintenance_after_result(result)
        self.assertEqual(operations, [("expert", 2), ("kv", 128), ("kv", 256)])
        allocator.clear()
        self.assertEqual(allocator.free_pages.tolist(), [1, 2, 3, 4])
        allocator.cache_maintenance_after_result(result)
        self.assertEqual(len(operations), 3)

    def test_physical_allocation_cannot_trigger_a_host_wait_or_early_return(self):
        loan = StagedKVLoan.__new__(StagedKVLoan)
        loan.allocator = SimpleNamespace(
            backed_num_pages=1250,
            free_pages=range(128),
            num_staged_pages=0,
            page_size=64,
        )
        loan.policy = OneWayKVStages(
            [80000, 160000, 220032], lambda _: self.fail("early return")
        )
        # Cached and in-flight pages are charged together.
        self.assertEqual(loan.used_tokens(), 71808)
        loan.prepare(8192)
        self.assertTrue(loan.fits_allocation(8192))
        self.assertFalse(loan.fits_allocation(8193))
        loan.prepare(8193)
        self.assertEqual(loan.policy.capacity, 80000)


if __name__ == "__main__":
    unittest.main()
