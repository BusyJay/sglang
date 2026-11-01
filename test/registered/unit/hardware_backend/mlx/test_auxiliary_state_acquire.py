"""MLX hybrid admission through the shared acquire entry point.

The MLX request table binds a request's auxiliary-state slot in its own
``alloc``, with the row, so the state storage it carries has nothing left to
bind. Admission used to fail on the first hybrid request, after the row and
the slot were taken, because the generic acquire read ReplaySSM fields the MLX
pool does not have. MLX-gated because the table's module imports mlx.core.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocation import ensure_mamba_capacity
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TreeSlot
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.components.base import ComponentData
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
        MlxAuxiliaryStateComponent,
        MlxAuxiliaryStateReqToTokenPool,
    )


def _req():
    return SimpleNamespace(kv=ReqKvInfo())


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestMlxAuxiliaryStateAcquire(CustomTestCase):
    def setUp(self):
        self.table = MlxAuxiliaryStateReqToTokenPool(
            size=4,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        self.slots = self.table.auxiliary_state_pool

    def test_prefix_snapshot_fork_restore_shares_storage_and_isolates_writes(self):
        """A pool fork must retain latent state after parent writes and slot release."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import MLACache

        cache = MLACache(4)
        cache.append_latent(mx.ones((1, 65536, 512), mx.bfloat16), materialize=False)
        cache.offset = 65536
        cache.append_latent(mx.full((1, 1, 512), 2, mx.bfloat16), materialize=False)
        cache.offset += 1
        mx.eval(cache.cache)
        parent = self.slots.alloc(1)
        before = mx.get_active_memory()
        self.slots.store_cache(parent, [cache], [0])
        child = self.slots.fork_from(parent)
        restored = MLACache(4)
        self.assertTrue(self.slots.restore_cache(child, [restored]))
        mx.eval(restored.cache)
        # The 64MiB prefix must be shared, allowing some allocator metadata.
        self.assertLess(mx.get_active_memory() - before, 1024 * 1024)
        self.slots.free(parent)
        restored.append_latent(mx.full((1, 2, 512), 7, mx.bfloat16), materialize=False)
        restored.offset += 2
        restored.cache[0][:, 0] = -3
        mx.eval(restored.cache)
        again = MLACache(4)
        self.assertTrue(self.slots.restore_cache(child, [again]))
        self.assertEqual(again.offset, 65537)
        self.assertTrue(mx.array_equal(again[0], cache[0]).item())
        self.assertEqual(again[0][0, 0, 0].item(), 1)
        self.slots.free(child)
        self.assertEqual(self.slots.available_size(), 4)

    def test_first_admission_binds_the_row_and_one_slot(self):
        req = _req()
        rows = acquire_rows(self.table, self.table.state, [req])
        self.assertEqual(len(rows), 1)
        self.assertTrue(req.kv.holds_mamba)
        self.assertEqual(self.slots.available_size(), 3)
        self.assertEqual(
            int(self.table.get_auxiliary_state_indices(rows)[0]),
            int(req.kv.mamba_pool_idx),
        )

    def test_a_request_holding_a_slot_keeps_it(self):
        req = _req()
        held = self.slots.alloc(1)[0]
        req.kv.mamba_pool_idx = held
        rows = acquire_rows(self.table, self.table.state, [req])
        self.assertEqual(int(req.kv.mamba_pool_idx), int(held))
        self.assertEqual(self.slots.available_size(), 3)
        self.assertEqual(
            int(self.table.get_auxiliary_state_indices(rows)[0]), int(held)
        )

    def test_multilayer_snapshot_seals_lazy_state_and_dflash_context(self):
        """One sealed prefix must survive later writes to every source layer."""
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

        cache = [NativeCache(4) for _ in range(3)]
        for i, layer in enumerate(cache):
            layer.offset = 8
            layer.cache = [mx.arange(16, dtype=mx.float32).reshape(1, 16) + i] + [
                None
            ] * 3
        features = mx.arange(24, dtype=mx.float32).reshape(1, 8, 3) * 2
        cache[0].mlx_dflash = (8, (0, 2), {"keys": (features,)}, (), ())
        slot = self.slots.alloc(1)
        self.slots.store_cache(slot, cache, range(3))
        for layer in cache:
            layer.cache[0][:] = -1
            layer.offset = 9
        features[:] = -2
        restored = [NativeCache(4) for _ in range(3)]
        self.assertTrue(self.slots.restore_cache(slot, restored))
        for i, layer in enumerate(restored):
            self.assertEqual(layer.offset, 8)
            self.assertTrue(
                mx.array_equal(layer.cache[0], mx.arange(16).reshape(1, 16) + i).item()
            )
        self.assertTrue(
            mx.array_equal(
                restored[0].mlx_dflash[2]["keys"][0], mx.arange(24).reshape(1, 8, 3) * 2
            ).item()
        )

    def test_admission_checks_auxiliary_capacity(self):
        self.slots.alloc(4)
        self.assertFalse(
            ensure_mamba_capacity(self.table, self.table.state, [_req()], None)
        )

    def test_first_prefix_match_uses_initialized_checkpoint_grid(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.table,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
        with get_context().override_server_args():
            component = MlxAuxiliaryStateComponent(
                SimpleNamespace(req_to_token_pool=self.table),
                params,
                TreeSlot(component="MAMBA", family="state", checkpoint_interval=4),
            )
        node = SimpleNamespace(
            component_data={component.component_type: ComponentData()}
        )
        result = MatchResult(0, node, node, node, full_kv_hit_length=7)
        matched = component.finalize_match_result_in_tree_core(result, None, [], 0)
        self.assertEqual(matched.mamba_branching_seqlen, 4)


if __name__ == "__main__":
    unittest.main()
