# SPDX-License-Identifier: Apache-2.0
"""Execute runtime resize with byte-backed tensors; check DMA/publication order."""

import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[5]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Row:
    def __init__(self, storage, slot, operations, fail):
        self.storage, self.slot, self.operations, self.fail = (
            storage,
            slot,
            operations,
            fail,
        )

    def copy_(self, source):
        self.operations.append(("move", source.slot, self.slot))
        if self.fail:
            raise RuntimeError("injected transfer failure")
        self.storage[self.slot] = source.storage[source.slot]


class Tensor:
    def __init__(self, storage, operations, fail=False):
        self.storage, self.operations, self.fail = storage, operations, fail

    def __getitem__(self, slot):
        return Row(self.storage, slot, self.operations, self.fail)


class TestGlobalResizeProtocol(CustomTestCase):
    def setUp(self):
        names = (
            "sglang.srt.layers.moe.qwen38_hybrid_policy",
            "hybrid_resize_subject",
            "global_kv_resize_subject",
        )
        previous = {name: sys.modules.get(name) for name in names}

        def restore_modules():
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        self.addCleanup(restore_modules)
        self.operations = []
        operations = self.operations

        class Event:
            def record(self, stream):
                operations.append(("record",))

            def synchronize(self):
                operations.append(("dma_complete",))

        class Stream:
            def wait_stream(self, stream):
                operations.append(("dependency",))

        self.stream = Stream()
        torch_stub = ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(
            current_stream=lambda _: self.stream,
            stream=lambda _: contextlib.nullcontext(),
            Event=Event,
        )
        env_stub = ModuleType("sglang.srt.environ")
        env_stub.envs = SimpleNamespace()
        self.policy_module = load(
            "sglang.srt.layers.moe.qwen38_hybrid_policy",
            "python/sglang/srt/layers/moe/qwen38_hybrid_policy.py",
        )
        with patch.dict(
            sys.modules, {"torch": torch_stub, "sglang.srt.environ": env_stub}
        ):
            self.runtime_module = load(
                "hybrid_resize_subject", "python/sglang/srt/layers/moe/qwen38_hybrid.py"
            )

    def subject(self, fail=False):
        policy = self.policy_module.GlobalLfu(2, 8, [2, 2])
        policy.seed(0, {0: 0, 1: 1})
        policy.seed(1, {0: 2, 1: 3})
        policy.frequency[0][:2] = [1, 2]
        policy.frequency[1][:2] = [8, 9]
        storage = [bytes([key[0], key[1]]) * 32 for key in policy.keys] + [None] * 4
        tensor = Tensor(storage, self.operations, fail)
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.lookups = {}
        subject.policy = policy
        subject._seed = lambda: None
        subject.drain = lambda: self.operations.append(("drain",))
        subject.fill_stream = self.stream
        subject.stats = __import__("collections").Counter()
        subject.failure = None
        subject.runtime = SimpleNamespace(
            staged_pool=SimpleNamespace(pool_views=lambda budget: {"w": tensor}),
            layer_computed={},
            device="cuda",
            transfer=self.stream,
            manifest=SimpleNamespace(num_layers=2),
            _refresh_staged_arena=lambda: self.operations.append(("arena_refresh",)),
            adaptive_migrated_experts=0,
            adaptive_evicted_experts=0,
            poisoned=False,
        )
        commit = policy.commit_resize

        def publish(plan):
            self.operations.append(("publish",))
            commit(plan)

        policy.commit_resize = publish
        return subject, storage

    def test_all_moves_complete_before_mapping_and_tail_release(self):
        subject, storage = self.subject()
        subject.resize(2)
        self.operations.append(("unmap_tail",))
        self.assertEqual(storage[:2], [bytes([1, 1]) * 32, bytes([1, 0]) * 32])
        self.assertEqual(subject.policy.layers[0].mapping, {})
        self.assertEqual(set(subject.policy.layers[1].mapping.values()), {0, 1})
        names = [item[0] for item in self.operations]
        self.assertLess(names.index("drain"), names.index("move"))
        self.assertLess(
            max(i for i, n in enumerate(names) if n == "move"),
            names.index("dma_complete"),
        )
        self.assertLess(names.index("dma_complete"), names.index("publish"))
        self.assertLess(names.index("publish"), names.index("unmap_tail"))

    def test_copy_failure_poisoned_without_publication(self):
        subject, _ = self.subject(fail=True)
        before = list(subject.policy.keys)
        with self.assertRaises(RuntimeError):
            subject.resize(2)
        self.assertEqual(subject.policy.keys, before)
        self.assertTrue(subject.runtime.poisoned)
        self.assertIsNotNone(subject.failure)
        self.assertNotIn("publish", [item[0] for item in self.operations])

    def test_growth_never_moves_or_evicts(self):
        subject, storage = self.subject()
        before = list(storage)
        subject.resize(7)
        self.assertEqual(storage, before)
        self.assertEqual(subject.policy.owner[4:], [None] * 3)
        self.assertEqual(subject.stats["resize_moves"], 0)
        self.assertEqual(subject.stats["resize_evictions"], 0)

    def test_seed_restores_legacy_arena_before_importing_keys(self):
        import collections
        import threading

        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        physical = [2]
        policy = self.policy_module.GlobalLfu(1, 4, [1])
        operations = self.operations
        runtime = SimpleNamespace(
            lock=threading.Lock(),
            _raise_error=lambda: operations.append(("old_errors",)),
            _drain_arena_readers=lambda: operations.append(("old_readers_drained",)),
            layer_cache_slots=[1],
            device_states={0: {"keys": Tensor([1]), "frequency": Tensor([2.0])}},
            manifest=SimpleNamespace(num_layers=1),
            arena_residents={2},
            arena_layer=0,
        )

        def return_arena():
            physical[0] = 1
            operations.append(("old_bytes_restored",))

        runtime._return_arena = return_arena
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.runtime = runtime
        subject.lookups = {}
        subject.policy = policy
        subject.seeded = False
        subject.pool = {}
        subject.b12x = {}
        subject.stats = collections.Counter()
        subject._seed()
        self.assertEqual(policy.layers[0].mapping, {1: 0})
        self.assertEqual(physical[policy.layers[0].mapping[1]], 1)
        self.assertEqual(runtime.arena_residents, set())
        self.assertIn("old_bytes_restored", [op[0] for op in operations])

    def test_seed_propagates_legacy_failure_without_publishing(self):
        import threading

        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.seeded = False

        def fail():
            raise RuntimeError("legacy DMA failure")

        subject.runtime = SimpleNamespace(lock=threading.Lock(), _raise_error=fail)
        with self.assertRaisesRegex(RuntimeError, "legacy DMA failure"):
            subject._seed()
        self.assertFalse(subject.seeded)

    def test_b12x_dispatch_preserves_masked_route_outputs(self):
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.gpu_backend = "b12x"
        subject.pool = {}
        subject.runtime = SimpleNamespace(layer_scales={0: {}})
        subject.b12x = {
            0: SimpleNamespace(
                run=lambda x, ids, weights, tensors, **kwargs: sum(
                    (slot + 1) * weight
                    for slot, weight in zip(ids, weights)
                    if slot >= 0
                )
            )
        }
        self.runtime_module.torch.float32 = "float32"
        # Noncontiguous global slots and inactive routes cover all/mixed/no hits.
        for ids, expected in (([600, 1024], 1626), ([600, -1], 601), ([-1, -1], 0)):
            with self.subTest(ids=ids):
                self.assertEqual(subject._gpu(None, ids, [1, 1], 0), expected)

    def test_missing_or_failed_b12x_raises_without_baseline(self):
        subject = self.runtime_module.HybridCache.__new__(
            self.runtime_module.HybridCache
        )
        subject.gpu_backend = "b12x"
        subject.pool = {}
        subject.runtime = SimpleNamespace(layer_scales={0: {}})
        subject.b12x = {}
        with self.assertRaisesRegex(RuntimeError, "B12X was selected"):
            subject._gpu(None, [-1], [1], 0)

        def fail(*args, **kwargs):
            raise RuntimeError("injected B12X launch failure")

        subject.b12x[0] = SimpleNamespace(run=fail)
        self.runtime_module.torch.float32 = "float32"
        with self.assertRaisesRegex(RuntimeError, "injected B12X launch failure"):
            subject._gpu(None, [700], [1], 0)

    def test_global_kv_return_compacts_even_when_legacy_preserve_is_false(self):
        module = load(
            "global_kv_resize_subject",
            "python/sglang/srt/mem_cache/qwen38_kv_stages.py",
        )
        operations = self.operations

        class Torch:
            @staticmethod
            def arange(*args, **kwargs):
                return []

            @staticmethod
            def cat(items):
                return []

            int64 = "int64"

        loan = module.StagedKVLoan.__new__(module.StagedKVLoan)
        loan.allocator = SimpleNamespace(
            backed_num_pages=2, page_size=1, device="cpu", free_pages=[]
        )
        loan.policy = SimpleNamespace(capacities=[2, 4, 6], index=0)
        loan.pool = SimpleNamespace(
            _post_capture_owner=SimpleNamespace(
                finalize=lambda _: operations.append(("kv_grow",))
            )
        )
        loan.runtime = SimpleNamespace(
            staged_trim_enqueued=True,
            staged_trim_done=SimpleNamespace(query=lambda: True),
            hybrid=object(),
            preserve_cache_on_resize=False,
            complete_staged_trim=lambda: operations.append(("compact_complete",)),
            staged_pool=SimpleNamespace(
                release_stage=lambda _: operations.append(("expert_unmap",))
            ),
            expert_loan_budgets=[8, 6, 4],
            pending_kv_stage=1,
        )
        with patch.dict(sys.modules, {"torch": Torch}):
            loan._return(1)
        self.assertEqual(
            [item[0] for item in operations],
            ["compact_complete", "expert_unmap", "kv_grow"],
        )


if __name__ == "__main__":
    unittest.main()
