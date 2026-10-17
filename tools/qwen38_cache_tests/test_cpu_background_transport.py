# SPDX-License-Identifier: Apache-2.0
"""Worker lifetime/pipelining tests without CUDA or model weights."""

import ast
import contextlib
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source


class FakeCudaEvent:
    """Stand-in for torch.cuda.Event; records which stream it landed on."""

    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream

    def synchronize(self):
        return None


class TestBackgroundTransport(unittest.TestCase):
    def test_stats_logging_with_fixed_cache_and_adaptive_pool(self):
        # Execute the actual statistics/logging methods without CUDA transport.
        def methods(path, class_name, names):
            source = ast.parse((REPO_ROOT / path).read_text())
            cls = next(
                node
                for node in source.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            return [
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ]

        base = ast.ClassDef(
            name="StatsBase",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=methods(
                "python/sglang/srt/layers/moe/qwen38_cache_runtime.py",
                "Qwen38SynchronousExpertRuntime",
                {"log_stats", "_maybe_log_stats"},
            ),
        )
        threaded = ast.ClassDef(
            name="StatsRuntime",
            bases=[ast.Name(id="StatsBase", ctx=ast.Load())],
            keywords=[],
            decorator_list=[],
            body=methods(
                "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
                "Qwen38ThreadedExpertRuntime",
                {"stats_snapshot"},
            ),
        )
        logger = mock.Mock()
        namespace = {
            "json": json,
            "logger": logger,
            "_DENSE_STARTUP_PREFILL_TOKENS": 512,
        }
        tree = ast.fix_missing_locations(
            ast.Module(body=[base, threaded], type_ignores=[])
        )
        exec(compile(tree, "actual-cache-stats", "exec"), namespace)
        namespace["StatsBase"].stats_snapshot = lambda self: {
            "run_invocations": self.run_invocations
        }
        runtime = namespace["StatsRuntime"]()
        for name in (
            "cache_protected_fraction",
            "host_plan_allocations",
            "fused_decode_plan_invocations",
            "decode_hit_fast_path_invocations",
            "decode_device_gate_invocations",
            "decode_device_gate_hits",
            "decode_device_gate_copy_calls",
            "decode_device_gate_copy_bytes",
            "decode_alpha_h2d_calls_avoided",
            "prefill_reused_experts",
            "prefill_reuse_d2d_bytes",
            "prefill_seed_runs",
            "prefill_h2d_bytes",
            "prefill_host_dispatch_seconds",
            "prefill_tokens_seen",
            "adaptive_rebalances",
            "adaptive_sample_drops",
            "adaptive_migrated_experts",
            "adaptive_evicted_experts",
            "adaptive_last_rebalance_seconds",
            "adaptive_evaluations",
            "adaptive_analysis_seconds",
            "adaptive_last_evaluation",
        ):
            setattr(runtime, name, 0)
        runtime.layer_cache_slots = [9, 8]
        runtime.layer_hits = runtime.layer_misses = [0, 0]
        runtime.packed_decode = runtime.packed_prefill = runtime.prefill_copy_mode = (
            "direct"
        )
        runtime.borrowed_arena = False
        runtime.borrowed_restore_runs = runtime.borrowed_restore_experts = 0
        runtime.packed_transport = None
        runtime.cache_policy = "lfu"
        runtime.adaptive_cache_mode = "rebalance"
        runtime.adaptive_cache_window = 16
        runtime.adaptive_cache_interval = 0
        runtime.prefill_reuse_cache = runtime.prefill_seed_cache = True
        runtime.fused_decode_plan = runtime.resident_alphas = True
        runtime.decode_hit_fast_path = True
        runtime.decode_device_gate = False
        runtime.prefill_copy_stats = []
        runtime.run_invocations = runtime.stats_log_interval = 1
        for pool, adaptive, total in (
            (None, False, 17),
            ({"weight": SimpleNamespace(shape=(17, 1))}, True, 17),
        ):
            with self.subTest(adaptive=adaptive):
                runtime.cache_pool = pool
                runtime.adaptive_cache = adaptive
                runtime._last_logged_runs = 0
                logger.reset_mock()
                self.assertEqual(runtime.stats_snapshot()["cache_slots_total"], total)
                runtime._maybe_log_stats()
                runtime.log_stats("final")
                self.assertEqual(
                    [call.args[1] for call in logger.info.call_args_list],
                    ["periodic", "final"],
                )
                for call in logger.info.call_args_list:
                    self.assertEqual(
                        json.loads(call.args[2])["cache_slots_total"], total
                    )

    @classmethod
    def setUpClass(cls):
        # Import against lightweight stand-ins so the worker protocol can run
        # on hosts without PyTorch/SGLang runtime dependencies.
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                device=lambda _: contextlib.nullcontext(), Event=FakeCudaEvent
            ),
            no_grad=lambda: contextlib.nullcontext(),
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "sglang.srt.layers.moe.qwen38_borrowed_arena": load_source(
                    "borrowed_arena_worker_test",
                    "python/sglang/srt/layers/moe/qwen38_borrowed_arena.py",
                ),
                "sglang.kernels.ops.moe.qwen38_expert_cache": SimpleNamespace(
                    cache_plan=None,
                    cache_route=None,
                    cache_scatter=None,
                    cache_alpha_scatter=None,
                    cache_decode_plan=None,
                ),
                "sglang.srt.layers.moe.qwen38_cache_device": SimpleNamespace(
                    Qwen38DeviceExpertRuntime=object
                ),
                "sglang.srt.layers.moe.qwen38_expert_cache": SimpleNamespace(
                    LayerExpertCachePolicy=None
                ),
                "sglang.srt.model_loader.qwen38_nvfp4_pack.format": SimpleNamespace(
                    REQUIRED_TENSORS=()
                ),
                "sglang.srt.layers.moe.qwen38_cache_runtime": SimpleNamespace(
                    PreparedExperts=None
                ),
            },
        ):
            cls.module = load_source(
                "qwen38_threaded_worker_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_threaded.py",
            )

    def _runtime(self, permits):
        runtime = self.module.Qwen38ThreadedExpertRuntime.__new__(
            self.module.Qwen38ThreadedExpertRuntime
        )
        runtime.device = "fake"
        runtime.borrowed_arena = False
        runtime.borrowed_layers = {}
        runtime.miss_mailbox = None
        runtime.jobs = queue.Queue(maxsize=permits)
        runtime.retired = queue.Queue()
        runtime.capacity = threading.BoundedSemaphore(permits)
        runtime.worker_error = None
        runtime.poisoned = False
        runtime.warm_decode_cache = False
        runtime.cache_policy = "lfu"
        runtime.prefill_reuse_cache = False
        runtime.prefill_seed_cache = False
        runtime.resident_alphas = False
        runtime.decode_copy_names = tuple(self.module.REQUIRED_TENSORS)
        runtime.prefill_copy_mode = "auto"
        runtime.prefill_copy_stats = [
            dict.fromkeys(
                (
                    "dense",
                    "sparse",
                    "resident",
                    "startup_dense",
                    "forced_dense",
                    "heuristic_dense",
                    "missing",
                    "host_missing",
                    "missing_runs",
                    "reuse_candidates",
                ),
                0,
            )
            for _ in range(2)
        ]
        runtime.prefill_h2d_bytes = 0
        runtime.prefill_host_dispatch_seconds = 0.0
        runtime.transferred_bytes = 0
        runtime.slots_for_layer = lambda _: 2
        runtime.arena_residents = None
        runtime.manifest = SimpleNamespace(num_experts=3)
        runtime.arena_layer = 0
        runtime.arena_last_use = None
        runtime.transfer = SimpleNamespace(
            wait_event=lambda _: None, synchronize=lambda: None
        )
        return runtime

    def _job(self, planned, completed):
        launched = threading.Event()
        launched.set()
        # Arena jobs carry the pinned overflow counter; 0 means no bad route.
        plan = SimpleNamespace(tolist=lambda: [0, 0, 0, 0])
        return self.module.TransferJob(0, True, plan, planned, completed, launched)

    def test_synchronize_drains_jobs_before_device_waits(self):
        runtime = self._runtime(1)
        runtime.jobs.put(object())
        entered = threading.Event()
        device_sync = mock.Mock()
        join = runtime.jobs.join

        def wait_for_retirement():
            entered.set()
            join()

        with (
            mock.patch.object(runtime.jobs, "join", side_effect=wait_for_retirement),
            mock.patch.object(
                self.module,
                "super",
                return_value=SimpleNamespace(synchronize=device_sync),
                create=True,
            ),
        ):
            waiter = threading.Thread(target=runtime.synchronize)
            waiter.start()
            try:
                self.assertTrue(entered.wait(3))
                device_sync.assert_not_called()
            finally:
                runtime.jobs.get_nowait()
                runtime.jobs.task_done()
                waiter.join(3)
            self.assertFalse(waiter.is_alive())
            device_sync.assert_called_once_with()

    def _threads(self, runtime):
        worker = threading.Thread(target=runtime._work)
        reaper = threading.Thread(target=runtime._retire)
        reaper.start()
        worker.start()
        return worker, reaper

    def test_transport_reports_ready_events_while_compute_pending(self):
        # Slot readiness must be reported by CUDA events recorded on the
        # transfer stream; it must not wait on prior compute completion.
        runtime = self._runtime(2)
        completion = threading.Event()
        planned = SimpleNamespace(synchronize=lambda: None)
        worker, reaper = self._threads(runtime)
        try:
            jobs = []
            for _ in range(2):
                runtime.capacity.acquire()
                completed = SimpleNamespace(synchronize=lambda: completion.wait(5))
                jobs.append(self._job(planned, completed))
                runtime.jobs.put(jobs[-1])
            for job in jobs:
                self.assertTrue(
                    job.ready_set.wait(3), "worker did not report slot readiness"
                )
                self.assertIsNone(job.error)
                self.assertIsInstance(job.ready, FakeCudaEvent)
                self.assertIs(job.ready.recorded_on, runtime.transfer)
            self.assertIs(runtime.arena_last_use, jobs[-1].computed)
        finally:
            completion.set()
            runtime.jobs.join()
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reaper.is_alive())
        self.assertIsNone(runtime.worker_error)

    def test_transfer_failure_propagates_to_consumer_and_retires(self):
        runtime = self._runtime(1)
        planned = SimpleNamespace(synchronize=lambda: None)
        completed = SimpleNamespace(synchronize=lambda: None)
        worker, reaper = self._threads(runtime)
        bad_plan = SimpleNamespace(tolist=lambda: [1, 0, 0, 0])
        runtime.capacity.acquire()
        job = self.module.TransferJob(
            0, False, bad_plan, planned, completed, threading.Event()
        )
        job.launched.set()
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3), "worker did not report failure")
            self.assertIsInstance(job.error, ValueError)
            self.assertIsNone(job.ready)
            self.assertIsNotNone(runtime.worker_error)
        finally:
            runtime.jobs.join()
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reaper.is_alive())

    def test_adaptive_rejects_invalid_layout_before_migration(self):
        runtime = self._runtime(1)
        runtime.adaptive_lock = threading.Lock()
        runtime.manifest = SimpleNamespace(num_layers=2, num_experts=8, top_k=2)
        runtime.layer_cache_slots = [4, 4]
        runtime.adaptive_min_slots = 4
        runtime.cache_pool = {"weight": SimpleNamespace(shape=(8, 1))}
        runtime._migrate_pool = mock.Mock()
        for target in ([3, 5], [4, 5], [4], [4, True]):
            runtime.adaptive_target = target
            with self.assertRaises(ValueError):
                runtime._maybe_rebalance()
        runtime._migrate_pool.assert_not_called()
        runtime.adaptive_min_slots = 2
        runtime.adaptive_target = [3, 3]
        with self.assertRaises(ValueError):
            runtime._maybe_rebalance()
        runtime._migrate_pool.assert_not_called()

    def test_poisoned_runtime_rejects_new_work(self):
        runtime = self._runtime(1)
        runtime.poisoned = True
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            runtime._raise_error()

    def test_cross_layer_arena_waits_before_bulk_fill(self):
        runtime = self._runtime(1)
        previous = object()
        runtime.arena_last_use = previous
        runtime.arena_loads = 0
        operations = []
        runtime.transfer.wait_event = lambda event: operations.append(("wait", event))
        runtime._ensure_arena_experts = lambda layer, counts, keys, **kwargs: (
            operations.append(("fill", layer)) or True
        )
        planned = SimpleNamespace(synchronize=lambda: None)
        computed = SimpleNamespace(synchronize=lambda: None)
        job = self._job(planned, computed)
        job.layer = 1
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsNone(job.error)
            self.assertEqual(
                operations, [("wait", planned), ("wait", previous), ("fill", 1)]
            )
            self.assertEqual(runtime.arena_loads, 1)
            self.assertIs(runtime.arena_last_use, computed)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)

    def test_warmup_dispatch_precedes_ready_notice(self):
        runtime = self._runtime(1)
        runtime.warm_decode_cache = True
        warm_calls = []
        runtime._warm_from_arena = lambda layer, counts: warm_calls.append(
            (layer, counts)
        )
        job = self._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: None),
        )
        job.plan = SimpleNamespace(tolist=lambda: [5, 2, 0, 0])
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsNone(job.error)
            self.assertEqual(warm_calls, [(0, [5, 2, 0])])
            self.assertIs(job.ready.recorded_on, runtime.transfer)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)

    def test_warmup_failure_propagates_without_ready_event(self):
        runtime = self._runtime(1)
        runtime.warm_decode_cache = True
        runtime._warm_from_arena = mock.Mock(side_effect=RuntimeError("D2D failed"))
        job = self._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: None),
        )
        job.plan = SimpleNamespace(tolist=lambda: [4, 2, 0])
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsInstance(job.error, RuntimeError)
            self.assertIsNone(job.ready)
            self.assertIsNotNone(runtime.worker_error)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)

    def test_prefill_reuse_excludes_residents_from_dense_host_load(self):
        runtime = self._runtime(1)
        runtime.prefill_reuse_cache = True
        runtime.prefill_reused_experts = runtime.prefill_reuse_d2d_bytes = 0
        runtime.arena_residents = set()
        runtime.manifest = SimpleNamespace(
            num_experts=4, tensors={0: {"weight": SimpleNamespace(nbytes=400)}}
        )
        runtime.device_states = {0: {"keys": "keys", "counts": "counts"}}
        runtime.caches = {0: {"weight": "cache"}}
        runtime.arena = {"weight": "arena"}
        loaded = []
        runtime._fill_arena_selected = lambda layer, experts: loaded.append(experts)
        runtime._fill_arena = mock.Mock(
            side_effect=AssertionError("must not overwrite reuse with full H2D")
        )
        with (
            mock.patch.object(self.module, "REQUIRED_TENSORS", ("weight",)),
            mock.patch.object(self.module, "cache_scatter") as scatter,
            mock.patch.object(
                self.module.torch.cuda,
                "stream",
                create=True,
                new=lambda _: contextlib.nullcontext(),
            ),
        ):
            self.assertTrue(runtime._ensure_arena_experts(0, [1, 1, 1, 1, 0], [2, -1]))
            scatter.assert_called_once_with("cache", "arena", "keys", "counts")
        self.assertEqual(loaded, [[0, 1, 3]])
        self.assertEqual(runtime.arena_residents, {0, 1, 2, 3})
        self.assertEqual(runtime.prefill_reuse_d2d_bytes, 100)
        self.assertFalse(runtime._ensure_arena_experts(0, [1, 1, 1, 1, 0], [2, -1]))

    def test_prefill_copy_choice_accounts_for_fragmentation(self):
        choose = self.module._prefer_dense_prefill
        self.assertFalse(choose(list(range(112, 512)), 512))
        fragmented = [expert for expert in range(512) if expert % 5]
        self.assertTrue(choose(fragmented, 512))
        self.assertFalse(choose(list(range(384)), 512))

    def test_fragmented_prefill_uses_bulk_copy_before_scatter(self):
        runtime = self._runtime(1)
        runtime.prefill_reuse_cache = True
        runtime.arena_residents = set()
        runtime.manifest = SimpleNamespace(num_experts=512)
        runtime._fill_arena = mock.Mock()
        runtime._fill_arena_selected = mock.Mock()
        counts = [1] * 512 + [0]
        keys = list(range(0, 512, 5))
        with mock.patch.object(self.module, "cache_scatter") as scatter:
            self.assertTrue(runtime._ensure_arena_experts(0, counts, keys))
        runtime._fill_arena.assert_called_once_with(0)
        runtime._fill_arena_selected.assert_not_called()
        scatter.assert_not_called()
        self.assertIsNone(runtime.arena_residents)
        self.assertEqual(runtime.prefill_copy_stats[0]["heuristic_dense"], 1)

    def test_forced_sparse_preserves_reuse_on_fragmented_routes(self):
        runtime = self._runtime(1)
        runtime.prefill_copy_mode = "sparse"
        runtime.prefill_reuse_cache = True
        runtime.prefill_reused_experts = runtime.prefill_reuse_d2d_bytes = 0
        runtime.arena_residents = set()
        runtime.manifest = SimpleNamespace(num_experts=512)
        runtime.device_states = {0: {"keys": "keys", "counts": "counts"}}
        runtime.caches = {0: {}}
        runtime._fill_arena = mock.Mock()
        runtime._fill_arena_selected = mock.Mock()
        keys = list(range(0, 512, 5))
        with mock.patch.object(
            self.module.torch.cuda,
            "stream",
            create=True,
            new=lambda _: contextlib.nullcontext(),
        ):
            runtime._ensure_arena_experts(0, [1] * 512 + [0], keys)
        runtime._fill_arena.assert_not_called()
        runtime._fill_arena_selected.assert_called_once_with(
            0, [expert for expert in range(512) if expert % 5]
        )
        self.assertEqual(runtime.arena_residents, set(range(512)))
        self.assertEqual(runtime.prefill_copy_stats[0]["sparse"], 1)
        self.assertEqual(runtime.prefill_copy_stats[0]["reuse_candidates"], len(keys))

    def test_alpha_update_failure_poisoning_precedes_ready_notice(self):
        runtime = self._runtime(1)
        runtime.resident_alphas = True
        runtime.adaptive_cache = False
        # Resident alpha tables replace the two per-slot alpha H2Ds, so the
        # per-expert copy set must shrink to the weight tensors only.
        runtime.decode_copy_names = ("w1",)
        runtime.caches = {
            0: {"w1": "slot_w1", "g1_alphas": "slot_g1", "g2_alphas": "slot_g2"}
        }
        runtime.alpha_tables = {0: {"g1_alphas": "g1", "g2_alphas": "g2"}}
        runtime.device_states = {0: {"plan": "device_plan"}}
        runtime.decode_alpha_h2d_calls_avoided = 0
        runtime.slot_last_use = {0: [None, None]}
        runtime._fill = mock.Mock()
        job = self._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: None),
        )
        job.arena = False
        job.plan = SimpleNamespace(tolist=lambda: [0, 1, 1, 0, 2, 0, 0, 0, 0])
        runtime.capacity.acquire()
        with (
            mock.patch.object(
                self.module,
                "cache_alpha_scatter",
                side_effect=RuntimeError("alpha write failed"),
            ) as scatter,
            mock.patch.object(
                self.module.torch.cuda,
                "stream",
                create=True,
                new=lambda _: contextlib.nullcontext(),
            ),
        ):
            worker, reaper = self._threads(runtime)
            try:
                runtime.jobs.put(job)
                self.assertTrue(job.ready_set.wait(3))
                runtime.jobs.join()
                self.assertIsInstance(job.error, RuntimeError)
                self.assertIsNone(job.ready)
                self.assertIsNotNone(runtime.worker_error)
                runtime._fill.assert_called_once_with(
                    0, 2, runtime.caches[0], 0, names=("w1",)
                )
                scatter.assert_called_once_with(
                    "g1", "g2", "slot_g1", "slot_g2", "device_plan"
                )
                self.assertEqual(runtime.decode_alpha_h2d_calls_avoided, 0)
            finally:
                runtime.jobs.put(None)
                worker.join(3)
                runtime.retired.put(None)
                reaper.join(3)

    def test_startup_prefill_forces_bulk_copy_before_scatter(self):
        runtime = self._runtime(1)
        runtime.prefill_reuse_cache = True
        runtime.arena_residents = set()
        runtime.manifest = SimpleNamespace(num_experts=4)
        runtime._fill_arena = mock.Mock()
        runtime._fill_arena_selected = mock.Mock()
        with mock.patch.object(self.module, "cache_scatter") as scatter:
            self.assertTrue(
                runtime._ensure_arena_experts(
                    0, [1, 0, 0, 0, 0], [0, -1], force_dense=True
                )
            )
        runtime._fill_arena.assert_called_once_with(0)
        runtime._fill_arena_selected.assert_not_called()
        scatter.assert_not_called()

    def test_verify_arena_is_sparse_even_with_dense_prefill_policy(self):
        runtime = self._runtime(1)
        runtime.arena_residents = set()
        runtime.prefill_copy_mode = "dense"
        runtime._fill_arena = mock.Mock()
        runtime._fill_arena_selected = mock.Mock()
        self.assertTrue(runtime._ensure_arena_experts(0, [1, 0, 1, 0], verify=True))
        runtime._fill_arena.assert_not_called()
        runtime._fill_arena_selected.assert_called_once_with(0, [0, 2])

    def test_startup_prefill_counts_each_chunk_once(self):
        runtime = self._runtime(1)
        runtime.prefill_tokens_seen = 0
        runtime.prefill_dense_current = False
        self.assertTrue(runtime._dense_startup_prefill(0, True, 4096))
        self.assertTrue(runtime._dense_startup_prefill(1, True, 4096))
        self.assertEqual(runtime.prefill_tokens_seen, 4096)
        self.assertFalse(runtime._dense_startup_prefill(0, True, 4096))
        self.assertFalse(runtime._dense_startup_prefill(1, True, 4096))
        self.assertEqual(runtime.prefill_tokens_seen, 8192)

    def test_empty_cache_prefill_seed_precedes_readiness(self):
        runtime = self._runtime(1)
        runtime.prefill_seed_cache = True
        runtime.prefill_seed_runs = 0
        calls = []
        runtime._warm_from_arena = lambda layer, counts: calls.append((layer, counts))
        job = self._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: None),
        )
        job.plan = SimpleNamespace(tolist=lambda: [3, 2, 1, 0, -1, -1])
        runtime.capacity.acquire()
        worker, reaper = self._threads(runtime)
        try:
            runtime.jobs.put(job)
            self.assertTrue(job.ready_set.wait(3))
            runtime.jobs.join()
            self.assertIsNone(job.error)
            self.assertEqual(calls, [(0, [3, 2, 1])])
            self.assertEqual(runtime.prefill_seed_runs, 1)
        finally:
            runtime.jobs.put(None)
            worker.join(3)
            runtime.retired.put(None)
            reaper.join(3)
