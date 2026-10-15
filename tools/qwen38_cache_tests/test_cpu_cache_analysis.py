import json
import subprocess
import sys
import unittest

from . import test_cpu_background_transport as transport
from ._load_source import REPO_ROOT
from .replay_cache import minimum_slots


class TestCacheAnalysis(unittest.TestCase):
    def test_gpu_sample_batches_keep_original_analysis_protocol_and_gaps(self):
        import io
        import queue
        import threading
        from types import SimpleNamespace
        from unittest import mock

        transport.TestBackgroundTransport.setUpClass()
        runtime = transport.TestBackgroundTransport()._runtime(1)
        runtime.analysis_stopping = threading.Event()
        runtime.adaptive_samples = queue.Queue()
        runtime.adaptive_lock = threading.Lock()
        runtime.manifest = SimpleNamespace(num_layers=2, num_experts=8, top_k=2)
        runtime.layer_cache_slots = [2, 2]
        runtime.cache_pool = {"weight": SimpleNamespace(shape=(4, 1))}
        runtime.adaptive_cache = True
        runtime.adaptive_cache_window = 3
        runtime.adaptive_cache_mode = "observe"
        runtime.cache_protected_fraction = 0.7
        runtime.adaptive_evaluations = 0
        runtime.adaptive_analysis_seconds = 0
        runtime.adaptive_process = SimpleNamespace(
            stdin=io.StringIO(),
            stdout=io.StringIO(
                json.dumps(dict(seconds=0.1, evaluation={}, target=None)) + "\n"
            ),
        )
        # Cycle 0 has a gap and must not be analyzed. Cycle 1 mixes GPU batches
        # and the legacy verify observer, maintaining the original step indices.
        batches = []

        class Batch(tuple):
            pass

        for item in [
            (0, 0, 1, 3, [[2], [3]]),
            (1, 0, 0, 3, [[4], [4], [5]]),
            (0, 1, 0, 2, [[1], [2]]),
            (0, 1, 2, [3]),
            (1, 1, 0, 3, [[4], [4], [5]]),
            None,
        ]:
            if item is not None and len(item) == 5:
                item = Batch(item)
                item.release = mock.Mock()
                batches.append(item)
            runtime.adaptive_samples.put(item)
        with mock.patch.object(
            transport.TestBackgroundTransport.module.select,
            "select",
            return_value=([1], [], []),
        ):
            runtime._analyze_routes()
        requests = runtime.adaptive_process.stdin.getvalue().splitlines()
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            json.loads(requests[0])["routes"], [[[1], [2], [3]], [[4], [4], [5]]]
        )
        self.assertEqual(runtime.adaptive_evaluations, 1)
        self.assertTrue(runtime.adaptive_cache)
        self.assertEqual(runtime.adaptive_samples.unfinished_tasks, 0)
        for batch in batches:
            batch.release.assert_called_once_with()
        with self.assertRaises(ValueError):
            list(
                transport.TestBackgroundTransport.module._sample_rows(
                    (0, 2, 0, 2, [[1]])
                )
            )

    def test_minimum_slots_with_nonconcave_curves(self):
        curves = [{1: 10, 2: 9, 3: 1}, {1: 8, 2: 1, 3: 0}]
        self.assertEqual(
            minimum_slots(curves, 9),
            {"layer_cache_slots": [3, 1], "used_slots": 4, "misses": 9},
        )
        with self.assertRaises(ValueError):
            minimum_slots(curves, 0)

    def test_standalone_analysis_process_resets_between_windows(self):
        worker = (
            REPO_ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
        )
        request = dict(
            current=[2, 2],
            num_experts=8,
            top_k=1,
            policy="slru",
            protected_fraction=0.5,
            routes=[[[step % 3] for step in range(32)], [[7]] * 32],
        )
        result = subprocess.run(
            [sys.executable, str(worker)],
            input=(json.dumps(request) + "\n") * 2,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(replies), 2)
        self.assertEqual(replies[0]["target"], [3, 1])
        self.assertEqual(replies[0]["evaluation"], replies[1]["evaluation"])

    def test_contiguous_sampling_windows_and_prefill_boundary(self):
        transport.TestBackgroundTransport.setUpClass()
        runtime = transport.TestBackgroundTransport()._runtime(1)
        runtime.adaptive_cache = True
        runtime.adaptive_cache_window = 3
        runtime.adaptive_cache_interval = 2
        runtime.adaptive_steps = [0, 0]
        from types import SimpleNamespace

        single = SimpleNamespace(shape=(1, 1))
        positions = [runtime._sample_position(0, single) for _ in range(7)]
        self.assertEqual(
            positions, [(0, 0), (0, 1), (0, 2), None, None, (1, 0), (1, 1)]
        )
        self.assertIsNone(runtime._sample_position(0, SimpleNamespace(shape=(2, 1))))
        self.assertEqual(runtime._sample_position(0, single), (2, 0))

    def test_verify_sampling_retains_multi_token_round(self):
        from types import SimpleNamespace

        transport.TestBackgroundTransport.setUpClass()
        runtime = transport.TestBackgroundTransport()._runtime(1)
        runtime.adaptive_cache = True
        runtime.adaptive_cache_window = 3
        runtime.adaptive_cache_interval = 2
        runtime.adaptive_steps = [0, 0]
        runtime.sampling_verify = True
        for width in (2, 3, 4):
            self.assertEqual(
                runtime._sample_position(0, SimpleNamespace(shape=(width, 10))),
                (0, width - 2),
            )

    def test_analysis_verify_union_respects_minimum_and_reserved_budget(self):
        worker = (
            REPO_ROOT / "python/sglang/srt/layers/moe/qwen38_cache_analysis_worker.py"
        )
        request = dict(
            current=[4, 4],
            budget=12,
            min_slots=4,
            num_experts=16,
            top_k=2,
            policy="lfu",
            protected_fraction=0.5,
            routes=[
                [
                    [step % 8, (step + 1) % 8, (step + 2) % 8, (step + 3) % 8]
                    for step in range(32)
                ],
                [[12, 13, 14, 15]] * 32,
            ],
        )
        reply = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
        )
        result = json.loads(reply.stdout)
        self.assertIsNotNone(result["target"])
        self.assertEqual(sum(result["target"]), 12)
        self.assertGreaterEqual(min(result["target"]), 4)

    def test_retirement_reuses_plan_only_after_compute_completion(self):
        import queue
        import threading
        from types import SimpleNamespace

        transport.TestBackgroundTransport.setUpClass()
        runtime = transport.TestBackgroundTransport()._runtime(1)
        runtime.host_plans = {16: queue.SimpleQueue()}
        completed = threading.Event()
        plan = SimpleNamespace(numel=lambda: 16)
        job = transport.TestBackgroundTransport()._job(
            SimpleNamespace(synchronize=lambda: None),
            SimpleNamespace(synchronize=lambda: completed.wait(3)),
        )
        job.plan = plan
        runtime.capacity.acquire()
        runtime.jobs.put(job)
        runtime.jobs.get_nowait()
        reaper = threading.Thread(target=runtime._retire)
        reaper.start()
        try:
            runtime.retired.put(job)
            self.assertTrue(runtime.host_plans[16].empty())
            completed.set()
            runtime.retired.join()
            self.assertIs(runtime.host_plans[16].get_nowait(), plan)
        finally:
            completed.set()
            runtime.retired.put(None)
            reaper.join(3)
        self.assertFalse(reaper.is_alive())


if __name__ == "__main__":
    unittest.main()
