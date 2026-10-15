"""Exercise deferred device-counter accounting without CUDA or torch."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TestMailboxAccounting(unittest.TestCase):
    def setUp(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/layers/moe/qwen38_cache_mailbox.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("flush_stats", "_abort", "_watch")
        ]
        self.compute = mock.Mock()
        namespace = {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    current_stream=lambda _: self.compute,
                    device=lambda _: nullcontext(),
                )
            ),
            "time": SimpleNamespace(monotonic=lambda: 2.0),
        }
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.mailbox = namespace["DecodeMissMailbox"]()
        self.runtime = SimpleNamespace(
            device="cuda",
            layer_computed={0: object(), 1: object()},
            transferred_bytes=0,
            decode_device_gate_copy_calls=0,
            decode_device_gate_copy_bytes=0,
            decode_device_gate_hits=0,
            resident_alphas=True,
            decode_alpha_h2d_calls_avoided=0,
            misses=0,
            hits=0,
            layer_misses=[0, 0],
            layer_hits=[0, 0],
        )
        self.mailbox.runtime = self.runtime
        self.mailbox.sizes = {0: (16, 32), 1: (32, 64)}
        self.mailbox.observed = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.mailbox.hit_stats = mock.Mock()

    def snapshot(self, rows):
        self.mailbox.hit_stats.cpu.return_value.tolist.return_value = rows
        self.mailbox.flush_stats()

    def test_deferred_multilayer_misses_and_hits_are_not_lost_or_double_counted(self):
        # Several same-layer plans completed before the CPU reads statistics.
        rows = [[7, 2, 3, 2], [4, 1, 5, 3]]
        self.snapshot(rows)
        r = self.runtime
        self.assertEqual((r.hits, r.misses), (11, 8))
        self.assertEqual(r.layer_hits, [7, 4])
        self.assertEqual(r.layer_misses, [3, 5])
        self.assertEqual(
            (r.decode_device_gate_hits, r.decode_device_gate_copy_calls), (3, 5)
        )
        self.assertEqual(r.transferred_bytes, 3 * 48 + 5 * 96)
        self.assertEqual(r.decode_device_gate_copy_bytes, r.transferred_bytes)
        self.assertEqual(r.decode_alpha_h2d_calls_avoided, 16)
        for event in r.layer_computed.values():
            self.compute.wait_event.assert_any_call(event)
        self.snapshot(rows)
        self.assertEqual(
            (r.hits, r.misses, r.decode_device_gate_copy_calls), (11, 8, 5)
        )
        self.assertEqual(r.transferred_bytes, 624)
        self.snapshot([[9, 3, 4, 3], [4, 1, 5, 3]])
        self.assertEqual(
            (r.hits, r.misses, r.decode_device_gate_copy_calls), (13, 9, 6)
        )
        self.assertEqual(r.transferred_bytes, 672)

    def test_nonresident_alpha_and_mixed_hit_miss_plan_accounting(self):
        self.runtime.resident_alphas = False
        self.snapshot([[1, 0, 1, 1], [0, 0, 0, 0]])
        self.assertEqual((self.runtime.hits, self.runtime.misses), (1, 1))
        self.assertEqual(self.runtime.decode_device_gate_hits, 0)
        self.assertEqual(self.runtime.decode_alpha_h2d_calls_avoided, 0)
        self.assertEqual(self.runtime.transferred_bytes, 48)

    def test_abort_poisoning_needs_no_ready_buffer_or_cuda_submission(self):
        self.mailbox.failed = [0]
        error = TimeoutError("decode expired")
        self.mailbox._abort(error)
        self.assertIs(self.runtime.worker_error, error)
        self.assertTrue(self.runtime.poisoned)
        self.assertEqual(self.mailbox.failed, [1])
        self.compute.assert_not_called()

    def test_completion_watchdog_still_poisoned_on_expired_gpu_reader(self):
        self.mailbox.failed = [0]
        self.mailbox.stopping = mock.Mock()
        self.mailbox.stopping.wait.return_value = False
        self.mailbox.lock = nullcontext()
        self.mailbox.pending = [(object(), 1.0, 0)]
        self.mailbox._retire_completed = mock.Mock()
        self.mailbox._watch()
        self.assertTrue(self.runtime.poisoned)
        self.assertIsInstance(self.runtime.worker_error, TimeoutError)
        self.assertEqual(self.mailbox.failed, [1])
