import unittest
from types import SimpleNamespace

from tools.qwen38_cache_tests._load_source import load_source

mod = load_source(
    "borrowed_arena", "python/sglang/srt/layers/moe/qwen38_borrowed_arena.py"
)


class FakeKeys:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Runtime(mod.BorrowedArenaMixin):
    def __init__(self):
        self.borrowed_arena = True
        self.borrowed_records = None
        self.borrowed_layers = {}
        self.layer_cache_slots = [3, 3, 3]
        self.manifest = SimpleNamespace(num_experts=4)
        self.device_states = {i: {"keys": FakeKeys([0, -1, 2])} for i in range(3)}
        self.caches = {i: i for i in range(3)}
        self.log = []
        self.transfer = self.metadata_stream = SimpleNamespace(synchronize=lambda: None)
        self.borrowed_restore_experts = self.borrowed_restore_runs = 0

    def _drain_arena_readers(self):
        self.log.append("drain")

    def _fill(self, layer, expert, tensors, slot):
        self.log.append((layer, expert, slot))


class Tests(unittest.TestCase):
    def test_tail_partial_and_whole_layers(self):
        self.assertEqual(mod.tail_overlaps([3, 3, 3], 4), {1: (2, 3), 2: (0, 3)})
        self.assertEqual(mod.tail_overlaps([1, 5, 3], 4), {1: (4, 5), 2: (0, 3)})
        with self.assertRaises(ValueError):
            mod.tail_overlaps([1], 2)

    def test_lease_restores_only_overwritten_residents_and_retains_policy(self):
        rt = Runtime()
        rt._borrow_arena()
        rt._borrow_arena()
        self.assertEqual(rt.log, ["drain"])
        self.assertEqual(rt.borrowed_records, {1: [(2, 2)], 2: [(0, 0), (2, 2)]})
        rt._return_arena()
        self.assertEqual(rt.log, ["drain", "drain", (1, 2, 2), (2, 0, 0), (2, 2, 2)])
        self.assertEqual(rt.borrowed_restore_experts, 3)
        self.assertEqual(rt.device_states[2]["keys"].tolist(), [0, -1, 2])
        self.assertIsNone(rt.borrowed_records)
        rt._return_arena()
        self.assertEqual(rt.borrowed_restore_runs, 1)

    def test_reader_drain_joins_all_streams_without_reacquiring_runtime_lock(self):
        rt = Runtime()
        order = []
        event = lambda name: SimpleNamespace(synchronize=lambda: order.append(name))
        rt.jobs = SimpleNamespace(join=lambda: order.append("jobs"))
        rt.miss_mailbox = event("mailbox")
        rt.metadata_stream = event("metadata")
        rt.transfer = event("transfer")
        rt.layer_computed = {0: event("compute")}
        rt.arena_last_use = event("arena")
        mod.BorrowedArenaMixin._drain_arena_readers(rt)
        self.assertEqual(
            order, ["jobs", "mailbox", "metadata", "transfer", "compute", "arena"]
        )

    def test_failed_restore_poisoned_and_lease_retained(self):
        rt = Runtime()
        rt._borrow_arena()

        def fail(*args):
            raise RuntimeError("copy failed")

        rt._fill = fail
        with self.assertRaises(RuntimeError):
            rt._return_arena()
        self.assertTrue(rt.poisoned)
        self.assertIsNotNone(rt.borrowed_records)


if __name__ == "__main__":
    unittest.main()
