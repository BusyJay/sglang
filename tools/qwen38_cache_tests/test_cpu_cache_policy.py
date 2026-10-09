import unittest

from ._load_source import load_source

cache_module = load_source(
    "qwen38_expert_cache_under_test",
    "python/sglang/srt/layers/moe/qwen38_expert_cache.py",
)
LayerExpertCachePolicy = cache_module.LayerExpertCachePolicy
remap_topk_ids = cache_module.remap_topk_ids


class TestCachePolicy(unittest.TestCase):
    def test_hit_miss_remap_and_generation(self):
        cache = LayerExpertCachePolicy(512, 3, decay=1.0)
        mapping, misses = cache.ensure([7, 20, 91])
        self.assertEqual(misses, [7, 20, 91])
        self.assertEqual(cache.remap([[7, 20], [20, 91]]), [[0, 1], [1, 2]])
        generations = [slot.generation for slot in cache.slots]

        mapping2, misses2 = cache.ensure([20, 91])
        self.assertEqual(misses2, [])
        self.assertEqual(mapping2, {20: mapping[20], 91: mapping[91]})
        self.assertEqual([slot.generation for slot in cache.slots], generations)

    def test_lfu_then_lru_eviction(self):
        cache = LayerExpertCachePolicy(16, 3, decay=1.0)
        cache.ensure([1, 2, 3])
        cache.ensure([1, 1])
        cache.ensure([4])
        self.assertIn(1, cache.expert_to_slot)
        self.assertNotIn(2, cache.expert_to_slot)
        self.assertIn(3, cache.expert_to_slot)
        self.assertIn(4, cache.expert_to_slot)

    def test_protection_prevents_overwrite(self):
        cache = LayerExpertCachePolicy(16, 2)
        cache.ensure([1, 2])
        cache.protect([1, 2])
        with self.assertRaisesRegex(RuntimeError, "no evictable"):
            cache.ensure([3])
        cache.release([2])
        cache.ensure([3])
        self.assertIn(1, cache.expert_to_slot)
        self.assertIn(3, cache.expert_to_slot)

    def test_failed_ensure_does_not_partially_mutate_mapping(self):
        cache = LayerExpertCachePolicy(16, 3)
        cache.ensure([1, 2, 3])
        cache.protect([1, 2])
        before = dict(cache.expert_to_slot)
        generations = [slot.generation for slot in cache.slots]
        with self.assertRaisesRegex(RuntimeError, "no evictable"):
            cache.ensure([4, 5])
        self.assertEqual(cache.expert_to_slot, before)
        self.assertEqual([slot.generation for slot in cache.slots], generations)

    def test_old_hot_expert_frequency_decays(self):
        cache = LayerExpertCachePolicy(16, 2, decay=0.5)
        cache.ensure([1, 2])
        for _ in range(8):
            cache.ensure([1])
        for _ in range(8):
            cache.ensure([2])
        cache.ensure([3])
        self.assertNotIn(1, cache.expert_to_slot)
        self.assertIn(2, cache.expert_to_slot)

    def test_tensor_remap_stays_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        topk = torch.tensor([[7, 20], [20, 91]], dtype=torch.int32)
        result = remap_topk_ids(topk, {7: 4, 20: 18, 91: 2})
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.tolist(), [[4, 18], [18, 2]])


if __name__ == "__main__":
    unittest.main()
