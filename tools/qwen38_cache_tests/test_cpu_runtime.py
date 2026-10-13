from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source
from .test_cpu_cache_policy import cache_module
from .test_cpu_pack_builder import builder_module, format_module
from .test_cpu_cache_capacity import capacity

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TestSynchronousRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.moe.qwen38_expert_cache": cache_module,
                "sglang.srt.model_loader.qwen38_nvfp4_pack.format": format_module,
            },
        ):
            cls.runtime_module = load_source(
                "qwen38_runtime_under_test",
                "python/sglang/srt/layers/moe/qwen38_cache_runtime.py",
            )
            cls.export_module = load_source(
                "sglang.srt.model_loader.qwen38_nvfp4_pack.export",
                "python/sglang/srt/model_loader/qwen38_nvfp4_pack/export.py",
            )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        # Tiny transformed storage; these bytes are not real NVFP4 weights.
        self.resident = []
        for layer_id in range(2):
            layer = SimpleNamespace(num_experts=4)
            for name in format_module.REQUIRED_TENSORS:
                if "alphas" in name:
                    tensor = torch.arange(1, 5, dtype=torch.float32) + layer_id * 10
                else:
                    tensor = (
                        torch.arange(32, dtype=torch.uint8).reshape(4, 8)
                        + layer_id * 32
                    )
                setattr(layer, name, tensor)
            layer.w13_input_scale_quant = torch.tensor(2.0)
            layer.w2_input_scale_quant = torch.tensor(3.0)
            self.export_module.export_runtime_layer(
                layer, source / f"layer-{layer_id:03d}", synthetic=True
            )
            self.resident.append(layer)
        manifest = builder_module.build_pack(
            source,
            self.root / "payload.bin",
            self.root / "manifest.json",
            checkpoint_revision="fixture",
            sglang_commit="fixture",
            num_layers=2,
            num_experts=4,
            top_k=2,
            hidden_size=16,
            intermediate_size=8,
            alignment=16,
        )
        self.pack = format_module.Qwen38ExpertPack(manifest)
        self.runtime = self.runtime_module.Qwen38SynchronousExpertRuntime(
            self.pack, "cpu", slots_per_layer=2
        )

    def tearDown(self):
        self.pack.close()
        self.temp.cleanup()

    def test_decode_hit_eviction_and_fixed_addresses(self):
        def check(prepared):
            for name in format_module.REQUIRED_TENSORS:
                self.assertTrue(
                    torch.equal(
                        prepared.tensors[name][prepared.topk_ids[0].long()],
                        getattr(self.resident[0], name)[ids[0].long()],
                    )
                )
            return prepared.tensors["w13_weight"].data_ptr()

        ids = torch.tensor([[0, 3]], dtype=torch.int32)
        pointer = self.runtime.run(0, ids, check)
        transferred = self.runtime.transferred_bytes
        self.assertEqual(self.runtime.run(0, ids, check), pointer)
        self.assertEqual(self.runtime.transferred_bytes, transferred)
        ids = torch.tensor([[2, 3]], dtype=torch.int32)
        self.assertEqual(self.runtime.run(0, ids, check), pointer)
        self.assertEqual(self.runtime.misses, 3)
        self.assertEqual(self.runtime.hits, 3)

    def test_prefill_arena_cross_layer_reuse_and_weighted_route_parity(self):
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weights = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        pointer = None
        for layer_id in [0, 1, 0]:

            def execute(prepared):
                self.assertTrue(prepared.arena)
                self.assertTrue(torch.equal(prepared.topk_ids, ids))
                for name in format_module.REQUIRED_TENSORS:
                    self.assertTrue(
                        torch.equal(
                            prepared.tensors[name],
                            getattr(self.resident[layer_id], name),
                        )
                    )
                self.assertEqual(prepared.tensors["w2_input_scale_quant"].item(), 3.0)
                result = (
                    prepared.tensors["g1_alphas"][prepared.topk_ids] * weights
                ).sum(1)
                expected = (self.resident[layer_id].g1_alphas[ids] * weights).sum(1)
                torch.testing.assert_close(result, expected)
                return prepared.tensors["w13_weight"].data_ptr()

            current = self.runtime.run(layer_id, ids, execute)
            if pointer is not None:
                self.assertEqual(current, pointer)
            pointer = current
        self.assertEqual(self.runtime.arena_loads, 3)
        self.runtime.run(0, ids, lambda _: None)
        self.assertEqual(self.runtime.arena_loads, 3)

    def test_stats_snapshot_tracks_cache_and_arena_activity(self):
        cache_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        arena_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

        self.runtime.run(0, cache_ids, lambda prepared: prepared.topk_ids.clone())
        self.runtime.run(0, cache_ids, lambda prepared: prepared.topk_ids.clone())
        self.runtime.run(0, arena_ids, lambda prepared: prepared.topk_ids.clone())

        stats = self.runtime.stats_snapshot()
        self.assertEqual(stats["run_invocations"], 3)
        self.assertEqual(stats["cache_invocations"], 2)
        self.assertEqual(stats["arena_invocations"], 1)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["arena_loads"], 1)
        self.assertAlmostEqual(stats["cache_hit_rate"], 0.5)

    def test_heterogeneous_layer_capacities_and_memory_budget(self):
        runtime = self.runtime_module.Qwen38SynchronousExpertRuntime(
            self.pack, "cpu", slots_per_layer=2, layer_cache_slots=[1, 3]
        )
        runtime.reserve()
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        first = runtime.run(0, ids, lambda prepared: prepared)
        second = runtime.run(1, ids, lambda prepared: prepared)
        self.assertTrue(first.arena)
        self.assertFalse(second.arena)
        self.assertEqual(second.num_experts, 3)
        self.assertEqual(runtime.caches[0]["w13_weight"].shape[0], 1)
        self.assertEqual(runtime.caches[1]["w13_weight"].shape[0], 3)
        self.assertEqual(runtime.memory_budget()["decode_cache_bytes"], self.runtime.memory_budget()["decode_cache_bytes"])

    def test_bounded_route_trace_records_actual_layer_and_routes(self):
        path = self.root / "trace.jsonl"
        runtime = self.runtime_module.Qwen38SynchronousExpertRuntime(
            self.pack, "cpu", slots_per_layer=2,
            route_trace_path=str(path), route_trace_limit=1,
        )
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        for layer in (1, 0):
            runtime.run(layer, ids, lambda prepared: None)
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],
                         [{"layer": 1, "routes": [[0, 1]]}])

    def test_padding_and_invalid_routes(self):
        ids = torch.tensor([[-1, 2]], dtype=torch.int32)
        mapped = self.runtime.run(0, ids, lambda p: p.topk_ids)
        self.assertEqual(mapped.tolist(), [[-1, 0]])
        for ids in [
            torch.tensor([[-2, 0]]),
            torch.tensor([[4, 0]]),
            torch.tensor([[0.0, 1.0]]),
        ]:
            with self.assertRaises(ValueError):
                self.runtime.run(0, ids, lambda _: None)

    def test_disabled_arena_rejects_overflow(self):
        self.runtime.prefill_arena = False
        with self.assertRaisesRegex(ValueError, "exceeds cache capacity"):
            self.runtime.run(0, torch.tensor([[0, 1], [2, 3]]), lambda _: None)
        self.assertEqual(self.runtime.caches, {})

    def test_failed_transfer_cannot_become_cache_hit(self):
        ids = torch.tensor([[0, 1]])
        with mock.patch.object(
            self.runtime, "_fill", side_effect=OSError("copy failed")
        ):
            with self.assertRaises(OSError):
                self.runtime.run(0, ids, lambda _: None)
        self.assertEqual(self.runtime.policies[0].expert_to_slot, {})
        self.runtime.run(0, ids, lambda _: None)
        self.assertEqual(self.runtime.misses, 2)

    def test_execution_failure_releases_serialized_scope(self):
        ids = torch.tensor([[0, 1]])

        def fail(_):
            raise RuntimeError("kernel failed")

        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            self.runtime.run(0, ids, fail)
        self.runtime.run(0, ids, lambda _: None)

    def test_memory_budget_excludes_full_pinned_copy(self):
        budget = self.runtime.memory_budget()
        self.assertEqual(budget["pinned_staging_bytes"], 0)
        self.assertEqual(budget["decode_cache_bytes"], 160)
        self.assertEqual(budget["prefill_arena_bytes"], 160)

    def test_reserve_precedes_kv_allocation_and_preserves_addresses(self):
        self.runtime.reserve()
        self.assertEqual(set(self.runtime.caches), {0, 1})
        pointer = self.runtime.caches[0]["w13_weight"].data_ptr()
        arena_pointer = self.runtime.arena["w13_weight"].data_ptr()
        self.runtime.reserve()
        self.assertEqual(self.runtime.caches[0]["w13_weight"].data_ptr(), pointer)
        self.assertEqual(self.runtime.arena["w13_weight"].data_ptr(), arena_pointer)
        self.runtime.run(0, torch.tensor([[0, 1]]), lambda p: None)
        self.assertEqual(self.runtime.caches[0]["w13_weight"].data_ptr(), pointer)

    def test_export_requires_real_sm120_unless_synthetic(self):
        with self.assertRaisesRegex(ValueError, "SM120"):
            self.export_module.export_runtime_layer(self.resident[0], self.root / "bad")
        self.assertFalse((self.root / "bad").exists())


class TestPublicMetadata(unittest.TestCase):
    def test_model_constraints_and_quantization(self):
        from .test_cpu_loader_filter import common_module

        config = json.loads(
            (REPO_ROOT / "flash-next-plan/model-metadata/config.json").read_text()
        )
        model = SimpleNamespace(**config)
        model.text_config = SimpleNamespace(**config["text_config"])
        self.assertEqual(common_module.validate_qwen38_model_config(model), [])
        self.assertEqual(config["quantization_config"]["quant_algo"], "MIXED_PRECISION")
        quantized = config["quantization_config"]["quantized_layers"]
        for layer in range(48):
            key = f"model.language_model.layers.{layer}.mlp.experts"
            self.assertEqual(quantized[key]["quant_algo"], "NVFP4")
        self.assertTrue(
            common_module.should_skip_checkpoint_tensor(
                "mtp.layers.0.mlp.experts.0.gate_proj.weight", language_model_only=True
            )
        )
