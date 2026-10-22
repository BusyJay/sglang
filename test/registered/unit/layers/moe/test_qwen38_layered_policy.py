# SPDX-License-Identifier: Apache-2.0
"""Calibration fails closed for unmeasured or ambiguous CPU partitions."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/qwen38_layered_policy.py"
)
_spec = importlib.util.spec_from_file_location("qwen38_layered_policy", _SOURCE)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


class TestLayeredPolicy(CustomTestCase):
    metadata = {"layers": 4, "experts": 16}

    def sample(self, **kw):
        sample = dict(
            tokens=2,
            top_k=3,
            active_experts=2,
            valid_routes=4,
            layer=1,
            cache_slots=4,
            candidate_route_profiles=[0x01, 0x21],
            valid_route_counts=[2, 2],
            hit_route_counts=[0, 0],
            shared_fill_overlap=False,
            cpu_prefix=1,
            baseline_us=100,
            selected_us=60,
            minimum_saving_us=10,
            minimum_saving_fraction=0.1,
            baseline_p95_us=120,
            selected_p95_us=90,
        )
        sample.update(kw)
        return sample

    def test_exact_profiles_and_capacity_change_key(self):
        samples = [
            self.sample(),
            self.sample(
                candidate_route_profiles=[0x01, 0x11],
                valid_route_counts=[3, 1],
                hit_route_counts=[1, 0],
                active_experts=3,
            ),
            self.sample(
                valid_route_counts=[3, 2],
                hit_route_counts=[1, 0],
                valid_routes=5,
                active_experts=3,
            ),
            self.sample(cache_slots=5),
            self.sample(shared_fill_overlap=True),
        ]
        rows = policy.calibration_rows(
            {"schema_version": 3, "metadata": self.metadata, "samples": samples},
            self.metadata,
        )
        self.assertEqual(len({tuple(row[:2]) for row in rows}), 5)
        self.assertEqual([row[3] for row in rows], [1, 1, 1, 1, 1])
        self.assertTrue(all(len(row) == policy.COLUMNS for row in rows))

    def test_workload_buckets_share_layers_capacity_and_token_permutations(self):
        def rows(sample):
            return policy.calibration_rows(
                dict(schema_version=4, metadata=self.metadata, samples=[sample]),
                self.metadata,
            )

        sample = self.sample(representatives=3)
        expected = rows(sample)
        self.assertEqual(expected[0][4:6], [-1, -1])
        self.assertEqual(
            rows(self.sample(layer=2, cache_slots=8, representatives=3)), expected
        )
        self.assertEqual(
            rows(self.sample(candidate_route_profiles=[0x10, 0x12], representatives=3)),
            expected,
        )
        self.assertNotEqual(
            rows(self.sample(shared_fill_overlap=True, representatives=3)), expected
        )
        # Same total routes, different CPU token work must not share a key.
        self.assertNotEqual(
            policy.workload_bucket(2, 3, 2, 4, False, [0x2, 0x2]),
            policy.workload_bucket(2, 3, 2, 4, False, [0x11, 0x11]),
        )
        with self.assertRaisesRegex(ValueError, "three measured"):
            rows(self.sample(representatives=2))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            policy.calibration_rows(
                dict(
                    schema_version=4,
                    metadata=self.metadata,
                    samples=[sample, self.sample(layer=2, representatives=3)],
                ),
                self.metadata,
            )

    def test_rejects_unknown_metadata_or_unprofitable_choice(self):
        source = {
            "schema_version": 3,
            "metadata": self.metadata,
            "samples": [self.sample()],
        }
        with self.assertRaises(ValueError):
            policy.calibration_rows(source, {"layers": 5, "experts": 16})
        source["samples"] = [self.sample(selected_us=95)]
        with self.assertRaises(ValueError):
            policy.calibration_rows(source, self.metadata)

    def test_zero_prefix_and_duplicate_key(self):
        source = {
            "schema_version": 3,
            "metadata": self.metadata,
            "samples": [self.sample(cpu_prefix=0)],
        }
        self.assertEqual(policy.calibration_rows(source, self.metadata)[0][3], 0)
        source["samples"].append(self.sample(cpu_prefix=1))
        with self.assertRaises(ValueError):
            policy.calibration_rows(source, self.metadata)

    def test_uncalibrated_branches_keep_original_gpu_dispatch(self):
        import torch

        from sglang.srt.layers.moe.qwen38_layered_hybrid import LayeredCpuHybrid
        from sglang.srt.layers.quantization.qwen38_nvfp4_cache import (
            Qwen38CachedNvFp4MoeMethod,
        )

        layered = LayeredCpuHybrid.__new__(LayeredCpuHybrid)
        layered.mode = "calibrated"
        layered.calibrated_branches = frozenset({(1, 3, 4, True)})
        bucketed = LayeredCpuHybrid.__new__(LayeredCpuHybrid)
        bucketed.mode = "calibrated"
        bucketed.calibrated_branches = frozenset({(-1, 3, -1, True)})
        self.assertTrue(bucketed.supports(2, 3, 8, True))
        self.assertFalse(bucketed.supports(2, 1, 8, True))
        self.assertFalse(bucketed.supports(2, 3, 8, False))
        for layer, tokens, slots, overlap, phase, use_cpu in (
            (1, 3, 4, True, "verify", True),
            (0, 3, 4, True, "verify", False),
            (1, 1, 4, True, "decode", False),
            (1, 3, 5, True, "verify", False),
            (1, 3, 4, False, "verify", False),
            (1, 3, 4, True, "prefill", False),
        ):
            with self.subTest(
                layer=layer, tokens=tokens, slots=slots, overlap=overlap, phase=phase
            ):
                x = torch.ones((tokens, 8), dtype=torch.bfloat16)
                gpu_output = x.clone()
                cpu_output = x * 2
                runtime = SimpleNamespace(
                    layered_hybrid=layered,
                    run=Mock(return_value=SimpleNamespace(hidden_states=gpu_output)),
                    run_layered=Mock(return_value=cpu_output),
                    slots_for_layer=Mock(return_value=slots),
                    _use_prefill_arena=Mock(return_value=False),
                )
                subject = SimpleNamespace(runtime=runtime, layer_id=layer)
                result = Qwen38CachedNvFp4MoeMethod._run_experts.__wrapped__(
                    subject,
                    x,
                    torch.ones((tokens, 3)),
                    torch.zeros((tokens, 3), dtype=torch.int32),
                    None,
                    execution_phase=phase,
                    shared_call=Mock() if overlap else None,
                    shared_input=x,
                )
                self.assertIs(result, cpu_output if use_cpu else gpu_output)
                self.assertEqual(runtime.run_layered.call_count, int(use_cpu))
                self.assertEqual(runtime.run.call_count, int(not use_cpu))

    def test_copy_stats_include_gpu_fallback_and_exclude_cpu_experts(self):
        import torch

        from sglang.srt.layers.moe.qwen38_cache_mailbox import DecodeMissMailbox

        mailbox = DecodeMissMailbox.__new__(DecodeMissMailbox)
        stats = torch.tensor([[2, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        runtime = SimpleNamespace(
            device="cuda",
            layer_computed={},
            layered_hybrid=SimpleNamespace(stats=stats),
            hits=0,
            layer_hits=[0, 0],
            misses=0,
            layer_misses=[0, 0],
            decode_device_gate_hits=0,
            transferred_bytes=0,
            decode_device_gate_copy_calls=0,
            decode_device_gate_copy_bytes=0,
            resident_alphas=True,
            decode_alpha_h2d_calls_avoided=0,
        )
        mailbox.runtime = runtime
        mailbox.hit_stats = torch.tensor([[1, 0, 2, 1], [2, 0, 3, 1]])
        mailbox.observed = [[0] * 4 for _ in range(2)]
        mailbox.observed_cpu_experts = [0, 0]
        mailbox.sizes = {0: (6, 4), 1: (12, 8)}
        with patch.object(torch.cuda, "current_stream", return_value=Mock()):
            mailbox.flush_stats()
            self.assertEqual(runtime.misses, 5)
            self.assertEqual(runtime.transferred_bytes, 70)
            self.assertEqual(runtime.decode_device_gate_copy_calls, 2)
            mailbox.flush_stats()
            self.assertEqual(runtime.transferred_bytes, 70)
            # A CPU-only miss performs neither GPU fill nor a copy call.
            mailbox.hit_stats[0, 2] += 1
            stats[0, 0] += 1
            stats[0, 1] += 1
            mailbox.flush_stats()
            self.assertEqual(runtime.misses, 6)
            self.assertEqual(runtime.transferred_bytes, 70)
            self.assertEqual(runtime.decode_device_gate_copy_calls, 2)
            # The same layer can later take the original GPU fallback.
            mailbox.hit_stats[0, 2] += 2
            mailbox.hit_stats[0, 3] += 1
            mailbox.flush_stats()
            self.assertEqual(runtime.transferred_bytes, 90)
            self.assertEqual(runtime.decode_device_gate_copy_calls, 3)
            self.assertEqual(runtime.decode_alpha_h2d_calls_avoided, 12)

    def test_library_and_execution_config_change_rejects_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "cpu.so"
            library.write_bytes(b"first binary")
            cpu = SimpleNamespace(library_path=library)
            runtime = SimpleNamespace(
                miss_mailbox=SimpleNamespace(fill_cta=8),
                cache_overlap_policy="auto",
                resident_alphas=True,
            )
            metadata = {
                **self.metadata,
                **policy.execution_metadata(cpu, runtime, 2),
            }
            source = {
                "schema_version": 3,
                "metadata": metadata,
                "samples": [self.sample()],
            }
            self.assertEqual(len(policy.calibration_rows(source, metadata)), 1)
            library.write_bytes(b"second binary")
            with self.assertRaises(ValueError):
                policy.calibration_rows(
                    source,
                    {**self.metadata, **policy.execution_metadata(cpu, runtime, 2)},
                )
            library.write_bytes(b"first binary")
            runtime.miss_mailbox.fill_cta = 4
            with self.assertRaises(ValueError):
                policy.calibration_rows(
                    source,
                    {**self.metadata, **policy.execution_metadata(cpu, runtime, 2)},
                )
            runtime.miss_mailbox.fill_cta = 8
            runtime.cache_overlap_policy = "always"
            with self.assertRaises(ValueError):
                policy.calibration_rows(
                    source,
                    {**self.metadata, **policy.execution_metadata(cpu, runtime, 2)},
                )


if __name__ == "__main__":
    unittest.main()
