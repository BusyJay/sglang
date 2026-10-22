# SPDX-License-Identifier: Apache-2.0
"""A CPU bypass leaves exactly the filtered per-layer LFU state on GPU."""

import mmap
import unittest

import torch

from sglang.kernels.ops.moe.qwen38_expert_cache import _module
from sglang.srt.layers.moe.qwen38_cache_buffers import AdaptiveMetadataBuffers
from sglang.srt.layers.moe.qwen38_layered_policy import calibration_rows
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestLayeredLfu(CustomTestCase):
    def test_cpu_only_miss_does_not_count_gpu_copy(self):
        module = _module()
        buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
        state = buffers.reset(0, 4)
        buffers.restore(0, state, [(0, 4.0, 8)])
        state["plan"].zero_()
        ids = torch.tensor([[0, 2]], dtype=torch.int32, device="cuda")
        remapped = torch.empty_like(ids)
        weights = torch.ones_like(ids, dtype=torch.float32)
        desc = torch.zeros(2, dtype=torch.int32, device="cuda")
        stats = torch.zeros(5, dtype=torch.int64, device="cuda")
        hits = torch.zeros(4, dtype=torch.int64, device="cuda")
        mapping = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
        host = torch.frombuffer(mapping, dtype=torch.uint8)
        address = module.mailbox_register(host)
        try:
            module.decode_plan_layered(
                ids,
                remapped,
                *(
                    state[k]
                    for k in (
                        "counts",
                        "lookup",
                        "keys",
                        "frequency",
                        "ages",
                        "generation",
                        "clock",
                        "plan",
                    )
                ),
                0.95,
                0,
                0.7,
                hits.data_ptr(),
                0,
                2,
                torch.ones((1, 32), dtype=torch.bfloat16, device="cuda"),
                weights,
                torch.empty((0, 47), dtype=torch.int32, device="cuda"),
                torch.empty_like(weights),
                torch.empty_like(ids),
                desc,
                stats,
                address,
                0,
                1,
                1,
                1,
                0,
            )
            self.assertEqual(hits.tolist(), [1, 0, 1, 0])
            self.assertEqual(stats.tolist(), [1, 1, 0, 1, 0])
            self.assertEqual(state["plan"][1].item(), 0)
        finally:
            torch.cuda.synchronize()
            module.mailbox_unregister(host)
            del host
            mapping.close()

    def test_calibrated_lookup_and_b12x_mask(self):
        """Synthetic costs exercise both schemas, never performance acceptance."""
        module = _module()
        buffers = AdaptiveMetadataBuffers(1, 8, 5, torch.device("cuda"))
        sample = dict(
            tokens=2,
            top_k=3,
            candidate_route_profiles=[2, 16],
            valid_route_counts=[3, 2],
            hit_route_counts=[1, 1],
            active_experts=4,
            valid_routes=5,
            cpu_prefix=1,
            layer=0,
            cache_slots=4,
            shared_fill_overlap=False,
            baseline_us=100,
            selected_us=50,
            minimum_saving_us=1,
            minimum_saving_fraction=0.05,
            selected_p95_us=60,
            baseline_p95_us=110,
        )
        metadata = dict(layers=2, experts=8)
        sample["representatives"] = 3
        for schema, overlap, layer, slots, expected_cpu in (
            (3, 0, 0, 4, 1),
            (3, 1, 0, 4, 0),
            (3, 0, 1, 5, 0),
            (4, 0, 1, 5, 1),
            (4, 1, 1, 5, 0),
        ):
            rows = calibration_rows(
                dict(schema_version=schema, metadata=metadata, samples=[sample]),
                metadata,
            )
            with self.subTest(schema=schema, overlap=overlap, layer=layer, slots=slots):
                state = buffers.reset(0, slots)
                buffers.restore(0, state, [(0, 4.0, 8), (1, 2.0, 9)])
                state["plan"].zero_()
                ids = torch.tensor(
                    [[0, 2, 2], [1, 3, -1]], dtype=torch.int32, device="cuda"
                )
                remapped = torch.empty_like(ids)
                weights = torch.tensor(
                    [[0.5, 0.2, 0.3], [0.6, 0.4, 0.0]], device="cuda"
                )
                gpu_weights = torch.empty_like(weights)
                cpu_ids = torch.empty_like(ids)
                desc = torch.zeros(2, dtype=torch.int32, device="cuda")
                stats = torch.zeros(5, dtype=torch.int64, device="cuda")
                table = torch.tensor(rows, dtype=torch.int32, device="cuda")
                x = torch.ones((2, 32), dtype=torch.bfloat16, device="cuda")
                mapping = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
                host = torch.frombuffer(mapping, dtype=torch.uint8)
                address = module.mailbox_register(host)
                try:
                    module.decode_plan_layered(
                        ids.view(1, -1),
                        remapped.view(1, -1),
                        *(
                            state[k]
                            for k in (
                                "counts",
                                "lookup",
                                "keys",
                                "frequency",
                                "ages",
                                "generation",
                                "clock",
                                "plan",
                            )
                        ),
                        0.95,
                        0,
                        0.7,
                        0,
                        0,
                        2,
                        x,
                        weights,
                        table,
                        gpu_weights,
                        cpu_ids,
                        desc,
                        stats,
                        address,
                        layer,
                        1,
                        -1,
                        1,
                        overlap,
                    )
                    self.assertEqual(desc.tolist(), [expected_cpu, 1])
                    if expected_cpu:
                        self.assertEqual(
                            cpu_ids.tolist(),
                            (
                                [[-1, 2, 2], [-1, -1, -1]]
                                if schema == 3
                                else [[-1, -1, -1], [-1, 3, -1]]
                            ),
                        )
                        self.assertEqual(
                            state["lookup"][2 if schema == 3 else 3].item(), -1
                        )
                        self.assertEqual(stats[:3].tolist(), [2, 1, 1])
                        self.assertTrue(torch.all(remapped >= 0).item())
                        torch.testing.assert_close(
                            gpu_weights,
                            torch.tensor(
                                (
                                    [[0.5, 0.0, 0.0], [0.6, 0.4, 0.0]]
                                    if schema == 3
                                    else [[0.5, 0.2, 0.3], [0.6, 0.0, 0.0]]
                                ),
                                device="cuda",
                            ),
                        )
                    else:
                        self.assertEqual(cpu_ids.tolist(), [[-1] * 3] * 2)
                        torch.testing.assert_close(gpu_weights, weights)
                finally:
                    torch.cuda.synchronize()
                    module.mailbox_unregister(host)
                    del host
                    mapping.close()

    def test_filtered_reference_and_zero_prefix(self):
        module = _module()
        actual_buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
        reference_buffers = AdaptiveMetadataBuffers(1, 8, 4, torch.device("cuda"))
        for ids, filtered, forced, cpu_count in (
            ([[0, 2, 2], [1, 3, -1]], [[0, -1, -1], [1, 3, -1]], 1, 1),
            ([[0, 2, 2], [1, 3, -1]], [[0, 2, 2], [1, 3, -1]], 0, 0),
            (
                [[0, 2, -1], [1, 3, -1], [0, 2, 2]],
                [[0, 2, -1], [1, -1, -1], [0, 2, 2]],
                1,
                1,
            ),
            (
                [[0, 2, -1], [1, 3, -1], [0, 2, 2]],
                [[0, 2, -1], [1, 3, -1], [0, 2, 2]],
                0,
                0,
            ),
        ):
            with self.subTest(forced=forced):
                actual = actual_buffers.reset(0, 4)
                expected = reference_buffers.reset(0, 4)
                for buffers, state in (
                    (actual_buffers, actual),
                    (reference_buffers, expected),
                ):
                    buffers.restore(0, state, [(0, 4.0, 8), (1, 2.0, 9)])
                    state["plan"].zero_()
                routes = torch.tensor(ids, dtype=torch.int32, device="cuda")
                remapped = torch.empty_like(routes)
                gpu_weights = torch.empty_like(routes, dtype=torch.float32)
                cpu_ids = torch.empty_like(routes)
                desc = torch.zeros(2, dtype=torch.int32, device="cuda")
                stats = torch.zeros(5, dtype=torch.int64, device="cuda")
                table = torch.empty((0, 47), dtype=torch.int32, device="cuda")
                x = torch.ones((len(ids), 32), dtype=torch.bfloat16, device="cuda")
                weights = torch.ones_like(routes, dtype=torch.float32)
                mapping = mmap.mmap(-1, mmap.PAGESIZE, flags=mmap.MAP_SHARED)
                host = torch.frombuffer(mapping, dtype=torch.uint8)
                address = module.mailbox_register(host)
                try:
                    module.decode_plan_layered(
                        routes.view(1, -1),
                        remapped.view(1, -1),
                        *(
                            actual[k]
                            for k in (
                                "counts",
                                "lookup",
                                "keys",
                                "frequency",
                                "ages",
                                "generation",
                                "clock",
                                "plan",
                            )
                        ),
                        0.95,
                        0,
                        0.7,
                        0,
                        0,
                        2,
                        x,
                        weights,
                        table,
                        gpu_weights,
                        cpu_ids,
                        desc,
                        stats,
                        address,
                        0,
                        1,
                        forced,
                        0,
                        0,
                    )
                    reference = torch.tensor(filtered, dtype=torch.int32, device="cuda")
                    module.decode_plan_parallel(
                        reference.view(1, -1),
                        torch.empty_like(reference).view(1, -1),
                        *(
                            expected[k]
                            for k in (
                                "counts",
                                "lookup",
                                "keys",
                                "frequency",
                                "ages",
                                "generation",
                                "clock",
                                "plan",
                            )
                        ),
                        0.95,
                        0,
                        0.7,
                        0,
                        0,
                        2,
                    )
                    for field in (
                        "counts",
                        "lookup",
                        "keys",
                        "frequency",
                        "ages",
                        "generation",
                        "clock",
                        "plan",
                    ):
                        torch.testing.assert_close(
                            actual[field], expected[field], rtol=0, atol=0
                        )
                    self.assertEqual(desc[0].item(), cpu_count)
                    if cpu_count:
                        selected_expert = 2 if len(ids) == 2 else 3
                        self.assertEqual(
                            cpu_ids.tolist(),
                            [
                                [
                                    expert if expert == selected_expert else -1
                                    for expert in row
                                ]
                                for row in ids
                            ],
                        )
                        self.assertEqual(stats[:3].tolist(), [2, 1, 1])
                    else:
                        self.assertEqual(cpu_ids.tolist(), [[-1] * 3 for _ in ids])
                finally:
                    torch.cuda.synchronize()
                    module.mailbox_unregister(host)
                    del host
                    mapping.close()


if __name__ == "__main__":
    unittest.main()
