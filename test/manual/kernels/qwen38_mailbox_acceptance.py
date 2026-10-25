"""Real pack -> registered mailbox -> native FC1/FC2 acceptance on SM120.

Set QWEN38_ACCEPTANCE_MANIFEST to a native-scale H2560/I640/topk10 pack
with at least two layers and >=256 experts (including the real E512 pack).
QWEN38_ACCEPTANCE_TIMELINE selects an output JSON path. Uses two layers, 256 total cache slots, and one resident
reference layer. No service throughput or KV-loan acceptance is implied.
"""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import qwen38_cache_mailbox as mailbox_module
from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
from sglang.srt.layers.quantization.qwen38_b12x_fc1_overlap import NativeFc1Stage
from sglang.srt.layers.quantization.qwen38_cached_b12x import CachedB12xW4A16
from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
    REQUIRED_TENSORS,
    Qwen38ExpertPack,
    load_manifest,
)


def mailbox_fixture_config(manifest):
    """Separate source expert geometry from the test's bounded cache pool."""
    if (manifest.hidden_size, manifest.intermediate_size, manifest.top_k) != (
        2560,
        640,
        10,
    ):
        raise ValueError("requires H2560/I640/topk10")
    if manifest.num_layers < 2 or manifest.num_experts < 256:
        raise ValueError("requires >=2 layers and >=256 source experts")
    if manifest.scale_encoding:
        raise ValueError("native scale pack required")
    # Keep each tensor's E dimension, nbytes and original file offset intact.
    selected = replace(
        manifest,
        num_layers=2,
        tensors={layer: manifest.tensors[layer] for layer in range(2)},
    )
    selected.validate_cutlass_layout()
    slots = 128
    cache_bytes = sum(
        record.nbytes // selected.num_experts * slots
        for records in selected.tensors.values()
        for name, record in records.items()
        if name in REQUIRED_TENSORS
    )
    return selected, dict(
        source_experts=selected.num_experts,
        adapter_max_experts=selected.num_experts,
        slots_per_layer=slots,
        cache_slots_total=2 * slots,
        fc1_capacity_min=slots - 7,
        fc1_capacity_max=slots + 7,
        cache_payload_bytes=cache_bytes,
        reference_layer_bytes=max(
            sum(r.nbytes for r in records.values())
            for records in selected.tensors.values()
        ),
        prefill_arena=False,
        mapped_source_bytes=selected.payload_nbytes,
    )


class RealMailboxAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest("SM120/SM121 required")
        path = os.environ.get("QWEN38_ACCEPTANCE_MANIFEST")
        if not path:
            raise unittest.SkipTest(
                "set QWEN38_ACCEPTANCE_MANIFEST to a real native pack"
            )
        manifest = load_manifest(path)
        cls.manifest, cls.fixture_config = mailbox_fixture_config(manifest)
        print("Mailbox fixture:", json.dumps(cls.fixture_config, sort_keys=True))

    def setUp(self):
        torch.manual_seed(20261025)
        self.pack = Qwen38ExpertPack(self.manifest)
        self.addCleanup(self.pack.close)
        self.runtime = Qwen38ThreadedExpertRuntime(
            self.pack,
            "cuda",
            slots_per_layer=self.fixture_config["slots_per_layer"],
            prefill_arena=self.fixture_config["prefill_arena"],
            max_pending=64,
            adaptive_cache=True,
            adaptive_cache_mode="rebalance",
            adaptive_cache_interval=1000000,
            resident_alphas=True,
            decode_early_cache=True,
        )
        self.runtime.reserve()
        self.addCleanup(self.runtime.close)
        self.adapter = CachedB12xW4A16(
            torch.device("cuda"),
            2560,
            640,
            self.fixture_config["adapter_max_experts"],
            enable_fp32=True,
            enable_fc1_overlap=True,
        )
        self.adapter.precompile_fc1(
            self.manifest.top_k,
            self.fixture_config["fc1_capacity_min"],
            self.fixture_config["fc1_capacity_max"],
        )
        self.assertIsNone(self.runtime.arena)
        self.assertFalse(self.runtime.borrowed_arena)
        self.reference = None
        self.reference_layer = None
        self.bindings = []

    def reference_for(self, layer):
        if self.reference_layer != layer:
            self.reference = None
            self.reference = {
                name: torch.frombuffer(
                    bytearray(view), dtype=getattr(torch, record.dtype)
                )
                .reshape(record.shape)
                .cuda()
                for name, record, view in self.pack.layer_tensors(layer)
            }
            self.reference_layer = layer
        return self.reference

    def run_routes(self, layer, ids, weights=None):
        x = torch.randn(ids.shape[0], 2560, device="cuda", dtype=torch.bfloat16) * 0.1
        if weights is None:
            weights = torch.full(ids.shape, 0.1, device="cuda")
        # Keep route buffers alive while graph bindings reference their pointers.
        self.bindings.append(weights)
        actual = self.runtime.run(
            layer,
            ids,
            lambda prepared: self.fail("native mailbox path was bypassed"),
            execution_phase="verify",
            native_fc1=(self.adapter, x, weights, lambda y: y),
        ).clone()
        expected = self.adapter.run(
            x, ids, weights, self.reference_for(layer), output_dtype=torch.float32
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_miss_union_rebalance_and_binding_retirement(self):
        with envs.SGLANG_OPT_QWEN38_SPLIT_FILL.override(True):
            self.runtime.miss_mailbox.refresh_targets()
            for index in range(11):
                target = [128 + index % 8, 128 - index % 8]
                self.runtime.adaptive_target = target
                self.runtime.enqueue_adaptive_snapshot()
                # This is the test's result boundary, before maintenance.
                self.runtime.synchronize()
                torch.cuda.synchronize()
                self.runtime.adaptive_after_result()
                for rows in (1, 2, 3):
                    for union in range(1, rows * 10 + 1):
                        ids = torch.arange(
                            rows * 10, device="cuda", dtype=torch.int32
                        ).reshape(rows, 10) % union + (index % 2) * (
                            self.manifest.num_experts // 2
                        )
                        # Explicitly invalidate policy behind the completed
                        # reader to make the first call all miss, then all hit.
                        self.runtime.synchronize()
                        state = self.runtime.device_states[index % 2]
                        state["lookup"].fill_(-1)
                        state["keys"].fill_(-1)
                        state["frequency"].zero_()
                        state["ages"].zero_()
                        mixed = ids.clone()
                        mixed[:, 5:] = (mixed[:, 5:] + 32) % self.manifest.num_experts
                        for routes in (ids, ids, mixed):
                            self.run_routes(index % 2, routes)
                    ids[:, -1] = -1
                    self.run_routes(index % 2, ids)
                self.assertLessEqual(len(self.adapter.fc1_stages), 8)
            self.runtime.synchronize()
            self.assertGreater(self.runtime.adaptive_rebalances, 0)
            self.assertGreater(self.adapter.fc1_stage_misses, 8)
            self.assertGreater(self.runtime.miss_mailbox.split_fill_invocations, 0)
            self.assertIsNone(self.runtime.worker_error)

    def test_original_unsplit_mailbox_regression(self):
        with envs.SGLANG_OPT_QWEN38_SPLIT_FILL.override(False):
            self.runtime.miss_mailbox.refresh_targets()
            self.assertFalse(self.runtime.miss_mailbox.split_layouts)
            for rows in (1, 2, 3):
                ids = torch.arange(rows * 10, device="cuda", dtype=torch.int32).reshape(
                    rows, 10
                )
                self.run_routes(0, ids)
                self.run_routes(0, ids)
            self.assertEqual(self.runtime.miss_mailbox.split_fill_invocations, 0)

    def test_delayed_w2_late_fc1_overlap_and_fc2_join(self):
        with envs.SGLANG_OPT_QWEN38_SPLIT_FILL.override(True):
            mailbox = self.runtime.miss_mailbox
            mailbox.refresh_targets()
            self.run_routes(
                0, torch.arange(10, device="cuda", dtype=torch.int32).reshape(1, 10)
            )
            self.runtime.synchronize()
            events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in (
                    "origin",
                    "w2_start",
                    "w2_done",
                    "late_start",
                    "late_done",
                    "fc2_start",
                )
            }
            module = mailbox_module._module()
            original_late, original_finish = (
                NativeFc1Stage.late_fc1,
                NativeFc1Stage.finish,
            )

            class DelayedModule:
                def __getattr__(self, name):
                    return getattr(module, name)

                def mailbox_fill_grid(self, plan, layout, *args):
                    if layout.data_ptr() == mailbox.split_layouts[0][1].data_ptr():
                        # Deliberately postpone W2 production after W13 ready.
                        torch.cuda._sleep(30000000)
                        events["w2_start"].record()
                        module.mailbox_fill_grid(plan, layout, *args)
                        events["w2_done"].record()
                    else:
                        module.mailbox_fill_grid(plan, layout, *args)

            def late(stage):
                events["late_start"].record()
                original_late(stage)
                events["late_done"].record()

            def finish(stage, **kwargs):
                events["fc2_start"].record()
                return original_finish(stage, **kwargs)

            routes = torch.arange(
                self.manifest.num_experts - 10,
                self.manifest.num_experts,
                device="cuda",
                dtype=torch.int32,
            ).reshape(1, 10)
            weights = torch.full(routes.shape, 0.1, device="cuda")
            # Warm this exact binding before introducing the stream delay.
            self.run_routes(0, routes, weights)
            self.runtime.synchronize()
            self.runtime.device_states[0]["lookup"].fill_(-1)
            self.runtime.device_states[0]["keys"].fill_(-1)
            events["origin"].record()
            with (
                mock.patch.object(
                    mailbox_module, "_module", return_value=DelayedModule()
                ),
                mock.patch.object(NativeFc1Stage, "late_fc1", late),
                mock.patch.object(NativeFc1Stage, "finish", finish),
            ):
                self.run_routes(0, routes, weights)
            self.runtime.synchronize()
            torch.cuda.synchronize()
            times = {
                name: events["origin"].elapsed_time(event)
                for name, event in events.items()
                if name != "origin"
            }
            self.assertLess(times["late_done"], times["w2_start"], times)
            self.assertGreaterEqual(times["fc2_start"], times["w2_done"], times)
            path = Path(
                os.environ.get(
                    "QWEN38_ACCEPTANCE_TIMELINE", "/tmp/qwen38-mailbox-timeline.json"
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    dict(
                        device=torch.cuda.get_device_name(),
                        milliseconds=times,
                        parity="bitwise",
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(path, times)


if __name__ == "__main__":
    unittest.main()
