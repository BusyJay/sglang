"""Dependency-free regressions for descriptor uploads and replay measurement."""

import ast
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT, load_source


def source_function(path, name, namespace):
    tree = ast.parse((REPO_ROOT / path).read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class PatchRevision(unittest.TestCase):
    def test_refresh_48_layers_nonblocking_views_and_address_changes(self):
        calls = []

        class Table:
            def __init__(self, rows, root=None):
                self.rows = rows
                self.root = root or self
                self.nbytes = len(rows) * 3 * 8

            def to(self, device, non_blocking):
                calls.append(("upload", device, non_blocking))
                return Table(self.rows)

            def __getitem__(self, index):
                return Table(self.rows[index], self.root)

        def tensor(rows, **kwargs):
            calls.append(("construct", kwargs))
            return Table(rows)

        split = SimpleNamespace(value=True)
        namespace = dict(
            torch=SimpleNamespace(tensor=tensor, int64="int64"),
            envs=SimpleNamespace(
                SGLANG_OPT_QWEN38_SPLIT_FILL=SimpleNamespace(get=lambda: split.value)
            ),
        )
        source_function(
            "python/sglang/srt/layers/moe/qwen38_cache_mailbox.py",
            "split_fill_names",
            namespace,
        )
        refresh = source_function(
            "python/sglang/srt/layers/moe/qwen38_cache_mailbox.py",
            "refresh_targets",
            namespace,
        )
        names = (
            "w2_weight",
            "w13_weight",
            "w2_blockscale_swizzled",
            "w13_blockscale_swizzled",
        )
        address = SimpleNamespace(value=10000)
        runtime = SimpleNamespace(
            decode_copy_names=names,
            resident_alphas=True,
            device="cuda:0",
            csf_banks=None,
            caches={
                layer: {
                    name: SimpleNamespace(
                        data_ptr=lambda l=layer, i=i: address.value + l * 100 + i
                    )
                    for i, name in enumerate(names)
                }
                for layer in range(48)
            },
            manifest=SimpleNamespace(
                scale_encoding={},
                tensors={
                    layer: {
                        name: SimpleNamespace(offset=layer * 1000 + i * 10)
                        for i, name in enumerate(names)
                    }
                    for layer in range(48)
                },
            ),
        )
        mailbox = SimpleNamespace(
            runtime=runtime,
            sizes={l: (11, 22, 33, 44) for l in range(48)},
            pack_address=100000,
        )
        for enabled in (True, False, True):
            split.value = enabled
            address.value += 10000
            calls.clear()
            refresh(mailbox, non_blocking=True)
            constructs = [c[1] for c in calls if c[0] == "construct"]
            self.assertEqual(len(constructs), 48)
            self.assertTrue(
                all(c["device"] == "cpu" and c["pin_memory"] for c in constructs)
            )
            self.assertEqual(sum(c[0] == "upload" for c in calls), 48)
            self.assertTrue(all(c[2] is True for c in calls if c[0] == "upload"))
            self.assertEqual(len(mailbox.upload_hosts), 48)
            for layer, table in mailbox.layouts.items():
                order = (1, 3, 0, 2) if enabled else (0, 1, 2, 3)
                self.assertEqual(
                    table.rows,
                    [
                        [
                            address.value + layer * 100 + i,
                            (11, 22, 33, 44)[i],
                            100000 + layer * 1000 + i * 10,
                        ]
                        for i in order
                    ],
                )
                if enabled:
                    first, second = mailbox.split_layouts[layer]
                    self.assertIs(first.root, table)
                    self.assertIs(second.root, table)
                    self.assertEqual(first.rows + second.rows, table.rows)
            if not enabled:
                self.assertEqual(mailbox.split_layouts, {})

    def mailbox_fixture(self, experts=None):
        # Use the archived official geometry and the real pack validator;
        # tensor records describe storage without allocating expert payloads.
        config = json.loads(
            (
                REPO_ROOT
                / "tools/qwen38_cache_tests/fp4_local_20261017/official-config.json"
            ).read_text()
        )["text_config"]
        common = load_source(
            "qwen38_revision_common",
            "python/sglang/srt/model_loader/qwen38_nvfp4_common.py",
        )
        for name, expected in common.QWEN38_EXPECTED.items():
            self.assertEqual(config[name], expected)
        fmt = load_source(
            "qwen38_revision_format",
            "python/sglang/srt/model_loader/qwen38_nvfp4_pack/format.py",
        )
        count = experts if experts is not None else config["num_experts"]
        hidden, width = config["hidden_size"], config["moe_intermediate_size"]
        shapes = {
            "w13_weight": ((count, 2 * width, hidden // 2), "uint8"),
            "w2_weight": ((count, hidden, width // 2), "uint8"),
            "w13_blockscale_swizzled": (
                (count, 2 * width, hidden // 16),
                "float8_e4m3fn",
            ),
            "w2_blockscale_swizzled": ((count, hidden, width // 16), "float8_e4m3fn"),
            "g1_alphas": ((count,), "float32"),
            "g2_alphas": ((count,), "float32"),
            "w13_input_scale_quant": ((1,), "float32"),
            "w2_input_scale_quant": ((1,), "float32"),
        }
        layers, offset = {}, 0
        for layer in range(config["num_hidden_layers"]):
            records = {}
            for name, (shape, dtype) in shapes.items():
                offset = (offset + 15) // 16 * 16
                size = math.prod(shape) * fmt.SUPPORTED_DTYPES[dtype]
                records[name] = fmt.TensorRecord(offset, size, shape, dtype)
                offset += size
            layers[layer] = records
        manifest = fmt.Qwen38ExpertPackManifest(
            payload_path=Path("/metadata-only/payload.bin"),
            payload_nbytes=offset,
            checkpoint_revision="official-geometry-metadata-fixture",
            sglang_commit="fixture",
            flashinfer_backend="flashinfer_cutlass",
            compute_capability="sm120",
            num_layers=config["num_hidden_layers"],
            num_experts=count,
            top_k=config["num_experts_per_tok"],
            hidden_size=hidden,
            intermediate_size=width,
            tensors=layers,
        )
        namespace = dict(replace=replace, REQUIRED_TENSORS=fmt.REQUIRED_TENSORS)
        helper = source_function(
            "test/manual/kernels/qwen38_mailbox_acceptance.py",
            "mailbox_fixture_config",
            namespace,
        )
        return manifest, helper, namespace

    def test_real_e512_and_e256_preserve_source_geometry_and_cache_budget(self):
        for experts in (512, 256):
            manifest, helper, _ = self.mailbox_fixture(experts)
            selected, config = helper(manifest)
            self.assertEqual(selected.num_experts, experts)
            self.assertEqual(selected.num_layers, 2)
            self.assertEqual(manifest.num_layers, 48)
            self.assertEqual(config["adapter_max_experts"], experts)
            self.assertEqual(config["cache_slots_total"], 256)
            self.assertFalse(config["prefill_arena"])
            self.assertEqual(
                (config["fc1_capacity_min"], config["fc1_capacity_max"]), (121, 135)
            )
            for layer in (0, 1):
                self.assertIs(selected.tensors[layer], manifest.tensors[layer])
                self.assertEqual(
                    selected.tensors[layer]["w13_weight"].shape[0], experts
                )
            self.assertEqual(
                config["reference_layer_bytes"],
                sum(r.nbytes for r in selected.tensors[0].values()),
            )
            self.assertEqual(selected.payload_nbytes, manifest.payload_nbytes)
        # Merely changing E in the manifest must fail the real storage
        # validator; all tensor records still describe the original E256.
        with self.assertRaises(ValueError):
            helper(replace(manifest, num_experts=512))
        invalid = replace(manifest, hidden_size=1280)
        with self.assertRaisesRegex(ValueError, "H2560"):
            helper(invalid)
        with self.assertRaisesRegex(ValueError, "native scale"):
            helper(replace(manifest, scale_encoding={"test": {}}))

    def test_actual_mailbox_setup_e512_uses_manifest_capacity(self):
        manifest, helper, namespace = self.mailbox_fixture()
        namespace.update(
            torch=SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True, get_device_capability=lambda: (12, 0)
                ),
                manual_seed=lambda seed: None,
                device=lambda name: name,
            ),
            os=SimpleNamespace(environ={"QWEN38_ACCEPTANCE_MANIFEST": "fixture.json"}),
            unittest=unittest,
            json=json,
            load_manifest=lambda path: manifest,
        )
        setup_class = source_function(
            "test/manual/kernels/qwen38_mailbox_acceptance.py", "setUpClass", namespace
        ).__func__
        instance = SimpleNamespace()
        setup_class(instance)
        self.assertEqual(instance.manifest.num_experts, 512)
        self.assertEqual(instance.manifest.tensors[0]["g1_alphas"].shape, (512,))
        instance.addCleanup = lambda callback: None
        instance.assertIsNone = self.assertIsNone
        instance.assertFalse = self.assertFalse
        # Dependency mocks capture the real setUp arguments, rather than
        # simulating the configuration under test.
        runtime = mock.Mock(arena=None, borrowed_arena=False)
        adapter = mock.Mock()
        constructor = mock.Mock(return_value=runtime)
        adapter_constructor = mock.Mock(return_value=adapter)
        namespace.update(
            Qwen38ExpertPack=lambda source: mock.Mock(),
            Qwen38ThreadedExpertRuntime=constructor,
            CachedB12xW4A16=adapter_constructor,
        )
        setup = source_function(
            "test/manual/kernels/qwen38_mailbox_acceptance.py", "setUp", namespace
        )
        setup(instance)
        self.assertEqual(constructor.call_args.kwargs["slots_per_layer"], 128)
        self.assertIs(constructor.call_args.kwargs["prefill_arena"], False)
        self.assertEqual(adapter_constructor.call_args.args[3], 512)
        self.assertEqual(adapter.precompile_fc1.call_args.args, (10, 121, 135))

    def test_measurement_waits_only_after_event_prewarm_and_final_repeat(self):
        replay = load_source(
            "qwen38_head_measure_under_test",
            "tools/qwen38_cache_tests/draft_head_replay.py",
        )
        log = []

        class Event:
            def __init__(self, enable_timing):
                pass

            def record(self):
                log.append("record")

            def synchronize(self):
                log.append("wait")

            def elapsed_time(self, other):
                self_test.assertEqual(log[-1], "wait")
                return 1.0

        self_test = self
        result, times = replay.measure_cuda_events(
            SimpleNamespace(cuda=SimpleNamespace(Event=Event)),
            lambda: log.append("project") or 42,
            100,
        )
        self.assertEqual(result, 42)
        self.assertEqual(times, [1.0] * 100)
        self.assertEqual(log[:201], ["record"] * 200 + ["wait"])
        self.assertEqual(log[201:], ["record", "project", "record"] * 100 + ["wait"])


if __name__ == "__main__":
    unittest.main()
