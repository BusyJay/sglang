"""Synthetic allocator protocol tests; do not certify a GPU footprint."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from ._load_source import load_source


class TestStartupMemory(unittest.TestCase):
    def test_disabled_does_not_import_cuda_or_allocate(self):
        m = load_source(
            "qwen38_memory_probe_disabled",
            "python/sglang/srt/layers/moe/qwen38_startup_memory.py",
        )
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            "sys.modules", {"torch": None}
        ):
            self.assertIsNone(m.get_startup_memory_probe("cuda"))
            m.memory_mark(None, "disabled")

    def test_snapshot_delta_inventory_and_summary(self):
        m = load_source(
            "qwen38_memory_probe_fixture",
            "python/sglang/srt/layers/moe/qwen38_startup_memory.py",
        )
        with tempfile.TemporaryDirectory() as td:
            storage = SimpleNamespace(data_ptr=lambda: 1234, nbytes=lambda: 4096)
            tensor = SimpleNamespace(
                device=SimpleNamespace(type="cuda", __str__=lambda: "cuda:0"),
                shape=(16,),
                dtype="float32",
                untyped_storage=lambda: storage,
            )

            # SimpleNamespace str is not a device string; explicit tiny stand-in.
            class Device:
                type = "cuda"
                index = 0

                def __str__(self):
                    return "cuda:0"

            tensor.device = Device()
            model = SimpleNamespace(
                named_parameters=lambda **k: [
                    ("embed.weight", tensor),
                    ("head.weight", tensor),
                ],
                named_buffers=lambda **k: [],
            )
            segments = [
                {
                    "device": 0,
                    "address": 100,
                    "total_size": 1000,
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "size": 400,
                            "requested_size": 300,
                            "frames": [{"filename": "fixture", "line": 1}],
                        },
                        {"state": "inactive", "size": 600},
                    ],
                }
            ]
            stats = {
                "allocated_bytes.all.current": 300,
                "active_bytes.all.current": 400,
                "reserved_bytes.all.current": 1000,
                "inactive_split_bytes.all.current": 600,
            }
            cuda = SimpleNamespace(
                synchronize=mock.Mock(),
                mem_get_info=mock.Mock(return_value=(10000, 20000)),
                memory_stats=mock.Mock(return_value=stats),
                memory=SimpleNamespace(
                    _record_memory_history=mock.Mock(),
                    _snapshot=mock.Mock(return_value={"segments": segments}),
                ),
            )
            with mock.patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
                probe = m.StartupMemoryProbe(tensor.device, td)
                probe.mark("fixture.inventory", model=model)
                inventory = json.loads(
                    next(probe.root.glob("*.inventory.json")).read_text()
                )
                self.assertEqual(inventory["cuda_storage_bytes"], 4096)
                self.assertEqual(len(inventory["storages"][0]["aliases"]), 2)
                cuda.mem_get_info.return_value = (9000, 20000)
                probe.mark("fixture.change")
            events = [json.loads(v) for v in probe.events.read_text().splitlines()]
            self.assertEqual(events[-1]["delta"]["driver_free_bytes"], -1000)
            self.assertEqual(len(list(probe.root.glob("*.pickle"))), 3)
            summary = load_source(
                "startup_memory_summary",
                "tools/qwen38_cache_tests/summarize_startup_memory.py",
            ).summarize(probe.root)
            self.assertEqual(
                summary["stages"][0]["largest_mixed_segments"][0]["inactive_bytes"], 600
            )

    def test_formal_benchmark_rejects_diagnostic_environment(self):
        from .benchmark.core import subprocess_environment

        with self.assertRaisesRegex(ValueError, "diagnostics alter timing"):
            subprocess_environment({"SGLANG_QWEN38_STARTUP_MEMORY_DIR": "/tmp/diag"})

    def test_real_postprocess_hooks_surround_device_context(self):
        import ast
        from contextlib import contextmanager
        from ._load_source import REPO_ROOT

        tree = ast.parse(
            (REPO_ROOT / "python/sglang/srt/model_loader/loader.py").read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DefaultModelLoader"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "postprocess_weights"
        )
        fn.decorator_list = []
        events = []
        module = object()
        method = SimpleNamespace(
            process_weights_after_loading=lambda m: events.append(("process", m))
        )

        @contextmanager
        def context(m, device):
            events.append(("enter", m))
            yield
            events.append(("exit", m))

        namespace = {
            "QuantizedRLModelLoader": SimpleNamespace(
                is_reload_scenario=lambda _: False
            ),
            "_modules_with_quant_method": lambda m: [(module, method)],
            "device_loading_context": context,
        }
        exec(
            compile(
                ast.Module(body=[fn], type_ignores=[]),
                "<production-postprocess>",
                "exec",
            ),
            namespace,
        )
        model = SimpleNamespace(named_modules=lambda: [("model.layer", module)])
        probe_module = SimpleNamespace(
            memory_mark=lambda p, stage, **kw: events.append((stage, kw))
        )
        with mock.patch.dict(
            "sys.modules", {"sglang.srt.layers.moe.qwen38_startup_memory": probe_module}
        ):
            namespace["postprocess_weights"](model, "cuda:0", object())
        self.assertEqual(
            [e[0] for e in events],
            [
                "postprocess.0000.before",
                "enter",
                "process",
                "exit",
                "postprocess.0000.after",
            ],
        )
        self.assertEqual(events[0][1]["metadata"]["module"], "model.layer")
        events.clear()
        # Disabled path must not need named_modules or probe module imports.
        namespace["postprocess_weights"](object(), "cpu")
        self.assertEqual([e[0] for e in events], ["enter", "process", "exit"])

    def test_real_torch_cpu_inventory_alias_and_meta(self):
        import torch

        m = load_source(
            "qwen38_memory_inventory_real",
            "python/sglang/srt/layers/moe/qwen38_startup_memory.py",
        )
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(8))
        model.alias = model.weight
        model.register_buffer("buffer_view", model.weight.detach()[2:])
        model.register_buffer("meta", torch.empty(4, device="meta"))
        inventory = m.storage_inventory(model)
        self.assertEqual(inventory["cpu_storage_bytes"], 32)
        self.assertEqual(inventory["meta_tensors"], 1)
        self.assertEqual(len(inventory["storages"]), 1)
        self.assertEqual(len(inventory["storages"][0]["aliases"]), 3)

    def test_cuda_alias_uses_one_probe(self):
        m = load_source(
            "qwen38_memory_probe_device_alias",
            "python/sglang/srt/layers/moe/qwen38_startup_memory.py",
        )

        class Device:
            type = "cuda"

            def __init__(self, name, index=None):
                self.index = (
                    index
                    if index is not None
                    else (int(name.split(":")[1]) if ":" in name else None)
                )

            def __str__(self):
                return "cuda" if self.index is None else f"cuda:{self.index}"

        fake_torch = SimpleNamespace(
            device=Device, cuda=SimpleNamespace(current_device=lambda: 0)
        )
        with mock.patch.dict(
            os.environ, {"SGLANG_QWEN38_STARTUP_MEMORY_DIR": "/tmp/fixture"}
        ), mock.patch.dict("sys.modules", {"torch": fake_torch}), mock.patch.object(
            m, "StartupMemoryProbe", return_value=object()
        ) as constructor:
            self.assertIs(
                m.get_startup_memory_probe("cuda"), m.get_startup_memory_probe("cuda:0")
            )
            constructor.assert_called_once()
