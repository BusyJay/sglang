"""Exercise the actual mailbox publication method without CUDA or torch."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Ready:
    def __init__(self):
        self.values = {}

    def __setitem__(self, key, value):
        self.values[key] = value


class TestMailboxPublication(unittest.TestCase):
    def test_single_miss_publishes_without_payload_access(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/layers/moe/qwen38_cache_mailbox.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DecodeMissMailbox"
        )
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("stage", "_record_misses")
        ]
        module = ast.Module(body=[cls], type_ignores=[])
        namespace = {}
        exec(compile(module, str(source), "exec"), namespace)
        mailbox = namespace["DecodeMissMailbox"]()
        r = SimpleNamespace(
            manifest=SimpleNamespace(top_k=10),
            transferred_bytes=0,
            decode_device_gate_copy_calls=0,
            decode_device_gate_copy_bytes=0,
            resident_alphas=True,
            decode_alpha_h2d_calls_avoided=0,
            misses=0,
            hits=0,
            layer_misses=[0],
            layer_hits=[0],
        )
        mailbox.runtime, mailbox.sizes, mailbox.ready = r, {0: (16, 32)}, Ready()
        # No payload or pack attribute: any attempted staging would fail.
        mailbox.stage(0, [0, 1, 3, 0, 7, 1], 1, 5)
        self.assertEqual(mailbox.ready.values[0], 56)
        self.assertEqual((r.misses, r.hits), (1, 2))
        self.assertEqual(
            (r.transferred_bytes, r.decode_device_gate_copy_bytes), (48, 48)
        )
        self.assertEqual(r.decode_alpha_h2d_calls_avoided, 2)

    def test_multiple_misses_also_publish_without_payload_access(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/layers/moe/qwen38_cache_mailbox.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DecodeMissMailbox"
        )
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("stage", "_record_misses")
        ]
        namespace = {}
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        mailbox = namespace["DecodeMissMailbox"]()
        mailbox.runtime = SimpleNamespace(
            manifest=SimpleNamespace(top_k=10),
            transferred_bytes=0,
            decode_device_gate_copy_calls=0,
            decode_device_gate_copy_bytes=0,
            resident_alphas=False,
            misses=0,
            hits=0,
            layer_misses=[0],
            layer_hits=[0],
        )
        mailbox.sizes, mailbox.ready = {0: (16, 32)}, Ready()
        mailbox.stage(0, [0, 2, 3, 0, 1, 0, 7, 1], 2, 6)
        self.assertEqual(mailbox.ready.values[0], 68)
        self.assertEqual(mailbox.runtime.transferred_bytes, 96)
