"""Offload behavior at main's new FFN boundary (no checkpoint/GPU)."""

import ast
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest import mock
from ._load_source import REPO_ROOT


class TestMainMerge(unittest.TestCase):
    def test_ffn_phase_overlap_and_boundary_finish(self):
        source = ast.parse(
            (REPO_ROOT / "python/sglang/srt/models/qwen4_exp.py").read_text()
        )
        fn = next(
            n
            for n in ast.walk(source)
            if isinstance(n, ast.FunctionDef) and n.name == "_run_ffn_stage"
        )
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        events, active = [], {}

        @contextmanager
        def scoped(**kwargs):
            active.update(kwargs)
            try:
                yield
            finally:
                active.clear()

        @contextmanager
        def boundary_exit(batch):
            yield SimpleNamespace(
                defer_moe_finalize=False,
                finish=lambda x: events.append(("finish", x)) or x,
            )

        class Moe:
            def __call__(self, x, batch, **kwargs):
                events.append(("mlp", active["qwen38_execution_phase"]))
                return x + 1

        mlp = Moe()
        method = SimpleNamespace(
            can_overlap_shared=lambda block, x, phase: phase != "prefill",
            run_shared_overlap=lambda block, x, phase: events.append(
                ("overlap", active["qwen38_execution_phase"])
            )
            or x + 2,
        )
        mlp.experts = SimpleNamespace(quant_method=method)
        model = SimpleNamespace(
            mlp=mlp,
            _decode_moe_comm=None,
            ffn_boundary=SimpleNamespace(
                prepare=lambda x, batch: x, exit=boundary_exit
            ),
        )
        namespace = dict(
            torch=SimpleNamespace(Tensor=object),
            ForwardBatch=object,
            Qwen2MoeSparseMoeBlock=Moe,
            get_forward=lambda: SimpleNamespace(scoped=scoped),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                "<merged-ffn>",
                "exec",
            ),
            namespace,
        )
        for phase in ("prefill", "decode", "verify"):
            events.clear()
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_target_verify=lambda: phase == "verify",
                    is_extend_without_speculative=lambda: phase == "prefill",
                )
            )
            result = namespace["_run_ffn_stage"](model, 10, batch)
            self.assertEqual(
                events,
                [
                    ("mlp" if phase == "prefill" else "overlap", phase),
                    ("finish", result),
                ],
            )
            self.assertEqual(result, 11 if phase == "prefill" else 12)
            self.assertEqual(active, {})

    def test_shared_overlap_leaves_reduction_to_new_boundary(self):
        from sglang.srt.layers.moe import qwen38_cache_overlap as overlap
        import torch

        block = SimpleNamespace(reduce_results=False, shared_expert_gate=None)
        block._forward_shared_experts = lambda x, apply_gate: x * 2

        def routed(x):
            call, original = overlap.current_fill_overlap()
            call.run(original)
            return x * 3

        block._forward_router_experts = routed
        with mock.patch(
            "sglang.srt.models.qwen2_moe.reduce_moe_output",
            side_effect=AssertionError("double reduction"),
        ):
            torch.testing.assert_close(
                overlap.run_shared_overlap(block, torch.ones(1, 2), "verify"),
                torch.full((1, 2), 5.0),
            )
