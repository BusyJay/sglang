"""Graph configuration and real eager-break replay protocol without CUDA."""

import json
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

from ._load_source import load_source


class TestDecodeGraphConfig(unittest.TestCase):
    def setUp(self):
        def declare(args, source, **values):
            for name, value in values.items():
                setattr(args, name, value)

        with mock.patch.dict("sys.modules", {
            "sglang.srt.arg_groups.overrides": SimpleNamespace(
                declare_resolution=declare, resolving_view=lambda args: args),
            "sglang.srt.runtime_context": SimpleNamespace(get_exec=None),
        }):
            self.config = load_source("sglang.srt.model_executor.cuda_graph_config",
                "python/sglang/srt/model_executor/cuda_graph_config.py")
            self.hook = load_source("qwen38_graph_hook_test",
                "python/sglang/srt/arg_groups/qwen38_cache_hook.py")
        patch = mock.patch.dict("sys.modules", {
            "sglang.srt.model_executor.cuda_graph_config": self.config,
        })
        patch.start()
        self.addCleanup(patch.stop)

    def args(self, backend="breakable", **values):
        args = SimpleNamespace(load_format="qwen38_nvfp4_cache",
            cuda_graph_config=self.config.CudaGraphConfig(
                decode=self.config.PhaseConfig(backend=backend),
                prefill=self.config.PhaseConfig(backend="disabled")),
            model_loader_extra_config=json.dumps({"runtime_mode": "threaded"}),
            enable_torch_compile=False)
        args.__dict__.update(values)
        return args

    def test_batch_one_defaults(self):
        args = self.args()
        self.hook.resolve_qwen38_cuda_graph(args)
        self.assertEqual(args.cuda_graph_config.decode.bs, [1])
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 1)

    def test_unsupported_backend_runtime_compile_prefill_and_size(self):
        for backend in ("full", "tc_piecewise"):
            with self.assertRaisesRegex(ValueError, "backend=breakable"):
                self.hook.resolve_qwen38_cuda_graph(self.args(backend))
        for mode in ("sync", "async", "device"):
            with self.assertRaisesRegex(ValueError, "runtime_mode=threaded"):
                self.hook.resolve_qwen38_cuda_graph(self.args(
                    model_loader_extra_config={"runtime_mode": mode}))
        with self.assertRaisesRegex(ValueError, "torch-compile"):
            self.hook.resolve_qwen38_cuda_graph(self.args(enable_torch_compile=True))
        args = self.args()
        args.cuda_graph_config.prefill.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "prefill=disabled"):
            self.hook.resolve_qwen38_cuda_graph(args)
        for field, value in (("bs", [1, 2]), ("max_bs", 2)):
            args = self.args()
            setattr(args.cuda_graph_config.decode, field, value)
            with self.assertRaisesRegex(ValueError, "only bs"):
                self.hook.resolve_qwen38_cuda_graph(args)

    def test_eager_remains_supported(self):
        self.hook.resolve_qwen38_cuda_graph(self.args("disabled",
            model_loader_extra_config={"runtime_mode": "sync"}))

    def test_mtp_keeps_request_batch_one_graph(self):
        args = self.args(speculative_algorithm="NEXTN")
        self.hook.resolve_qwen38_cuda_graph(args)
        self.assertEqual(args.cuda_graph_config.decode.bs, [1])
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 1)


class Tensor:
    def __init__(self, value):
        self.value = value

    def copy_(self, other):
        self.value = other.value


class TestEagerBreakReplay(unittest.TestCase):
    def setUp(self):
        prefix = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
        fake_torch = SimpleNamespace(
            is_tensor=lambda x: isinstance(x, Tensor),
            Stream=object,
            device=object,
        )
        with mock.patch.dict("sys.modules", {
            "torch": fake_torch,
            prefix + ".cuda_utils": SimpleNamespace(checkCudaErrors=None),
            "sglang.srt.utils": SimpleNamespace(get_device_module=None,
                is_hip=lambda: False, is_xpu=lambda: False),
        }):
            self.graph = load_source("qwen38_real_break_test",
                "python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py")

    def test_new_routes_inputs_and_errors_reach_host_on_every_replay(self):
        calls, breaks = [], []
        capture = SimpleNamespace(_barrier_fn=None,
            _end_current_segment=lambda: calls.append("end"),
            _begin_new_segment=lambda: calls.append("begin"),
            cuda_graph=SimpleNamespace(_break_fns=breaks))
        ids, hidden = Tensor(0), Tensor(10)

        @self.graph.eager_on_graph(True)
        def run(route, activation):
            calls.append((route.value, activation.value))
            if route.value < 0:
                raise RuntimeError("worker failed")
            return Tensor(route.value + activation.value)

        token = self.graph._current_capture_var.set(capture)
        try:
            with mock.patch.object(self.graph, "_weak_ref_if_tensor", side_effect=lambda x: x):
                output = run(ids, hidden)
        finally:
            self.graph._current_capture_var.reset(token)
        self.assertEqual(calls, ["end", (0, 10), "begin"])
        for route in (3, 1, 3, 0):
            ids.value, hidden.value = route, hidden.value + 1
            self.assertIs(breaks[0](), output)
            self.assertEqual(output.value, route + hidden.value)
        ids.value = -1
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            breaks[0]()
        # No capture context: same boundary is an ordinary eager call.
        ids.value = 2
        self.assertEqual(run(ids, hidden).value, 2 + hidden.value)

    def test_quant_boundary_skips_capture_transport_and_replays_fresh_remap(self):
        topk = namedtuple("TopK", "topk_weights topk_ids router_logits")
        dispatch = namedtuple("Dispatch", "hidden_states hidden_states_scale topk_output hidden_states_pre_quant")
        calls, breaks = [], []
        capture = SimpleNamespace(_barrier_fn=None,
            _end_current_segment=lambda: None, _begin_new_segment=lambda: None,
            cuda_graph=SimpleNamespace(_break_fns=breaks))
        runner = SimpleNamespace(run=lambda batch, _quant: SimpleNamespace(
            hidden_states=Tensor(batch.hidden_states.value + batch.topk_output.topk_ids.value)))

        phases = []
        def transport(_layer, ids, execute, *, execution_phase=None):
            phases.append(execution_phase)
            calls.append(ids.value)
            # Different global routes yield different slot mappings each replay.
            return execute(SimpleNamespace(num_experts=2,
                tensors={name: None for name in (
                    "w13_weight", "w2_weight", "w13_input_scale_quant",
                    "w13_blockscale_swizzled", "g1_alphas", "w2_input_scale_quant",
                    "w2_blockscale_swizzled", "g2_alphas")},
                topk_ids=Tensor(ids.value % 2)))

        with mock.patch.dict("sys.modules", {
            "torch": SimpleNamespace(bfloat16="bf16", zeros_like=lambda _: Tensor(0)),
            "sglang.srt.layers.quantization.base_config": SimpleNamespace(
                FusedMoEMethodBase=object, QuantizationConfig=object, QuantizeMethodBase=object),
            "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": self.graph,
            "sglang.srt.layers.moe.moe_runner": SimpleNamespace(MoeRunner=None),
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass": SimpleNamespace(
                FlashInferCutlassMoeQuantInfo=lambda **_: None,
                materialize_swiglu_params_for_cutlass=None),
            "sglang.srt.layers.moe.token_dispatcher.standard": SimpleNamespace(StandardDispatchOutput=dispatch),
            "sglang.srt.layers.moe.topk": SimpleNamespace(StandardTopKOutput=topk),
            "sglang.srt.layers.moe.utils": SimpleNamespace(MoeRunnerBackend=None),
        }):
            module = load_source("qwen38_quant_break_test",
                "python/sglang/srt/layers/quantization/qwen38_nvfp4_cache.py")
            method = module.Qwen38CachedNvFp4MoeMethod(None, "fixture")
            method.runtime = SimpleNamespace(run=transport, supports_execution_phase=True)
            method.runners[2] = runner
            method.swiglu_buffers = SimpleNamespace(get=lambda _: (None, None, None))
            hidden, ids = Tensor(10), Tensor(0)
            token = self.graph._current_capture_var.set(capture)
            try:
                with mock.patch.object(self.graph, "_weak_ref_if_tensor", side_effect=lambda x: x):
                    output = method._run_experts(hidden, None, ids, None, execution_phase="verify")
            finally:
                self.graph._current_capture_var.reset(token)
            self.assertEqual(calls, [])
            self.assertEqual(output.value, 0)
            for route in (3, 2, 1, 0):
                ids.value = route
                hidden.value += 1
                breaks[0]()
                self.assertEqual(output.value, hidden.value + route % 2)
            self.assertEqual(calls, [3, 2, 1, 0])
            self.assertEqual(phases, ["verify"] * 4)


class TestBreakableBackendOutputBuffer(unittest.TestCase):
    def test_verify_keeps_all_logits_and_hc_rows(self):
        import ast
        from ._load_source import REPO_ROOT

        source = ast.parse((REPO_ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "DecodeCudaGraphRunner")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("cuda_graph_output_rows", "cuda_graph_output_capacity_rows")]
        namespace = {"ForwardMode": SimpleNamespace(TARGET_VERIFY="verify")}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "output-rows", "exec"), namespace)
        runner = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False), capture_forward_mode="verify", max_num_token=4)
        runner.cuda_graph_output_rows = lambda output: namespace["cuda_graph_output_rows"](runner, output)
        for width in (2, 3, 4):
            output = SimpleNamespace(next_token_logits=SimpleNamespace(shape=(width, 248320)), hidden_states=SimpleNamespace(shape=(width, 10240)))
            self.assertEqual(runner.cuda_graph_output_rows(output), width)
            self.assertEqual(namespace["cuda_graph_output_capacity_rows"](runner, output), 4)
        runner.model_runner.is_draft_worker = True
        self.assertIsNone(runner.cuda_graph_output_rows(output))

    def test_logits_processor_output_buffer_roundtrip(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - CPU-only minimal env
            self.skipTest("PyTorch unavailable")

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )

        backend = object.__new__(BreakableCudaGraphBackend)
        logits = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden,
            customized_info={"tag": ["kept"]},
        )

        buffer = backend._alloc_full_buffer(output, 4)
        self.assertEqual(tuple(buffer.next_token_logits.shape), (4, 3))
        self.assertEqual(tuple(buffer.hidden_states.shape), (4, 4))
        self.assertEqual(buffer.customized_info, {"tag": ["kept"]})

        backend._copy_output_to_buffer(output, buffer, 2)
        stored = backend._slice_output(buffer, 2)
        self.assertTrue(torch.equal(stored.next_token_logits, logits))
        self.assertTrue(torch.equal(stored.hidden_states, hidden))
        self.assertEqual(stored.customized_info, {"tag": ["kept"]})
