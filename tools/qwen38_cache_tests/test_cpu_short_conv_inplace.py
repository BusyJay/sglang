import ast
import unittest
from types import SimpleNamespace
from unittest import mock

from ._load_source import REPO_ROOT


class TestShortConvInplace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch unavailable")
        cls.torch = torch
        source = ast.parse(
            (REPO_ROOT / "python/sglang/srt/models/qwen4_exp.py").read_text()
        )
        method = next(
            n
            for n in ast.walk(source)
            if isinstance(n, ast.FunctionDef) and n.name == "_short_conv"
        )
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )

        cls.namespace = {
            "torch": torch,
            "F": torch.nn.functional,
            "select_verify_intermediate_state_indices": select_verify_intermediate_state_indices,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "actual-short-conv",
                "exec",
            ),
            cls.namespace,
        )
        cls.short_conv = staticmethod(cls.namespace["_short_conv"])

    def exercise(self, verify=False, decode=False, grad=False):
        torch = self.torch
        torch.manual_seed(120)
        channels, width = 4, 1 if decode else 3
        lengths = torch.tensor([width, width if decode else 2])
        req = torch.repeat_interleave(torch.arange(2), lengths)
        offsets = torch.cat([torch.arange(n) for n in lengths])
        batch = SimpleNamespace(
            use_decode_fast_path=decode,
            lengths=lengths,
            row_width=width,
            req_indices=req,
            token_offsets=offsets,
            state_indices=torch.tensor([1, 2]),
            mode=SimpleNamespace(is_target_verify=lambda: verify),
            valid_tokens=(torch.arange(width)[None, :] < lengths[:, None]).flatten(),
        )
        model = SimpleNamespace(
            layer_id=0,
            short_conv_dilation=2,
            short_conv_state_len=4,
            conv_channels=channels,
            conv1d=SimpleNamespace(weight=torch.randn(channels, 1, 3)),
        )
        x = torch.randn(req.numel(), channels, requires_grad=grad)
        initial = torch.randn(5, channels, 4)
        expected_state = initial.clone()
        expected_intermediate = torch.zeros(2, width, channels, 4)
        state = initial.clone()
        intermediate = expected_intermediate.clone()
        pool = SimpleNamespace(
            size=2,
            short_conv_layer_cache=lambda _: state,
            short_conv_layer_intermediate_cache=lambda _: intermediate,
        )
        self.namespace["get_req_to_token_pool"] = lambda: pool
        self.namespace["_ple_track_targets"] = lambda *_: (
            (torch.tensor([3, 4]), torch.tensor([1, 1])) if not verify else None
        )
        # Run the same implementation with activation forced out-of-place.
        activation = torch.nn.functional.silu
        reference_namespace = dict(self.namespace)
        reference_namespace["F"] = SimpleNamespace(
            conv1d=torch.nn.functional.conv1d,
            silu=lambda t, **_: activation(t, inplace=False),
        )
        reference_namespace["get_req_to_token_pool"] = lambda: SimpleNamespace(
            size=2,
            short_conv_layer_cache=lambda _: expected_state,
            short_conv_layer_intermediate_cache=lambda _: expected_intermediate,
        )
        reference = type(self.short_conv)(self.short_conv.__code__, reference_namespace)
        original_x = x.detach().clone()
        observed = []

        def capture(tensor, inplace=False):
            output = activation(tensor, inplace=inplace)
            observed.append((inplace, output.data_ptr() == tensor.data_ptr()))
            return output

        with (
            torch.set_grad_enabled(grad),
            mock.patch.dict(
                "sys.modules",
                {
                    "sglang.kernels.ops.mamba.qwen4_short_conv": SimpleNamespace(
                        can_fuse_qwen4_short_conv_state=lambda *_: False,
                        fused_qwen4_short_conv_state=None,
                    ),
                },
            ),
        ):
            expected = reference(
                model, x, SimpleNamespace(req_pool_indices=torch.arange(2)), batch
            )
            with mock.patch.object(torch.nn.functional, "silu", side_effect=capture):
                actual = self.short_conv(
                    model, x, SimpleNamespace(req_pool_indices=torch.arange(2)), batch
                )
            self.assertTrue(torch.equal(actual, expected))
            if grad:
                actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected.sum(), x)[0]
                self.assertTrue(torch.allclose(actual_grad, expected_grad))
        self.assertEqual(observed, [(not grad, not grad)])
        self.assertTrue(torch.equal(x.detach(), original_x))
        self.assertTrue(torch.equal(state, expected_state))
        self.assertTrue(torch.equal(intermediate, expected_intermediate))

    def test_prefill_state_and_track_slots(self):
        self.exercise()

    def test_verify_intermediate_states(self):
        self.exercise(verify=True)

    def test_decode_state_and_track_slots(self):
        self.exercise(decode=True)

    def test_autograd_uses_original_activation_behavior(self):
        self.exercise(grad=True)
