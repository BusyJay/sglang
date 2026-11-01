# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestMlaPreparedStrides(CustomTestCase):
    def test_clone_explicit_update_and_changed_layout_recapture(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from sglang.srt.hardware_backend.mlx.models.glm5_next import (
            MLA,
            ModelArgs,
            _clone_graph_modules,
            _GraphObservation,
            _observe_graph,
        )

        args = ModelArgs(
            model_type="glm5_next",
            text_config={
                "hidden_size": 512,
                "rms_norm_eps": 1e-5,
                "attention_bias": False,
                "num_attention_heads": 2,
                "q_lora_rank": 512,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 0,
                "v_head_dim": 64,
                "index_head_dim": 128,
                "index_n_heads": 4,
                "index_topk": 32,
                "index_kpool": 4,
                "index_kpool_always_select_tail": True,
            },
        )
        attn = MLA(args)
        attn.set_dtype(mx.bfloat16)
        for parent, name in (
            (attn, "q_a_proj"),
            (attn, "kv_a_proj_with_mqa"),
            (attn, "q_b_proj"),
            (attn.indexer, "wk"),
            (attn.indexer, "weights_proj"),
            (attn.indexer, "wq_b"),
        ):
            setattr(
                parent, name, getattr(parent, name).to_quantized(bits=4, group_size=64)
            )
        owner = SimpleNamespace(_observation=None)
        key, weights, _ = _observe_graph(owner, attn)
        self.assertIn("_e_projection_sources", _GraphObservation._excluded)
        self.assertNotIn("_e_projection_strides", _GraphObservation._excluded)
        layout = attn._e_projection_strides
        clone = _clone_graph_modules(attn)

        def run(x, params):
            clone.update(params)
            return clone.project_verify_rows(x, selecting=True)

        compiled = mx.compile(run)
        x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)

        def exact(a, b):
            mx.eval(a, b)
            left = [v for _, v in tree_flatten(a) if isinstance(v, mx.array)]
            right = [v for _, v in tree_flatten(b) if isinstance(v, mx.array)]
            self.assertEqual(len(left), len(right))
            self.assertTrue(
                all(mx.array_equal(v, w).item() for v, w in zip(left, right))
            )

        exact(compiled(x, weights), attn.project_verify_rows(x, selecting=True))
        same, _, _ = _observe_graph(owner, attn)
        self.assertEqual(key, same)
        # Different physical packed/scale/bias strides and different values.
        p = attn.q_a_proj
        q = mx.quantize(
            mx.random.normal((512, 512)).astype(mx.bfloat16), group_size=64, bits=4
        )
        for name, value in zip(("weight", "scales", "biases"), q):
            storage = mx.zeros((value.shape[0] * 2, value.shape[1] * 3), value.dtype)
            storage[::2, ::3] = value
            setattr(p, name, storage[::2, ::3])
        new_key, new_weights, _ = _observe_graph(owner, attn)
        self.assertNotEqual(key, new_key)
        self.assertNotEqual(layout, attn._e_projection_strides)
        clone = _clone_graph_modules(attn)

        def new_run(y, params):
            clone.update(params)
            return clone.project_verify_rows(y, selecting=True)

        recaptured = mx.compile(new_run)
        exact(recaptured(x, new_weights), attn.project_verify_rows(x, selecting=True))
        # Dense wk retains original whole-group eligibility and separate math.
        from mlx import nn

        attn.indexer.wk = nn.Linear(512, 128, bias=False)
        attn.indexer.wk.set_dtype(mx.bfloat16)
        _observe_graph(owner, attn)
        self.assertTrue(all(v is None for v in attn._e_projection_strides[1]))


if __name__ == "__main__":
    unittest.main()
