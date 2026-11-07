"""Independent DFlash2 BF16 equations, cache boundaries, and loading oracles.

CPU NumPy math rounds explicitly to BF16 at the trained operation boundaries;
the attention oracle expands GQA and softmax instead of calling MLX SDPA.
"""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")


def bf16(x):
    """IEEE round-to-nearest-even BF16, evaluated entirely on CPU."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def cpu(x):
    import mlx.core as mx
    import numpy as np

    return np.asarray(x.astype(mx.float32))


def linear(x, weight):
    return bf16(x @ cpu(weight).T)


def rms(x, module):
    import numpy as np

    return bf16(
        x
        / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + module.eps)
        * cpu(module.weight)
    )


def conv_cpu(x, delta, base, side, group_size):
    """Destination-token coefficients, causal taps, rounding after each op."""
    import numpy as np

    batch, length, width = x.shape
    taps = base.shape[1]
    out = np.zeros_like(x)
    for b in range(batch):
        for t in range(length):
            for c in range(width):
                acc = np.float32(0)
                for tap in range(min(t + 1, taps)):
                    coefficient = bf16(
                        base[side, tap, c] + delta[b, t, tap, c // group_size]
                    )
                    product = bf16(coefficient * x[b, t - tap, c])
                    acc = bf16(acc + product)
                out[b, t, c] = acc
    return out


def rope_cpu(x, positions, theta):
    import numpy as np

    dim = x.shape[-1]
    freq = theta ** (-np.arange(dim // 2, dtype=np.float32) * 2 / dim)
    angle = positions[..., None, None].astype(np.float32) * freq
    left, right = np.split(x, 2, axis=-1)
    return bf16(
        np.concatenate(
            [
                left * np.cos(angle) - right * np.sin(angle),
                left * np.sin(angle) + right * np.cos(angle),
            ],
            axis=-1,
        )
    ).transpose(0, 2, 1, 3)


def config():
    return dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=16,
        num_hidden_layers=2,
        num_target_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=32,
        sliding_window=12,
        rms_norm_eps=1e-5,
        is_causal=False,
        dflash_config=dict(
            target_layer_ids=[0, 2],
            block_size=8,
            mask_token_id=31,
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=8,
            selector_top_k=4,
        ),
    )


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires MLX")
class TestDFlash2Mlx(CustomTestCase):
    def test_adaptive_readout_refreshes_and_keeps_exact_candidate_scores(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.draft_readout import (
            DraftReadout,
            DraftReadoutState,
        )
        from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import FP8Linear

        weight, scales = mx.quantize(
            mx.random.normal((128, 64)).astype(mx.bfloat16),
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        head = FP8Linear(
            weight.view(mx.uint8), scales, block_size=32, activation_mode="a16"
        )
        states = [DraftReadoutState(), DraftReadoutState()]
        states[0].observe([0, 65, 127])
        states[1].observe([12, 125])
        hidden = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
        full = head(hidden).astype(mx.float32)
        for iteration in range(9):
            readout = DraftReadout(head, states, 128, refresh_period=8)
            result = readout(hidden)
            if iteration in (0, 8):
                self.assertTrue(mx.array_equal(result.astype(mx.float32), full).item())
                readout.topk(result.astype(mx.float32), None, 16)
            else:
                logits, ids = result
                expected = mx.where(
                    ids < 128, mx.take(full, mx.minimum(ids, 127), axis=-1), -mx.inf
                )
                self.assertTrue(
                    mx.array_equal(logits.astype(mx.float32), expected).item()
                )
                values, got = readout.topk(logits.astype(mx.float32), ids, 16)
                from sglang.kernels.ops.sampling.mlx_topk import mlx_topk

                want_values, want = mlx_topk(full)
                self.assertTrue(mx.array_equal(got, want).item())
                self.assertTrue(mx.array_equal(values, want_values).item())
        self.assertEqual([s.full_ct for s in states], [2, 2])
        self.assertEqual([s.indexed_ct for s in states], [7, 7])

    def setUp(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.dflash2 import DFlash2Model

        self.mx, self.np = mx, np
        self.rng = np.random.default_rng(194)
        self.model = DFlash2Model(config())
        weights = []
        for name, value in tree_flatten(self.model.parameters()):
            data = self.rng.normal(0, 0.12, value.shape).astype(np.float32)
            if "norm.weight" in name:
                data += 1
            if "base_kernel" in name:
                data[:, 0] += 1
            weights.append((name, mx.array(data).astype(mx.bfloat16)))
        self.model.load_weights(weights)
        self.embedding = nn.Embedding(32, 16)
        self.head = nn.Linear(16, 32, bias=False)
        self.embedding.weight = self.array((32, 16), 0.3)
        self.head.weight = self.array((32, 16), 0.3)

    def array(self, shape, scale=1):
        return self.mx.array(
            self.rng.normal(0, scale, shape).astype(self.np.float32)
        ).astype(self.mx.bfloat16)

    def close(self, actual, expected, atol=0.015, rtol=0.015):
        self.np.testing.assert_allclose(cpu(actual), expected, atol=atol, rtol=rtol)

    def test_dynamic_conv_both_sides_and_bf16_rounding(self):
        """Protect input-derived output kernels and block-boundary tap isolation."""
        from sglang.srt.hardware_backend.mlx.dflash2 import GroupedDynamicConv

        mx = self.mx
        conv = GroupedDynamicConv(16, 2, 4)
        conv.base_kernel = self.array((2, 2, 16), 0.4)
        conv.kernel_projection.weight = self.array((16, 16), 0.2)
        x = self.array((2, 8, 16))
        dynamic = linear(cpu(x), conv.kernel_projection.weight).reshape(2, 8, 2, 2, 4)
        actual, output_kernel = conv.prepare(x)
        expected = conv_cpu(cpu(x), dynamic[:, :, 0], cpu(conv.base_kernel), 0, 4)
        self.close(actual, expected, atol=0, rtol=0)
        self.close(output_kernel, dynamic[:, :, 1], atol=0, rtol=0)
        transformed = self.array((2, 8, 16))
        expected = conv_cpu(
            cpu(transformed), dynamic[:, :, 1], cpu(conv.base_kernel), 1, 4
        )
        self.close(conv.finish(transformed, output_kernel), expected, atol=0, rtol=0)
        changed = mx.concatenate([x[:1] * 100, x[1:]], axis=0)
        self.close(conv.prepare(changed)[0][1:], cpu(actual[1:]), atol=0, rtol=0)

    def rounding_boundary(self, actual, expected, label):
        """Bound each native primitive by one BF16 ULP at its output scale."""
        np = self.np
        error = cpu(actual) - expected
        maximum = float(np.max(np.abs(expected)))
        # BF16 has seven fraction bits. Use the binade containing the maximum,
        # rather than an arbitrary accumulated hidden-state tolerance.
        ulp = 2 ** (np.floor(np.log2(max(maximum, np.finfo(np.float32).tiny))) - 7)
        maxabs = float(np.max(np.abs(error)))
        rms_error = float(np.sqrt(np.mean(error * error)))
        print(f"{label}: maxabs={maxabs:.8g} rms={rms_error:.8g} BF16_ULP={ulp:.8g}")
        self.assertLessEqual(maxabs, ulp)
        return cpu(actual)

    def oracle_hidden(self, states, anchors):
        """CPU assembly with independently checked native rounding boundaries.

        SDPA here gets independently assembled, unpadded, expanded GQA tensors.
        No production masks, projected inputs, or hidden outputs are reused.
        """
        np, mx, model = self.np, self.mx, self.model

        def norm(x, module):
            value = module(mx.array(x).astype(mx.bfloat16))
            return self.rounding_boundary(value, rms(x, module), "RMSNorm")

        def project(x, weight):
            value = mx.matmul(mx.array(x).astype(mx.bfloat16), weight.T)
            return self.rounding_boundary(value, linear(x, weight), "Linear")

        batch, block = len(states), model.block_size
        ids = np.full((batch, block), model.mask_token_id, dtype=np.int32)
        ids[:, 0] = anchors
        h = cpu(self.embedding.weight)[ids]
        positions = (
            np.asarray([int(s["next_position"].item()) for s in states])[:, None]
            + np.arange(block)[None]
        )
        for index, layer in enumerate(model.layers):
            x = norm(h, layer.input_layernorm)
            dynamic = project(x, layer.attention_conv.kernel_projection.weight).reshape(
                batch, block, 2, 2, 4
            )
            x = conv_cpu(
                x, dynamic[:, :, 0], cpu(layer.attention_conv.base_kernel), 0, 4
            )
            attn = layer.self_attn
            q = rope_cpu(
                norm(
                    project(x, attn.q_proj.weight).reshape(batch, block, 2, 8),
                    attn.q_norm,
                ),
                positions,
                model.rope_theta,
            )
            k = rope_cpu(
                norm(
                    project(x, attn.k_proj.weight).reshape(batch, block, 1, 8),
                    attn.k_norm,
                ),
                positions,
                model.rope_theta,
            )
            v = (
                project(x, attn.v_proj.weight)
                .reshape(batch, block, 1, 8)
                .transpose(0, 2, 1, 3)
            )
            out = []
            for row, state in enumerate(states):
                keys = np.concatenate(
                    [cpu(state["keys"][index]), k[row : row + 1]], axis=2
                )
                values = np.concatenate(
                    [cpu(state["values"][index]), v[row : row + 1]], axis=2
                )
                keys, values = np.repeat(keys, 2, axis=1), np.repeat(values, 2, axis=1)
                scores = q[row : row + 1] @ keys.swapaxes(-1, -2) * attn.scale
                scores -= scores.max(axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=-1, keepdims=True)
                value = mx.fast.scaled_dot_product_attention(
                    mx.array(q[row : row + 1]).astype(mx.bfloat16),
                    mx.array(keys).astype(mx.bfloat16),
                    mx.array(values).astype(mx.bfloat16),
                    scale=attn.scale,
                )
                out.append(self.rounding_boundary(value, bf16(probs @ values), "SDPA"))
            out = np.concatenate(out).transpose(0, 2, 1, 3).reshape(batch, block, 16)
            out = project(out, attn.o_proj.weight)
            out = conv_cpu(
                out, dynamic[:, :, 1], cpu(layer.attention_conv.base_kernel), 1, 4
            )
            h = bf16(h + out)
            x = norm(h, layer.post_attention_layernorm)
            dynamic = project(x, layer.mlp_conv.kernel_projection.weight).reshape(
                batch, block, 2, 2, 4
            )
            x = conv_cpu(x, dynamic[:, :, 0], cpu(layer.mlp_conv.base_kernel), 0, 4)
            gate, up = (
                project(x, layer.mlp.gate_proj.weight),
                project(x, layer.mlp.up_proj.weight),
            )
            import mlx.nn as nn

            activated = self.rounding_boundary(
                nn.silu(mx.array(gate).astype(mx.bfloat16)),
                bf16(gate / (1 + np.exp(-gate))),
                "SiLU",
            )
            x = project(bf16(activated * up), layer.mlp.down_proj.weight)
            x = conv_cpu(x, dynamic[:, :, 1], cpu(layer.mlp_conv.base_kernel), 1, 4)
            h = bf16(h + x)
        return norm(h, model.norm)[:, 1:]

    def hidden(self, anchors, states, count=7):
        """Observe the shared-head input without replacing model computations."""
        seen = []

        def head(x):
            seen.append(x)
            return self.head(x)

        tokens = self.model.draft(anchors, states, self.embedding, head, count)
        self.mx.eval(tokens, seen)
        return tokens, seen[0]

    def test_chunked_cache_window_and_full_noncausal_oracle(self):
        """Cache trimming must preserve absolute RoPE and every block key."""
        mx = self.mx
        features = self.array((1, 20, 32))
        positions = mx.arange(37, 57, dtype=mx.int32)
        full = self.model.consume(features, positions, self.model.make_state())
        state = self.model.make_state()
        snapshots = []
        for start, stop in ((0, 3), (3, 12), (12, 15), (15, 20)):
            snapshots.append((state, state["positions"].tolist()))
            state = self.model.consume(
                features[:, start:stop], positions[start:stop], state
            )
        self.assertEqual(state["positions"].tolist(), list(range(46, 57)))
        self.assertEqual(state["next_position"].item(), 57)
        for snapshot, before in snapshots:
            self.assertEqual(snapshot["positions"].tolist(), before)
        for i in range(2):
            self.rounding_boundary(state["keys"][i], cpu(full["keys"][i]), "chunked K")
            self.rounding_boundary(
                state["values"][i], cpu(full["values"][i]), "chunked V"
            )
        anchors = mx.array([[3]], dtype=mx.int32)
        tokens, actual = self.hidden(anchors, [state])
        self.close(actual, self.oracle_hidden([state], [3]), atol=0, rtol=0)
        short, short_hidden = self.hidden(anchors, [state], 2)
        self.assertEqual(short.tolist(), tokens[:, :2].tolist())
        self.close(short_hidden, cpu(actual[:, :2]), atol=0, rtol=0)
        projected = self.model.hidden_norm(self.model.fc(features[:, -11:]))
        fc = self.model.fc(features[:, -11:])
        self.rounding_boundary(
            fc, linear(cpu(features[:, -11:]), self.model.fc.weight), "context fc"
        )
        self.rounding_boundary(
            projected, rms(cpu(fc), self.model.hidden_norm), "context hidden_norm"
        )
        for i, layer in enumerate(self.model.layers):
            v = (
                linear(cpu(projected), layer.self_attn.v_proj.weight)
                .reshape(1, 11, 1, 8)
                .transpose(0, 2, 1, 3)
            )
            self.close(full["values"][i], v, atol=0, rtol=0)
            k = layer.self_attn.k_norm(
                layer.self_attn.k_proj(projected).reshape(1, 11, 1, 8)
            )
            self.close(
                full["keys"][i],
                rope_cpu(
                    cpu(k),
                    self.np.asarray(positions[-11:])[None],
                    self.model.rope_theta,
                ),
                atol=0,
                rtol=0,
            )

    def test_short_proposals_keep_full_keys_and_exact_prefix_outputs(self):
        """Final FFN cropping must retain complete attention KV and exact outputs."""
        mx = self.mx
        states = [
            self.model.consume(
                self.array((1, n, 32)),
                mx.arange(offset, offset + n, dtype=mx.int32),
                self.model.make_state(),
            )
            for n, offset in ((3, 9), (11, 104))
        ]
        anchors = mx.array([[5], [7]], mx.int32)
        tokens, hidden = self.hidden(anchors, states)
        native = mx.fast.scaled_dot_product_attention
        shapes = []
        ffn_rows = []
        mlp_type = type(self.model.layers[-1].mlp)
        native_mlp = mlp_type.__call__

        def record_mlp(module, x):
            ffn_rows.append(x.shape[1])
            return native_mlp(module, x)

        def record(q, k, v, **kwargs):
            shapes.append((q.shape[-2], k.shape[-2]))
            return native(q, k, v, **kwargs)

        for count in (1, 3, 5, 7):
            shapes.clear()
            ffn_rows.clear()
            with (
                patch.object(mx.fast, "scaled_dot_product_attention", record),
                patch.object(mlp_type, "__call__", record_mlp),
            ):
                short, actual = self.hidden(anchors, states, count)
            self.assertEqual(short.tolist(), tokens[:, :count].tolist())
            self.close(actual, cpu(hidden[:, :count]), atol=0, rtol=0)
            self.assertEqual(shapes[-1], (8, 19))
            self.assertTrue(all(q == 8 for q, _ in shapes[:-1]))
            self.assertEqual(ffn_rows[-1], min(8, max(6, count + 1)))
            self.assertTrue(all(n == 8 for n in ffn_rows[:-1]))

    def test_batch_variable_context_lengths_offsets_and_state_immutability(self):
        """Padding is masked; request offsets and proposal KV cannot leak."""
        mx = self.mx
        states = [
            self.model.consume(
                self.array((1, n, 32)),
                mx.arange(offset, offset + n, dtype=mx.int32),
                self.model.make_state(),
            )
            for n, offset in ((3, 9), (11, 104))
        ]
        frozen = [[cpu(k).copy() for k in s["keys"]] for s in states]
        anchors = mx.array([[5], [7]], dtype=mx.int32)
        tokens, hidden = self.hidden(anchors, states)
        self.close(hidden, self.oracle_hidden(states, [5, 7]), atol=0, rtol=0)
        for i, state in enumerate(states):
            individual, h = self.hidden(anchors[i : i + 1], [state])
            self.assertEqual(individual.tolist(), tokens[i : i + 1].tolist())
            self.close(h, cpu(hidden[i : i + 1]), atol=0, rtol=0)
            for j, k in enumerate(state["keys"]):
                self.close(k, frozen[i][j], atol=0, rtol=0)

        # Demonstrate sensitivity to the exact mask and position regressions
        # that an oracle reusing production SDPA inputs could otherwise hide.
        native = mx.fast.scaled_dot_product_attention
        captured_masks = []

        def record(q, k, v, **kwargs):
            captured_masks.append(kwargs["mask"])
            return native(q, k, v, **kwargs)

        with patch.object(mx.fast, "scaled_dot_product_attention", record):
            repeated, _ = self.hidden(anchors, states)
        self.assertEqual(repeated.tolist(), tokens.tolist())
        self.assertEqual(len(captured_masks), len(self.model.layers))
        for mask in captured_masks:
            self.assertEqual(
                mask.tolist(), [[[[False] * 8 + [True] * 11]], [[[True] * 19]]]
            )

        def causal(q, k, v, **kwargs):
            prefix = k.shape[-2] - q.shape[-2]
            allowed = (
                mx.arange(k.shape[-2])[None, :]
                <= prefix + mx.arange(q.shape[-2])[:, None]
            )
            kwargs["mask"] = kwargs["mask"] & allowed[None, None]
            return native(q, k, v, **kwargs)

        def expose_padding(q, k, v, **kwargs):
            kwargs["mask"] = mx.ones_like(kwargs["mask"])
            return native(q, k, v, **kwargs)

        for mutation in (causal, expose_padding):
            with patch.object(mx.fast, "scaled_dot_product_attention", mutation):
                _, changed = self.hidden(anchors, states)
            self.assertFalse(mx.array_equal(changed, hidden).item())
        shifted = [dict(s, next_position=s["next_position"] + 1) for s in states]
        _, changed = self.hidden(anchors, shifted)
        self.assertFalse(mx.array_equal(changed, hidden).item())

    def test_selector_cpu_bf16_scores_anchor_dependency_and_ties(self):
        """Selector dot rounds to BF16 before FP32 unary addition; ties pick first."""
        mx, np = self.mx, self.np
        selector = self.model.candidate_selector
        selector.predecessor_codebook = self.array((32, 8), 1.5)
        selector.successor_codebook = self.array((32, 8), 1.5)
        hidden = self.array((2, 7, 16))
        logits = mx.array(self.rng.normal(0, 0.5, (2, 7, 32)).astype(np.float32))
        anchors = mx.array([[3], [9]], dtype=mx.int32)
        projected = linear(cpu(hidden), selector.hidden_projection.weight)
        raw = cpu(logits)
        ids = np.argsort(-raw, axis=-1, kind="stable")[..., :4]
        pred = np.array([3, 9])
        expected = []
        real_matmul = mx.matmul
        observed = []

        def dot(a, b, *args, **kwargs):
            result = real_matmul(a, b, *args, **kwargs)
            if a.shape == (2, 1, 8) and b.shape == (2, 8, 4):
                observed.append(result)
            return result

        for slot in range(7):
            product = bf16(
                cpu(selector.predecessor_codebook)[pred] * projected[:, slot]
            )
            successor = cpu(selector.successor_codebook)[ids[:, slot]]
            edge = bf16(np.einsum("br,bkr->bk", product, successor))
            scores = np.take_along_axis(raw[:, slot], ids[:, slot], axis=-1) + edge
            pick = scores.argmax(axis=-1)
            pred = ids[np.arange(2), slot, pick]
            expected.append(pred.copy())
        with patch.object(mx, "matmul", dot):
            actual = selector.select(hidden, logits, anchors)
            mx.eval(actual, observed)
        np.testing.assert_array_equal(np.asarray(actual), np.stack(expected, axis=1))
        pred = np.array([3, 9])
        for slot, edge_actual in enumerate(observed):
            product = bf16(
                cpu(selector.predecessor_codebook)[pred] * projected[:, slot]
            )
            edge = bf16(
                np.einsum(
                    "br,bkr->bk",
                    product,
                    cpu(selector.successor_codebook)[ids[:, slot]],
                )
            )
            self.rounding_boundary(edge_actual[:, 0], edge, "selector dot")
            pred = expected[slot]
        selector.predecessor_codebook = mx.zeros((32, 8), dtype=mx.bfloat16)
        selector.successor_codebook = mx.zeros((32, 8), dtype=mx.bfloat16)
        tied = mx.zeros((2, 7, 32), dtype=mx.float32)
        self.assertEqual(selector.select(hidden, tied, anchors).tolist(), [[0] * 7] * 2)

    def test_strict_local_checkpoint_keys_shapes_dtype_and_target(self):
        """Reject partial, quantized or mismatched snapshots before drafting."""
        from mlx.utils import tree_flatten

        from sglang.srt.hardware_backend.mlx.dflash2 import load_dflash2

        mx = self.mx
        target = SimpleNamespace(
            args=SimpleNamespace(hidden_size=16, vocab_size=32, num_hidden_layers=4)
        )
        weights = dict(tree_flatten(self.model.parameters()))
        root = Path(__file__).resolve().parents[5] / "benchmark" / "mlx"
        with tempfile.TemporaryDirectory(
            prefix="dflash2_oracle_", dir=root
        ) as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config()))
            file = path / "model.safetensors"
            mx.save_safetensors(str(file), weights)
            loaded = load_dflash2(path, target)
            for name, value in tree_flatten(loaded.parameters()):
                self.close(value, cpu(weights[name]), atol=0, rtol=0)
            key = "fc.weight"
            for kind in ("missing", "extra", "shape", "dtype"):
                with self.subTest(kind=kind):
                    bad = dict(weights)
                    if kind == "missing":
                        del bad[key]
                    elif kind == "extra":
                        bad["unexpected.weight"] = mx.zeros((1,), dtype=mx.bfloat16)
                    elif kind == "shape":
                        bad[key] = mx.zeros((16, 31), dtype=mx.bfloat16)
                    else:
                        bad[key] = bad[key].astype(mx.float32)
                    mx.save_safetensors(str(file), bad)
                    with self.assertRaises(ValueError):
                        load_dflash2(path, target)
            mx.save_safetensors(str(file), weights)
            wrong = copy.deepcopy(target)
            wrong.args.num_hidden_layers = 5
            with self.assertRaisesRegex(ValueError, "depth"):
                load_dflash2(path, wrong)


if __name__ == "__main__":
    unittest.main()
