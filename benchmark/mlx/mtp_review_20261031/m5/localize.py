import argparse
import json
from pathlib import Path

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.models.glm5_next import _compiled_residual
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import clone_cache

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
parser = argparse.ArgumentParser(
    description="Localize serial/batched FFN differences per layer without propagating them."
)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
p = args.model_dir
m = load_mtp_model(p)
base = m.make_cache()
mx.eval(m(mx.array([[10, 11, 12, 13] * 4]), base))
x = m.model.embed_tokens(mx.array([[31, 32]]))
x = mx.repeat(x[..., None, :], 4, axis=-2)
rows = []


def delta(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


for i, layer in enumerate(m.layers):
    c = clone_cache(base[i])
    att = []
    for j in range(2):
        norm, post, res = layer._branch_inputs(
            x[:, j : j + 1], "attn", layer.input_layernorm
        )
        att.append(layer._branch_output(layer.self_attn(norm, c), post, res))
    a = mx.concatenate(att, 1)
    serial = [
        layer._branch_inputs(v, "ffn", layer.post_attention_layernorm) for v in att
    ]
    batched = layer._branch_inputs(a, "ffn", layer.post_attention_layernorm)
    serial_norm = mx.concatenate([v[0] for v in serial], 1)
    serial_branch = mx.concatenate([layer.mlp(v[0]) for v in serial], 1)
    batch_same_input = layer.mlp(serial_norm, batch_fusion=True)
    batch_branch = layer.mlp(batched[0], batch_fusion=True)
    target = mx.concatenate(
        [layer._branch_output(layer.mlp(v[0]), v[1], v[2]) for v in serial], 1
    )
    got = layer._branch_output(batch_branch, batched[1], batched[2])
    mx.eval(target, got, batch_same_input, serial_branch)
    r = dict(
        layer=i,
        norm=delta(serial_norm, batched[0]),
        post=delta(mx.concatenate([v[1] for v in serial], 1), batched[1]),
        residual=delta(mx.concatenate([v[2] for v in serial], 1), batched[2]),
        ffn_same_input=delta(serial_branch, batch_same_input),
        output=delta(target, got),
    )
    rows.append(r)
    print(json.dumps(r), flush=True)
    x = target
args.output.write_text(json.dumps(rows, indent=2) + "\n")
