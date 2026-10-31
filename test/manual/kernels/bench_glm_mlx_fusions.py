"""Warm MLX fusion latency; report no DRAM bandwidth or end-to-end speed claim.

Run with the normal sglang Python environment on an idle Apple GPU. KDA uses
production projection geometry; the MoE down block uses 16 experts with the
production 2048->4096 projection and eight selected experts. Fresh outputs and
advancing KDA caches prevent benchmarking a previously evaluated lazy graph.
Weights remain resident/hot; full-model opposite-order trials are separate.
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.kimi_linear import _group_expert_select

from sglang.kernels.ops.layernorm.mlx_hc import mlx_hc_collapse_norm
from sglang.kernels.ops.moe.mlx_glm import mlx_glm_down_combine, mlx_glm_router
from sglang.srt.hardware_backend.mlx.models.glm5_next import (
    KDA,
    ModelArgs,
    NativeCache,
    _hyper_connection,
)


def compare(reference, fused, iterations):
    rows = {"reference": [], "fused": []}
    for name, fn in (("reference", reference), ("fused", fused)):
        for _ in range(3):
            mx.eval(fn())
    for order in (("reference", "fused"), ("fused", "reference")):
        for _ in range(iterations):
            for name in order:
                fn = reference if name == "reference" else fused
                start = time.perf_counter()
                mx.eval(fn())
                rows[name].append(1e6 * (time.perf_counter() - start))
    medians = {name: statistics.median(values) for name, values in rows.items()}
    return {"median_us": medians, "speedup": medians["reference"] / medians["fused"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mx.random.seed(20261030)
    report = {"device": mx.device_info(), "scope": __doc__, "rows": {}}
    dim = 4096
    x = mx.random.normal((1, 1, 4, dim)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 4 * dim)) * 0.03
    base, scale = mx.random.normal((24,)), mx.array([0.5, 1.5, 0.75])
    norm = nn.RMSNorm(dim)
    norm.set_dtype(mx.bfloat16)
    norm.weight = mx.random.uniform(low=0.1, high=2, shape=(dim,)).astype(mx.bfloat16)

    def hc_reference():
        c, post, residual = _hyper_connection(x, fn, base, scale, 4, 1e-6, 1e-6, 20)
        return norm(c), post, residual

    def hc_fused():
        mix = (
            mx.fast.rms_norm(x.reshape(1, 1, -1).astype(mx.float32), None, 1e-6) @ fn.T
        )
        return mlx_hc_collapse_norm(
            x, mix, base, scale, norm.weight, eps=1e-6, norm_eps=norm.eps, iters=20
        )

    report["rows"]["hc"] = compare(hc_reference, hc_fused, args.iterations)
    config = ModelArgs(
        model_type="glm5_next",
        text_config=dict(
            hidden_size=4096,
            qk_rope_head_dim=0,
            rms_norm_eps=1e-6,
            linear_attn_config=dict(
                num_heads=64,
                head_dim=128,
                short_conv_kernel_size=4,
                gate_lower_bound=-5.0,
            ),
        ),
    )
    ref = KDA(config)
    ref.set_dtype(mx.bfloat16)
    ref.A_log = ref.A_log.astype(mx.float32)
    ref.dt_bias = ref.dt_bias.astype(mx.float32)
    fused = copy.deepcopy(ref)
    ref.prepare_for_inference()
    fused.prepare_for_inference()
    fused.fuse_decode = True
    caches = [NativeCache(4), NativeCache(4)]
    tokens = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)

    def kda_step(model, cache):
        out = model(tokens, cache)
        return out, cache.state

    mx.eval(ref.parameters(), fused.parameters())
    report["rows"]["kda"] = compare(
        lambda: kda_step(ref, caches[0]),
        lambda: kda_step(fused, caches[1]),
        args.iterations,
    )
    router = mx.random.normal((288, 4096)) * 0.03
    bias = mx.random.normal((288,)) * 0.01
    report["rows"]["router"] = compare(
        lambda: _group_expert_select(
            tokens.astype(mx.float32) @ router.T, bias, 8, 1, 1, 2.5, True, "sigmoid"
        ),
        lambda: mlx_glm_router(
            tokens, router, bias, top_k=8, scaling=2.5, renormalize=True
        ),
        args.iterations,
    )
    w, s, b = mx.quantize(
        (mx.random.normal((16, 4096, 2048)) * 0.03).astype(mx.float16),
        bits=5,
        group_size=64,
    )
    act = mx.random.normal((1, 1, 8, 2048)).astype(mx.float16)
    indices = mx.array([[[7, 3, 14, 0, 9, 1, 8, 2]]], mx.uint32)
    weights = mx.softmax(mx.random.normal((1, 1, 8)), axis=-1)
    shared = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    mx.eval(w, s, b, act, indices, weights, shared)

    def down_reference():
        routed = mx.gather_qmm(
            act[..., None, :],
            w,
            s,
            b,
            rhs_indices=indices,
            transpose=True,
            bits=5,
            group_size=64,
        ).squeeze(-2)
        routed = routed.astype(mx.bfloat16)
        return (routed * weights[..., None]).sum(-2).astype(mx.bfloat16) + shared

    report["rows"]["down_combine"] = compare(
        down_reference,
        lambda: mlx_glm_down_combine(
            act, indices, weights, w, s, b, shared, bits=5, group_size=64
        ),
        args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rows"], indent=2))


if __name__ == "__main__":
    main()
