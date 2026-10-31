"""Actual-checkpoint MoE ablation; evaluate only selected MLP weight arrays."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.utils import load_model

from sglang.kernels.ops.gemm import mlx_affine_rows as ops
from sglang.srt.hardware_backend.mlx.models import glm5_next as g

p = argparse.ArgumentParser()
p.add_argument("--model-dir", type=Path, required=True)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
spec = importlib.util.spec_from_file_location(
    "_routed_baseline",
    args.baseline / "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
model, _ = load_model(
    args.model_dir, lazy=True, get_model_classes=lambda config: (g.Model, g.ModelArgs)
)
model.configure_optimizations("affine_fused")
blocks = {}
for layer in model.layers:
    if isinstance(layer.mlp, g.MoE):
        bits = layer.mlp.switch_mlp.gate_proj.bits
        blocks.setdefault(bits, layer.mlp)

shared = r"""
    constexpr uint PACK = glm_pack_factor<BITS>();
    constexpr uint BYTES = glm_bytes_per_pack<BITS>();
    constexpr uint VPT = 2 * PACK;
    constexpr uint BLOCK = 32 * VPT;
    for (uint k = 0; k < K; k += BLOCK) {
        float inputs[VPT];
        float sum = glm_load_vector<T,VPT,BITS>(x+size_t(tok)*K+k+lane*VPT,inputs);
        for (uint r = 0; r < RPS; ++r) {
            size_t pos = base + r;
            size_t offset = pos * (K * BITS / 8) + (k / PACK + lane * 2) * BYTES;
            size_t meta = pos * (K / GS) + k / GS + lane / (GS / VPT);
            gate[r] += glm_qdot<VPT,BITS>((const device uchar*)gw+offset,inputs,float(gs[meta]),float(gb[meta]),sum);
            up[r] += glm_qdot<VPT,BITS>((const device uchar*)uw+offset,inputs,float(us[meta]),float(ub[meta]),sum);
        }
    }
"""
kernels = {}
for rps in (1, 2, 4):
    source = ops._ROUTED_GATE_UP
    start = source.index("    glm_qmv_rows")
    end = source.index("    for (uint i = 0;", start)
    source = source[:start] + shared + source[end:]
    source = (
        source.replace(
            "(threadgroup_position_in_grid.x * 4 + sg) * 4",
            "(threadgroup_position_in_grid.x * 4 + sg) * RPS",
        )
        .replace("gate[4]", "gate[RPS]")
        .replace("up[4]", "up[RPS]")
        .replace("i < 4", "i < RPS")
    )
    kernel = mx.fast.metal_kernel(
        name=f"delivery_shared_rps{rps}",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "indices", "limit"],
        output_names=["out"],
        source=source,
        header=ops._HEADER,
    )

    def wrapped(*, _kernel=kernel, _rps=rps, **kw):
        kw["template"] = [*kw["template"], ("RPS", _rps)]
        grid = kw["grid"]
        kw["grid"] = (grid[0] * (4 // _rps), grid[1], grid[2])
        return _kernel(**kw)

    kernels[rps] = wrapped


@contextmanager
def candidate(name):
    if name == "native":
        with patch.object(g.MoE, "__call__", old.MoE.__call__):
            yield
    elif name == "fused":
        yield
    else:
        with patch.object(
            ops, "_routed_gate_up_kernel", return_value=kernels[int(name[-1])]
        ):
            yield


names = ["native", "fused", "shared1", "shared2", "shared4"]
report = dict(device=mx.device_info(), checkpoint=args.model_dir.name, cases=[])
for bits, block in sorted(blocks.items()):
    block.prepare_for_inference()
    block.switch_mlp.prepare_for_inference()
    mx.eval(block.parameters())
    for rows in (1, 3):
        mx.random.seed(412)
        x = mx.random.normal((1, rows, model.args.hidden_size)).astype(mx.bfloat16)
        records = {}
        reference = None
        for name in names:
            with candidate(name):
                for _ in range(5):
                    mx.eval(block(x, batch_fusion=rows > 1))
                out = block(x, batch_fusion=rows > 1)
                mx.eval(out)
                if reference is None:
                    reference = out
                records[name] = dict(
                    candidate=name,
                    bits=bits,
                    rows=rows,
                    equal=mx.array_equal(out, reference).item(),
                    samples_ms=[],
                )
        for trial in range(8):
            for name in names if trial % 2 == 0 else names[::-1]:
                with candidate(name):
                    started = time.perf_counter()
                    for _ in range(4):
                        mx.eval(block(x, batch_fusion=rows > 1))
                    records[name]["samples_ms"].append(
                        (time.perf_counter() - started) * 250
                    )
        for record in records.values():
            record["median_ms"] = statistics.median(record["samples_ms"])
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
        report["peak_bytes"] = mx.get_peak_memory()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
