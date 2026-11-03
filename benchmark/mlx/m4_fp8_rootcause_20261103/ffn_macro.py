"""Eight distinct actual-shape shared FFNs, raw300-step ABBA and source hashes.

Synthetic packed weights and activations, original block-FP8 execution recipe.
This is an operator-chain macrobenchmark, not GLM or HTTP throughput.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from common import OUT, ROOT, baseline, direct, paired, load

repeated = load("repeated_quant", OUT / "repeated_quant.py")
import sys

sys.modules["fp8_quant_fused"] = repeated
pair_module = load("pair_hoisted", ROOT / "benchmark/mlx/throughput_20261103/quant_hoisted.py")
parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=4)
parser.add_argument("--layers", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--capture", action="store_true")
parser.add_argument("--capture-mode", default="baseline")
parser.add_argument("--modes", default="direct,shared,pair,repeated")
parser.add_argument("--compiled-group", type=int, default=0)
parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
args = parser.parse_args()
mx.set_cache_limit(256 * 2**20)
mx.set_memory_limit(3 * 2**30)
mx.random.seed(422)
weights = []
for _ in range(args.layers):
    banks = []
    for n, k in ((2048, 4096), (2048, 4096), (4096, 2048)):
        w = mx.random.randint(32, 60, (n, k), dtype=mx.uint8)
        # Symmetric signs and a small per-block factor keep repeated residuals
        # finite. These are synthetic weights, not a substitute checkpoint.
        w = w | (mx.random.randint(0, 2, w.shape, dtype=mx.uint8) * 128)
        s = mx.full(((n + 127) // 128, k // 128), 0.015 / (k**0.5), mx.float32)
        mx.eval(w, s)
        banks.append(SimpleNamespace(weight=w, scales=s))
    weights.append(banks)
x = mx.random.normal((args.count, 4096)).astype(
    mx.float32 if args.dtype == "fp32" else mx.bfloat16
)
mx.eval(x)


def proj(x, p, quant):
    return baseline.mlx_fp8_rows(x, p.weight, p.scales, block_size=128, quantized_input=quant)


def layer(value, banks, mode):
    gate, up, down = banks
    if mode == "repeated":
        g = repeated.project(value, gate.weight, gate.scales)
        u = repeated.project(value, up.weight, up.scales)
    elif mode in ("pair", "pair1"):
        quant = direct.mlx_fp8_dynamic_32(value)
        if mode == "pair1":
            g, u = pair_module.pair(value, gate, up, quant)
        else:
            g, u = paired.mlx_block_fp8_pair(value, gate.weight, gate.scales,
                                           up.weight, up.scales, quant)
    else:
        quant_fn = direct.mlx_fp8_dynamic_32 if mode in ("direct", "shared") else baseline.mlx_fp8_dynamic_32
        if mode == "shared":
            # The same quantized codes/scales are passed to both projections.
            quant = quant_fn(value)
            g, u = proj(value, gate, quant), proj(value, up, quant)
        else:
            g = proj(value, gate, quant_fn(value))
            u = proj(value, up, quant_fn(value))
    activated = nn.silu(mx.minimum(g, 7.0)) * mx.clip(u, -7.0, 7.0)
    if mode == "repeated":
        result = repeated.project(activated, down.weight, down.scales)
    else:
        quant_fn = baseline.mlx_fp8_dynamic_32 if mode == "baseline" else direct.mlx_fp8_dynamic_32
        result = proj(activated, down, quant_fn(activated))
    return value + result


def chain(value, mode):
    if args.compiled_group:
        for fn in compiled[mode]:
            value = fn(value)
        return value
    for banks in weights:
        value = layer(value, banks, mode)
    return value


def save(data):
    suffix = f"capture_{args.capture_mode}" if args.capture else f"macro_g{args.compiled_group}"
    (OUT / f"ffn_{suffix}_c{args.count}_{args.dtype}.json").write_text(json.dumps(data, indent=2))


modes = ("baseline", *args.modes.split(","))
compiled = {}
if args.compiled_group:
    for mode in modes:
        compiled[mode] = []
        for start in range(0, args.layers, args.compiled_group):
            def block(value, start=start, mode=mode):
                for banks in weights[start : start + args.compiled_group]:
                    value = layer(value, banks, mode)
                return value
            compiled[mode].append(mx.compile(block))
data = {"device": mx.device_info(), "dtype": str(x.dtype), "count": args.count,
        "layers": args.layers, "steps": args.steps,
        "scope": "synthetic eight-layer shared block-FP8 FFN chain; no whole-model throughput",
        "weight_bytes": sum(p.weight.nbytes + p.scales.nbytes for banks in weights for p in banks),
        "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
        "numerics": [], "trials": [], "first_ms": {}, "source_hashes": {},
        "compiled_group": args.compiled_group}
for p in (OUT / "ffn_macro.py", OUT / "common.py", OUT / "repeated_quant.py",
          ROOT / "benchmark/mlx/throughput_20261103/quant_hoisted.py",
          ROOT / "python/sglang/kernels/ops/quantization/mlx_fp8_dynamic.py"):
    data["source_hashes"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
path = ROOT / "python/sglang/kernels/ops/gemm/mlx_block_fp8.py"
data["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
expected = chain(x, "baseline")
mx.eval(expected)
for mode in modes:
    start = time.perf_counter()
    result = chain(x, mode)
    mx.eval(result)
    data["first_ms"][mode] = (time.perf_counter() - start) * 1000
    diff = mx.abs(result - expected)
    rec = {"mode": mode, "exact": mx.array_equal(result, expected).item(),
           "max_abs": mx.max(diff).item()}
    data["numerics"].append(rec)
    print("MACRO_NUMERIC", rec, flush=True)
    assert rec["exact"], rec
save(data)
if args.capture:
    # Drain prior numerical checks before starting capture, then use a distinct
    # evaluated input so all layer dispatches belong to this chain.
    capture_input = x + 0.001
    mx.eval(capture_input)
    mx.synchronize()
    path = str(OUT / f"ffn_{args.capture_mode}_c{args.count}_l{args.layers}.gputrace")
    mx.metal.start_capture(path)
    capture_stream = mx.new_stream(mx.gpu)
    with mx.stream(capture_stream):
        result = chain(capture_input, args.capture_mode)
        mx.eval(result)
        mx.synchronize(capture_stream)
    mx.metal.stop_capture()
    print("CAPTURE", path, flush=True)
else:
    # ABBA per mechanism; each sample performs300 evaluations of a dependency
    # chain with distinct banks. Save per-step latency and peak live allocations.
    for candidate in modes[1:]:
        final_reference = None
        for mode in ("baseline", candidate, candidate, "baseline"):
            for _ in range(3):
                mx.eval(chain(x, mode))
            mx.reset_peak_memory()
            raw = []
            value = x
            for step in range(args.steps):
                start = time.perf_counter()
                value = chain(value, mode)
                mx.eval(value)
                raw.append(time.perf_counter() - start)
            finite = mx.all(mx.isfinite(value)).item()
            assert finite, mode
            if final_reference is None:
                final_reference = value
            exact_final = mx.array_equal(value, final_reference).item()
            assert exact_final, (candidate, mode, "300-step chain diverged")
            row = {"candidate": candidate, "mode": mode, "raw_seconds": raw,
                   "median_ms": statistics.median(raw) * 1000,
                   "mean_ms": statistics.mean(raw) * 1000,
                   "active_bytes": mx.get_active_memory(), "peak_bytes": mx.get_peak_memory(),
                   "final_300_exact": exact_final}
            data["trials"].append(row)
            save(data)
            print("MACRO300", args.count, candidate, mode, row["mean_ms"], row["peak_bytes"], flush=True)
    data["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    save(data)
