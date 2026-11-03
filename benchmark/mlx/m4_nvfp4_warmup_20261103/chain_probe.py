"""Move identical compiled FFN first evaluation to an explicit startup phase."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time

import mlx.core as mx

from common import OUT, ROOT, bank, fixture, native, routing

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", action="store_true")
parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()
mx.set_memory_limit(2 * 2**30)
mx.set_cache_limit(128 * 2**20)
mx.random.seed(120)
ids, experts = routing(1)
banks = [[bank(experts, n, k) for n, k in ((2048, 4096), (2048, 4096), (4096, 2048))]
         for _ in range(4)]
x = mx.random.normal((1, 1, 4096)).astype(mx.float32 if args.dtype == "fp32" else mx.bfloat16)
mx.eval(x)
scores = mx.full((1, 1, 8, 1), 1 / 8, mx.float32)
mx.eval(scores)


def layer(x, projections):
    gate, up, down = projections
    activated = fixture.production.mlx_nvfp4_swiglu(
        x[..., None, None, :], gate.weight, gate.scales, gate.weight_scale_2 * 2688,
        up.weight, up.scales, up.weight_scale_2 * 2688, ids, limit=7.0,
    )
    return x + mx.sum(native(activated, down, ids).squeeze(-2) * scores, axis=-2).astype(x.dtype)


functions = []
for start in range(0, 4, 2):
    def fn(value, start=start):
        for projections in banks[start:start + 2]:
            value = layer(value, projections)
        return value
    functions.append(mx.compile(fn))


def chain(value):
    for fn in functions:
        value = fn(value)
    return value


startup_ms = 0.0
if args.warmup:
    begin = time.perf_counter()
    mx.eval(chain(x))
    startup_ms = (time.perf_counter() - begin) * 1000
begin = time.perf_counter()
value = chain(x)
mx.eval(value)
first_ms = (time.perf_counter() - begin) * 1000
raw = []
for _ in range(300):
    begin = time.perf_counter()
    value = chain(value)
    mx.eval(value)
    raw.append((time.perf_counter() - begin) * 1000)
assert mx.all(mx.isfinite(value)).item()
data = dict(
    args=vars(args), startup_warmup_ms=startup_ms, first_request_step_ms=first_ms,
    steady_mean_ms=statistics.mean(raw), steady_median_ms=statistics.median(raw),
    raw_ms=raw, peak_bytes=mx.get_peak_memory(), device=mx.device_info(), mlx=mx.__version__,
    swap_after=subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
    scope="M4 four-layer actual-width synthetic compiled FFN chain; new process, OS/compiler disk cache not cleared",
    source_sha256=hashlib.sha256((ROOT / "python/sglang/kernels/ops/gemm/mlx_nvfp4_swiglu.py").read_bytes()).hexdigest(),
)
name = f'chain_{args.dtype}_{"warm" if args.warmup else "lazy"}_{args.sample}.json'
(OUT / name).write_text(json.dumps(data, indent=2))
print({key: data[key] for key in ("startup_warmup_ms", "first_request_step_ms", "steady_mean_ms")}, flush=True)
