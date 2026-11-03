"""Raw alternating native/scan/list projection timings and native parity."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time

import mlx.core as mx

from common import OUT, bank, native, old_project, routing
from candidates import project, worklist

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="real", choices=("real", "none", "all"))
parser.add_argument("--count", type=int, default=4)
parser.add_argument("--shape", default="gate", choices=("gate", "down"))
parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()
mx.set_cache_limit(128 * 2**20)
mx.set_memory_limit(2 * 2**30)
mx.random.seed(913)
ids, experts = routing(args.count, args.pattern)
n, k = (2048, 4096) if args.shape == "gate" else (4096, 2048)
p = bank(experts, n, k)
shape = (1, args.count, 1, 1, k) if args.shape == "gate" else (1, args.count, 8, 1, k)
x = mx.random.normal(shape).astype(mx.float32 if args.dtype == "fp32" else mx.bfloat16)
work = worklist(ids)
mx.eval(x, ids, *work)
arms = {"native": lambda: native(x, p, ids)}
for rps in (2, 4):
    arms[f"old{rps}"] = lambda rps=rps: old_project(x, p, ids, rps)
    for kind, static in (("single", True), ("scan", True), ("list", False), ("list", True)):
        arms[f"{kind}_{int(static)}_r{rps}"] = lambda kind=kind, static=static, rps=rps: project(x, p, ids, kind=kind, static=static, rps=rps)
    arms[f"prelist_r{rps}"] = lambda rps=rps: project(x, p, ids, kind="list", static=True, rps=rps, work=work)
for fused in (False, True):
    arms[f"safe_single_{int(fused)}"] = lambda fused=fused: project(x, p, ids, kind="single", static=True, math="safe", fused=fused, rps=4)
data = {"args": vars(args), "device": mx.device_info(), "mlx": mx.__version__,
        "ids": ids.tolist(), "experts": experts, "first_ms": {}, "numerics": {}, "trials": [],
        "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
        "source_hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob("*.py")}}
path = OUT / f"micro_{args.shape}_{args.pattern}_c{args.count}_{args.dtype}.json"
reference = native(x, p, ids)
mx.eval(reference)
for name, fn in arms.items():
    start = time.perf_counter()
    y = fn()
    mx.eval(y)
    data["first_ms"][name] = (time.perf_counter() - start) * 1000
    data["numerics"][name] = {"exact": mx.array_equal(y, reference).item(),
                               "max_abs": mx.max(mx.abs(y.astype(mx.float32) - reference.astype(mx.float32))).item(),
                               "changed": mx.sum(y != reference).item()}
    print("NUMERIC", name, data["numerics"][name], flush=True)
for repeat in range(4):
    row = {}
    for name in list(arms) if repeat % 2 == 0 else list(reversed(arms)):
        fn = arms[name]
        mx.eval(fn())
        start = time.perf_counter()
        for _ in range(args.steps):
            mx.eval(fn())
        row[name] = (time.perf_counter() - start) * 1000 / args.steps
    data["trials"].append(row)
    path.write_text(json.dumps(data, indent=2))
    print("TRIAL", repeat, row, flush=True)
data["median_ms"] = {name: statistics.median(row[name] for row in data["trials"]) for name in arms}
data["peak_bytes"] = mx.get_peak_memory()
data["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
path.write_text(json.dumps(data, indent=2))
print("SUMMARY", data["median_ms"], flush=True)
