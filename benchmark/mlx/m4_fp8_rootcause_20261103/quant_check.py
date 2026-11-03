"""Finite codebook boundary oracle, exact code/scale checks and AB/BA timing."""

import json
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

from common import OUT, baseline, direct

mx.set_cache_limit(256 * 2**20)
mx.random.seed(311)
book = np.array(
    [m * 2**-9 if e == 0 else (1 + m / 8) * 2 ** (e - 7)
     for e in range(16) for m in range(8) if e * 8 + m < 127],
    dtype=np.float32,
)
mid = (book[:-1] + book[1:]) / 2
values = np.concatenate((book, mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf),
                         np.array([0.0, -0.0, 448.0, 1000.0], np.float32)))
values = np.concatenate((values, -values))
# Each group has max448, so the production scale is exactly1. The independent
# oracle searches all finite codebook entries and selects an even code on ties.
blocks = np.zeros((len(values), 32), np.float32)
blocks[:, 0] = values
blocks[:, 1] = 448.0
checks = []
for dtype in (mx.float32, mx.bfloat16):
    x = mx.array(blocks).astype(dtype)
    codes, scales = direct.mlx_fp8_dynamic_32(x)
    oldc, olds = baseline.mlx_fp8_dynamic_32(x)
    mx.eval(codes, scales, oldc, olds)
    assert mx.array_equal(codes, oldc).item(), str(dtype)
    assert mx.array_equal(scales, olds).item(), str(dtype)
    actual_x = np.array(x.astype(mx.float32))
    maxima = np.max(np.abs(actual_x), axis=1)
    expected_scales = np.exp2(np.ceil(np.log2(maxima / 448))).astype(np.float32)
    assert np.array_equal(np.array(scales).ravel(), expected_scales)
    distances = np.abs(np.minimum(np.abs(actual_x[:, 0]) / expected_scales, 448)[:, None] - book)
    minimum = distances.min(1, keepdims=True)
    choices = distances == minimum
    # Prefer even code then low code. Adjacent midpoint codes always one even.
    expected = np.array([next((j for j in np.flatnonzero(row) if j % 2 == 0),
                             int(np.flatnonzero(row)[0])) for row in choices], np.uint8)
    expected |= np.where(np.signbit(actual_x[:, 0]), 128, 0).astype(np.uint8)
    assert np.array_equal(np.array(codes)[:, 0], expected), str(dtype)
    checks.append({"dtype": str(dtype), "boundary_cases": len(values), "exact": True})
for dtype in (mx.float32, mx.bfloat16):
    for count in (1, 4, 6, 8, 17, 33):
        for magnitude in (2**-12, 1, 2**12):
            x = (mx.random.normal((count, 4096)) * magnitude).astype(dtype)
            a = baseline.mlx_fp8_dynamic_32(x)
            b = direct.mlx_fp8_dynamic_32(x)
            mx.eval(a, b)
            assert all(mx.array_equal(u, v).item() for u, v in zip(a, b))
print("QUANT_NUMERIC", checks, flush=True)
trials = []
for count in (1, 4, 32, 256):
    x = mx.random.normal((count, 4096))
    mx.eval(x)
    for j in range(9):
        row = {"count": count, "repeat": j}
        for name in (("binary", "direct") if j % 2 == 0 else ("direct", "binary")):
            fn = baseline.mlx_fp8_dynamic_32 if name == "binary" else direct.mlx_fp8_dynamic_32
            mx.eval(fn(x))
            start = time.perf_counter()
            for _ in range(30):
                mx.eval(fn(x))
            row[name] = (time.perf_counter() - start) / 30
        trials.append(row)
    print("QUANT_MICRO", count,
          {n: statistics.median(r[n] for r in trials if r["count"] == count) * 1000
           for n in ("binary", "direct")}, flush=True)
(OUT / "quant_check.json").write_text(json.dumps({"checks": checks, "trials": trials,
    "device": mx.device_info(), "swap": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)}, indent=2))
