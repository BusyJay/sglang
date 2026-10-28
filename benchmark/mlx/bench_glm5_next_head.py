"""Compare final-position vocabulary projection with slicing full logits.

Random BF16 activations and a random Q4 affine head at production dimensions.
This is a short module probe, not a whole-model performance or quality test.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not all(t > 0 for t in args.query_lengths):
        parser.error("Require positive repeats and query lengths")

    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(41)
    head = nn.QuantizedLinear(4096, 154880, bias=False, group_size=64, bits=4)
    head.set_dtype(mx.bfloat16)
    mx.eval(head.parameters())
    mx.set_cache_limit(1024**3)
    rows = []
    for tokens in args.query_lengths:
        hidden = mx.random.normal((1, tokens, 4096)).astype(mx.bfloat16)
        mx.eval(hidden)
        times = {"full": [], "last": []}
        peaks = {"full": [], "last": []}
        outputs = {}
        for trial in range(args.repeats + 1):
            order = ("full", "last") if trial % 2 == 0 else ("last", "full")
            for label in order:
                mx.reset_peak_memory()
                resident = mx.get_active_memory()
                start = time.perf_counter()
                output = (
                    head(hidden)[:, -1:, :]
                    if label == "full"
                    else head(hidden[:, -1:, :])
                )
                mx.eval(output)
                elapsed = time.perf_counter() - start
                if trial:
                    times[label].append(elapsed)
                    peaks[label].append((mx.get_peak_memory() - resident) / 2**20)
                outputs[label] = output
        delta = outputs["full"].astype(mx.float32) - outputs["last"].astype(mx.float32)
        max_error = float(mx.max(mx.abs(delta)))
        if not bool(mx.all(mx.isfinite(delta))) or max_error > 0.02:
            raise AssertionError(f"Last-position head mismatch: {max_error}")
        row = {"query_tokens": tokens, "max_abs_error": max_error}
        for label in ("full", "last"):
            row[label + "_samples_s"] = times[label]
            row[label + "_median_s"] = statistics.median(times[label])
            row[label + "_extra_peak_MiB"] = statistics.median(peaks[label])
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "method": "Random Q4 affine head (4096 -> 154880, group size 64), BF16 activations, alternating timed pairs after one warmup pair. No full-model claim.",
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
