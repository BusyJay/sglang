"""Short paired MLA/mHC prefill probes with random BF16 production shapes.

Compare a Git revision or saved module source with the working implementation.
This loads no checkpoint weights and does not estimate whole-model TTFT.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reference_arg = parser.add_mutually_exclusive_group(required=True)
    reference_arg.add_argument("--reference-ref")
    reference_arg.add_argument("--reference-source", type=Path)
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quantization-bits", type=int, choices=(4, 5, 6))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not all(t > 1 for t in args.query_lengths):
        parser.error("Require positive repeats and query lengths > 1")

    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import hf_hub_download
    from mlx.utils import tree_flatten

    from sglang.srt.hardware_backend.mlx.models import glm5_next as current

    source_path = "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py"
    source = (
        args.reference_source.read_text()
        if args.reference_source
        else subprocess.check_output(
            ["git", "show", f"{args.reference_ref}:{source_path}"], text=True
        )
    )
    reference = types.ModuleType("glm5_next_prefill_reference")
    sys.modules[reference.__name__] = reference
    exec(compile(source, "reference_glm5_next.py", "exec"), reference.__dict__)
    model_id = "mlx-community/GLM-5.3-Flash-4bit"
    config_path = hf_hub_download(model_id, "config.json", local_files_only=True)
    config = current.ModelArgs.from_dict(json.loads(Path(config_path).read_text()))
    mx.random.seed(23)
    before = reference.MLA(config)
    before.set_dtype(mx.bfloat16)
    if args.quantization_bits:
        nn.quantize(before, group_size=64, bits=args.quantization_bits)
    mx.eval(before.parameters())
    after = current.MLA(config)
    if args.quantization_bits:
        nn.quantize(after, group_size=64, bits=args.quantization_bits)
    after.load_weights(tree_flatten(before.parameters()))
    mx.eval(after.parameters())
    mx.set_cache_limit(1024**3)
    h = config.hc_mult
    fn = mx.random.normal(((2 + h) * h, h * config.hidden_size)) * 0.01
    base = mx.random.normal(((2 + h) * h,))
    scale = mx.ones((3,))
    mx.eval(fn, base, scale)
    rows = []
    for tokens in args.query_lengths:
        if tokens > config.index_topk:
            parser.error("Dense MLA probe query length must not exceed index_topk")
        x = mx.random.normal((1, tokens, config.hidden_size)).astype(mx.bfloat16)
        streams = mx.random.normal((1, tokens, h, config.hidden_size)).astype(
            mx.bfloat16
        )
        mx.eval(x, streams)
        for component in ("mla", "mhc"):
            timings = {"before": [], "after": []}
            peaks = {"before": [], "after": []}
            # Alternate order to reduce systematic cache/clock bias. First
            # pair warms the shape and compilation; only subsequent pairs count.
            for trial in range(args.repeats + 1):
                order = ("before", "after") if trial % 2 == 0 else ("after", "before")
                for label in order:
                    module = before if label == "before" else after
                    implementation = reference if label == "before" else current
                    cache = implementation.NativeCache(3)
                    mx.reset_peak_memory()
                    resident = mx.get_active_memory()
                    start = time.perf_counter()
                    if component == "mla":
                        output = module(x, cache)
                        mx.eval(output, cache.cache)
                    else:
                        output = implementation.hyper_connection(
                            streams, fn, base, scale, config
                        )
                        mx.eval(output)
                    elapsed = time.perf_counter() - start
                    extra_peak = mx.get_peak_memory() - resident
                    if trial:
                        timings[label].append(elapsed)
                        peaks[label].append(extra_peak / 2**20)
                    del output, cache
                    gc.collect()
            for label in ("before", "after"):
                row = {
                    "component": component,
                    "implementation": label,
                    "query_tokens": tokens,
                    "samples_s": timings[label],
                    "median_s": statistics.median(timings[label]),
                    "median_extra_peak_MiB": statistics.median(peaks[label]),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = {
        "method": "Random BF16 production shapes, identical weights/inputs, one warmup pair then alternating timed calls with a warm 1 GiB allocator cache. MLA uses an empty cache. No full-model TTFT or quality claim.",
        "model_id": model_id,
        "reference_ref": args.reference_ref,
        "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256(
            Path(source_path).read_bytes()
        ).hexdigest(),
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "repeats": args.repeats,
        "quantization_bits": args.quantization_bits,
        "quantization_group_size": 64 if args.quantization_bits else None,
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
