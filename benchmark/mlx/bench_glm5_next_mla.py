"""Compare MLA latency and temporary active memory with a previous Git revision.

Uses random BF16 weights at the checkpoint's real attention dimensions. This
isolates MLA and does not predict full-model latency or validate model quality.
Run from the checkout root with MLX_ENABLE_TF32 set explicitly.
"""

import argparse
import gc
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
    parser.add_argument("--reference-ref", required=True)
    parser.add_argument("--context-length", type=int, default=3072)
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[1, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not all(
        0 < q <= args.context_length for q in args.query_lengths
    ):
        parser.error("Require positive repeats and 0 < query length <= context length")

    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx.utils import tree_flatten

    from sglang.srt.hardware_backend.mlx.models import glm5_next as current

    source_path = "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py"
    source = subprocess.check_output(
        ["git", "show", f"{args.reference_ref}:{source_path}"], text=True
    )
    reference = types.ModuleType("glm5_next_benchmark_reference")
    sys.modules[reference.__name__] = reference
    exec(compile(source, "reference_glm5_next.py", "exec"), reference.__dict__)
    config_path = hf_hub_download(
        "mlx-community/GLM-5.3-Flash-4bit", "config.json", local_files_only=True
    )
    config = current.ModelArgs.from_dict(json.loads(Path(config_path).read_text()))
    mx.random.seed(23)
    before = reference.MLA(config)
    before.set_dtype(mx.bfloat16)
    mx.eval(before.parameters())
    after = current.MLA(config)
    after.load_weights(tree_flatten(before.parameters()))
    mx.eval(after.parameters())
    mx.set_cache_limit(1024**3)
    rows = []
    for query_length in args.query_lengths:
        prefix_length = args.context_length - query_length
        prefix = [
            mx.random.normal((1, prefix_length, width)).astype(mx.bfloat16)
            for width in (
                config.kv_lora_rank,
                config.index_head_dim,
                config.index_head_dim,
            )
        ]
        x = mx.random.normal((1, query_length, config.hidden_size)).astype(mx.bfloat16)
        mx.eval(prefix, x)
        for label, module, cache_type in (
            ("before", before, reference.NativeCache),
            ("after", after, current.NativeCache),
        ):
            timings, peaks = [], []
            for i in range(args.repeats + 1):
                cache = cache_type(3)
                cache.cache, cache.offset = list(prefix), prefix_length
                mx.clear_cache()
                mx.reset_peak_memory()
                resident = mx.get_active_memory()
                start = time.perf_counter()
                output = module(x, cache)
                mx.eval(output, cache.cache)
                elapsed = time.perf_counter() - start
                extra_peak = mx.get_peak_memory() - resident
                if i:
                    timings.append(elapsed)
                    peaks.append(extra_peak)
                del output, cache
                gc.collect()
            rows.append(
                {
                    "implementation": label,
                    "context_tokens": args.context_length,
                    "query_tokens": query_length,
                    "median_s": statistics.median(timings),
                    "median_extra_peak_MiB": statistics.median(peaks) / 2**20,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    report = {
        "method": "Isolated random BF16 MLA, production shapes, identical weights/prefix, one warmup then timed calls. Extra active-memory peak excludes resident weights/inputs. No full-model performance attribution.",
        "reference_ref": args.reference_ref,
        "repeats": args.repeats,
        "heads": config.num_attention_heads,
        "latent_rank": config.kv_lora_rank,
        "topk": config.index_topk,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
