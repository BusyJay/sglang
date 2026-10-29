"""Paired incremental GLM indexer/MLA decode with production shapes.

Uses identical random BF16 weights, raw prefix state and decode inputs for the
reference revision and working tree. Both caches advance through a real token
stream. This isolates indexer/MLA work; it does not measure full-model serving.
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
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", required=True)
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[1024, 4096, 16384]
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("indexer", "mla"),
        default=["indexer", "mla"],
    )
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.context_lengths + [args.repeats, args.warmup]) < 1:
        parser.error("context lengths, repeats and warmup must be positive")

    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx.utils import tree_flatten

    from sglang.srt.hardware_backend.mlx.models import glm5_next as current

    path = "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py"
    reference_ref = subprocess.check_output(
        ["git", "rev-parse", args.reference_ref], text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{reference_ref}:{path}"], text=True
    )
    reference = types.ModuleType("glm5_next_indexer_reference")
    sys.modules[reference.__name__] = reference
    exec(compile(source, "reference_glm5_next.py", "exec"), reference.__dict__)
    config_path = hf_hub_download(
        "mlx-community/GLM-5.3-Flash-4bit", "config.json", local_files_only=True
    )
    config = current.ModelArgs.from_dict(json.loads(Path(config_path).read_text()))
    mx.set_cache_limit(1024**3)
    rows = []

    for component in args.components:
        mx.random.seed(23)
        before = getattr(reference, "Indexer" if component == "indexer" else "MLA")(
            config
        )
        before.set_dtype(mx.bfloat16)
        after = getattr(current, "Indexer" if component == "indexer" else "MLA")(config)
        after.set_dtype(mx.bfloat16)
        after.load_weights(tree_flatten(before.parameters()))
        mx.eval(before.parameters(), after.parameters())
        for context in args.context_lengths:
            mx.random.seed(31)
            prefix = [
                mx.random.normal((1, context, dim)).astype(mx.bfloat16)
                for dim in (
                    config.kv_lora_rank,
                    config.index_head_dim,
                    config.index_head_dim,
                )
            ]
            if component == "indexer":
                prefix[0] = None
            caches = {
                "before": reference.NativeCache(3),
                "after": current.NativeCache(4),
            }
            for cache in caches.values():
                cache.cache[:3] = prefix
                cache.offset = context
            indexer = after if component == "indexer" else after.indexer
            # Convert the identical existing prefix once, outside timing. New
            # requests build this state incrementally during their prefills.
            empty = mx.zeros((1, 0, config.index_head_dim), mx.bfloat16)
            indexer._update_pool_cache(empty, empty, caches["after"], context)
            steps = args.warmup + args.repeats
            inputs = mx.random.normal((1, steps, config.hidden_size)).astype(
                mx.bfloat16
            )
            qr = mx.random.normal((1, steps, config.q_lora_rank)).astype(mx.bfloat16)
            mx.eval(inputs, qr, *[c.cache for c in caches.values()])
            timings = {"before": [], "after": []}
            max_error = 0.0
            for step in range(steps):
                outputs = {}
                for label in (
                    ("before", "after") if step % 2 == 0 else ("after", "before")
                ):
                    module = before if label == "before" else after
                    cache = caches[label]
                    x = inputs[:, step : step + 1]
                    query = qr[:, step : step + 1]
                    start = time.perf_counter()
                    if component == "indexer":
                        output = module(x, query, cache, context + step)
                    else:
                        output = module(x, cache)
                    mx.eval(cache.cache, *([] if output is None else [output]))
                    elapsed = (time.perf_counter() - start) * 1000
                    if step >= args.warmup:
                        timings[label].append(elapsed)
                    outputs[label] = output
                left, right = outputs["before"], outputs["after"]
                if left is None or right is None:
                    if left is not None or right is not None:
                        raise RuntimeError("dense/sparse branch mismatch")
                elif component == "indexer":
                    if not mx.array_equal(
                        mx.sort(left, axis=-1), mx.sort(right, axis=-1)
                    ).item():
                        raise RuntimeError("selected index sets differ")
                else:
                    error = mx.max(
                        mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
                    ).item()
                    max_error = max(max_error, error)
                    if not mx.allclose(left, right, atol=0.01, rtol=0.03).item():
                        raise RuntimeError(f"MLA output mismatch: {error}")
            medians = {
                label: statistics.median(values) for label, values in timings.items()
            }
            row = dict(
                component=component,
                prefix_tokens=context,
                decode_steps=steps,
                before_ms=medians["before"],
                after_ms=medians["after"],
                speedup=medians["before"] / medians["after"],
                max_abs_error=max_error,
                indexer_cache_bytes={
                    label: sum(a.nbytes for a in c.cache[1:] if a is not None)
                    for label, c in caches.items()
                },
                trials_ms=timings,
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "trials_ms"}),
                flush=True,
            )
            del caches, cache, prefix, inputs, qr, outputs, output, left, right
            gc.collect()
            mx.clear_cache()
        del before, after, indexer
        gc.collect()
        mx.clear_cache()

    report = dict(
        scope=__doc__,
        reference_ref=reference_ref,
        working_source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        device=mx.device_info(),
        mlx=version("mlx"),
        mlx_lm=version("mlx-lm"),
        mlx_enable_tf32=os.environ.get("MLX_ENABLE_TF32", "unset"),
        dtype="bfloat16",
        repeats=args.repeats,
        warmup=args.warmup,
        memory_scope="logical bytes of indexer cache arrays including spare pooled capacity; excludes latent KV and allocator cache",
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
