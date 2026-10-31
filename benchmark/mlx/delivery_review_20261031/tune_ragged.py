"""Alternate indexed MPP tile shapes against one-gather native attention."""

import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from sglang.kernels.ops.attention import mlx_ragged as r

original = r._kernels
layout = (
    "const uint WIDTH = lengths[ROWS];\nconst uint KV_LENGTH = lengths[ROWS + 1];\n"
)
header = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
variants = {}
for prefetch in (False, True):
    scores = r._NAX_SCORES_REGULAR.replace(
        "16, 32, 16, false", "16, 32, 32, false"
    ).replace("k += 16", "k += 32")
    values = r._NAX_VALUES.replace("16, 32, 16, false", "16, 32, 32, false").replace(
        "k += 16", "k += 32"
    )
    if prefetch:
        marker = "    const uint partitions = length < 256 ? 2 : 1;"
        scores = scores.replace(
            marker,
            """
    uint slots[32];
    for (uint i = 0; i < b.get_capacity(); ++i) {
        auto coord = b.get_multidimensional_index(i);
        uint p = col + coord[1];
        slots[i] = p < length ? indices[size_t(r) * WIDTH + p] : 0;
    }
"""
            + marker,
        )
        scores = scores.replace(
            "uint slot = p < length ? indices[size_t(r) * WIDTH + p] : 0;",
            "uint slot = slots[i];",
        )
    sk = mx.fast.metal_kernel(
        name=f"delivery_qk32_{prefetch}",
        input_names=["queries", "kv", "indices", "lengths"],
        output_names=["scores"],
        source=layout + scores,
        header=header,
    )
    vk = mx.fast.metal_kernel(
        name=f"delivery_pv32_{prefetch}",
        input_names=["probs", "kv", "indices", "lengths"],
        output_names=["out"],
        source=layout + values,
        header=header,
    )
    variants["k32_prefetch" if prefetch else "k32"] = (sk, original(True, True)[1], vk)

report = dict(device=mx.device_info(), cases=[])
for kv_dtype in (mx.bfloat16, mx.float32):
    for prefix in (16, 128, 2050):
        for rows in (2, 3, 5, 9):
            mx.random.seed(751)
            q = mx.random.normal((1, 64, rows, 512))
            kv = mx.random.normal((1, prefix + rows, 512)).astype(kv_dtype)
            lengths = [prefix + j + 1 for j in range(rows)]
            width = max(lengths)
            selected = [
                mx.random.permutation(prefix + j + 1).astype(mx.int32)
                for j in range(rows)
            ]
            idx = mx.stack(
                [mx.pad(i, (0, width - i.size), constant_values=-1) for i in selected]
            )[None]

            def native():
                out = []
                for j, i in enumerate(selected):
                    picked = mx.take(kv, i, axis=1)[:, None]
                    out.append(
                        mx.fast.scaled_dot_product_attention(
                            q[:, :, j : j + 1], picked, picked, scale=256**-0.5
                        )
                    )
                return mx.concatenate(out, 2)

            def indexed(name):
                with (
                    patch.object(r, "_kernels", return_value=variants[name])
                    if name in variants
                    else patch.object(r, "_kernels", original)
                ):
                    return r.mlx_ragged_latent_attention(
                        q, kv, idx, lengths, scale=256**-0.5
                    )

            fns = {
                "native": native,
                **{name: lambda n=name: indexed(n) for name in ["current", *variants]},
            }
            ref = native()
            mx.eval(ref)
            records = {}
            for name, fn in fns.items():
                actual = fn()
                mx.eval(actual)
                records[name] = dict(
                    candidate=name,
                    kv_dtype=str(kv_dtype),
                    prefix=prefix,
                    rows=rows,
                    equal=mx.array_equal(ref, actual).item(),
                    samples_ms=[],
                )
                for _ in range(20):
                    mx.eval(fn())
            for repeat in range(8):
                for name in fns if repeat % 2 == 0 else list(fns)[::-1]:
                    started = time.perf_counter()
                    for _ in range(8):
                        mx.eval(fns[name]())
                    records[name]["samples_ms"].append(
                        (time.perf_counter() - started) * 125
                    )
            for record in records.values():
                record["median_ms"] = statistics.median(record["samples_ms"])
                report["cases"].append(record)
            print(
                json.dumps(
                    {
                        name: dict(equal=v["equal"], ms=v["median_ms"])
                        for name, v in records.items()
                    }
                ),
                flush=True,
            )
            Path(__file__).with_suffix(".json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
