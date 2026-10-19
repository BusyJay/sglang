"""Reuse the same synthetic weights and seeded inputs for opt-in A/B checks."""

import json
import os
import runpy
import sys
from pathlib import Path

from sglang.srt.model_loader.qwen38_nvfp4_pack import parity

original = parity.check_exported_layer


def matrix(layer, exported, *args, **kwargs):
    reports = []
    for direct, early in [(False, False), (True, False), (False, True), (True, True)]:
        for graph in (False, True):
            options = dict(
                kwargs,
                decode_graph=graph,
                prefill_direct_cache=direct,
                decode_early_cache=early,
            )
            try:
                report = original(layer, exported, *args, **options)
            except Exception as error:
                report = dict(
                    status="failed",
                    prefill_direct_cache=direct,
                    decode_early_cache=early,
                    decode_graph=graph,
                    error=str(error),
                )
            reports.append(report)
            print(json.dumps(report), flush=True)
    result = dict(
        reports=reports,
        gpu=__import__("torch").cuda.get_device_name(),
        backend=os.getenv("SGLANG_QWEN38_CACHE_W4A16_BACKEND", "triton"),
        scope="synthetic single-layer warm repeated invocations, not model throughput",
    )
    result["status"] = (
        "passed" if all(r["status"] == "passed" for r in reports) else "failed"
    )
    Path("/results/optimization/ab-" + result["backend"] + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)


parity.check_exported_layer = matrix
sys.argv = [
    "cached_w4a16_gpu_check.py",
    "--cache-overlap-only",
    "--iterations",
    "30",
    "--output",
    "/results/unused.json",
]
runpy.run_path(
    "tools/qwen38_cache_tests/cached_w4a16_gpu_check.py", run_name="__main__"
)
