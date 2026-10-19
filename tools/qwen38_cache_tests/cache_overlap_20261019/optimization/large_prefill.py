"""Exercise two complete MoE subchunks and a tail against resident CUTLASS."""

import inspect
import json
import runpy
import sys
from pathlib import Path

from sglang.srt.model_loader.qwen38_nvfp4_pack import parity

source = inspect.getsource(parity.check_exported_layer)
needle = "max(2, (min(slots, layer.num_experts) // config.top_k) + 2)"
assert needle in source
namespace = dict(vars(parity))
exec(compile(source.replace(needle, "4101"), parity.__file__, "exec"), namespace)
check = namespace["check_exported_layer"]


def run(layer, exported, *args, **kwargs):
    report = check(
        layer,
        exported,
        *args,
        **dict(
            kwargs,
            prefill_direct_cache=True,
            decode_early_cache=True,
            decode_graph=False,
            benchmark_iterations=0,
        ),
    )
    Path("/results/optimization/large-prefill.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)
    raise SystemExit(0)


parity.check_exported_layer = run
sys.argv = [
    "cached_w4a16_gpu_check.py",
    "--cache-overlap-only",
    "--iterations",
    "1",
    "--output",
    "/results/optimization/unused.json",
]
runpy.run_path(
    "tools/qwen38_cache_tests/cached_w4a16_gpu_check.py", run_name="__main__"
)
