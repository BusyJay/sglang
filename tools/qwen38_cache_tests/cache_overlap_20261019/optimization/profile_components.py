"""Profile warmed single-layer calls and preserve scopes for attribution."""

import functools
import json
import runpy
import sys
from pathlib import Path

import torch

from sglang.srt.layers.moe import (
    qwen38_cache_mailbox,
    qwen38_cache_overlap,
    qwen38_prefill_routes,
)
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass
from sglang.srt.layers.quantization import qwen38_cached_b12x, qwen38_cached_w4a16
from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedNvFp4MoeMethod
from sglang.srt.model_loader.qwen38_nvfp4_pack import parity

out = Path("/results/optimization/profiles")
out.mkdir(parents=True, exist_ok=True)


def scope(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


scope(qwen38_cached_w4a16, "cached_w4a16_moe", "component.triton_w4a16")
scope(qwen38_cached_b12x.CachedB12xW4A16, "run", "component.b12x_w4a16")
scope(qwen38_prefill_routes, "partition_prefill_routes", "component.partition_routes")
scope(qwen38_cache_mailbox.DecodeMissMailbox, "run", "component.mailbox")
scope(qwen38_cache_overlap, "fill_with_overlap", "component.fill_overlap")
scope(flashinfer_cutlass, "_run_flashinfer_cutlass", "component.cutlass_w4a4")
original_run = Qwen38CachedNvFp4MoeMethod._run_experts
counts = {}


@functools.wraps(original_run)
def profiled(self, *args, **kwargs):
    phase = kwargs.get("execution_phase")
    tokens = args[0].shape[0]
    key = (id(self), phase, tokens)
    counts[key] = counts.get(key, 0) + 1
    if counts[key] == 10 and (phase == "decode" or (phase == "prefill" and tokens > 1)):
        tag = f"d{int(self.runtime.prefill_direct_cache)}e{int(self.runtime.decode_early_cache)}-{phase}-{tokens}"
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                with torch.profiler.record_function("component.layer_call"):
                    result = original_run(self, *args, **kwargs)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out / (tag + ".trace.json")))
        (out / (tag + ".txt")).write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )
        print("PROFILE " + tag, flush=True)
        return result
    return original_run(self, *args, **kwargs)


Qwen38CachedNvFp4MoeMethod._run_experts = profiled
original_check = parity.check_exported_layer


def matrix(layer, exported, *args, **kwargs):
    reports = []
    for direct, early in [(False, False), (True, False), (False, True), (True, True)]:
        reports.append(
            original_check(
                layer,
                exported,
                *args,
                **dict(
                    kwargs,
                    decode_graph=False,
                    prefill_direct_cache=direct,
                    decode_early_cache=early,
                    benchmark_iterations=15,
                ),
            )
        )
    (out / "parity.json").write_text(json.dumps(reports, indent=2))
    raise SystemExit(0)


parity.check_exported_layer = matrix
sys.argv = [
    "cached_w4a16_gpu_check.py",
    "--cache-overlap-only",
    "--iterations",
    "15",
    "--output",
    "/results/investigation/unused.json",
]
runpy.run_path(
    "tools/qwen38_cache_tests/cached_w4a16_gpu_check.py", run_name="__main__"
)
