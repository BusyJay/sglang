"""Force real misses before every timed decode; never benchmark warmed hits as misses."""

import json
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch

from sglang.srt.layers.quantization.qwen38_nvfp4_cache import Qwen38CachedNvFp4MoeMethod

original = Qwen38CachedNvFp4MoeMethod._run_experts
completed = False


def run(self, x, weights, ids, logits, pre_quant=None, **kwargs):
    global completed
    if completed:
        return original(self, x, weights, ids, logits, **kwargs)
    completed = True
    runtime = self.runtime
    runtime.decode_early_cache = False
    for values in (list(range(10)), list(range(10, 16)) + [-1] * 4):
        seed_ids = torch.tensor([values], device=x.device, dtype=ids.dtype)
        original(self, x, weights, seed_ids, logits, execution_phase="decode")
    runtime.synchronize()
    runtime.gate_warmed_layers.add(0)
    state = runtime.device_states[0]
    metadata = {k: v.clone() for k, v in state.items()}
    payload = {k: v.clone() for k, v in runtime.caches[0].items()}
    results = []
    iterations = 12
    runtime.prefill_tokens_seen = 1 << 20
    for tokens, misses in ((n, m) for n in (1024,) for m in range(11)):
        route = list(range(10 - misses)) + list(range(16, 16 + misses))
        selected = torch.tensor([route], device=x.device, dtype=ids.dtype).repeat(
            tokens, 1
        )
        inputs, router = x.repeat(tokens, 1), weights.repeat(tokens, 1)
        expected = None
        for mode in ("off", "always", "auto"):
            early = mode != "off"
            runtime.cache_overlap_policy = "auto" if mode == "auto" else "always"
            runtime.prefill_direct_cache = early
            before = dict(runtime.prefill_copy_stats[0])
            times, wall = [], []
            # First iteration warms every selected backend specialization.
            for iteration in range(iterations + 3):
                runtime.synchronize()
                for k, v in metadata.items():
                    state[k].copy_(v)
                for k, v in payload.items():
                    runtime.caches[0][k].copy_(v)
                runtime.arena_layer = None
                runtime.arena_residents = set()
                torch.cuda.synchronize()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                started = time.perf_counter()
                output = original(
                    self, inputs, router, selected, None, execution_phase="prefill"
                )
                end.record()
                end.synchronize()
                if iteration >= 3:
                    times.append(begin.elapsed_time(end))
                    wall.append((time.perf_counter() - started) * 1000)

            if expected is None:
                expected = output.clone()
            else:
                torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            if (tokens, misses) in ((3, 0), (3, 1), (3, 5), (2048, 5)):
                runtime.synchronize()
                for k, v in metadata.items():
                    state[k].copy_(v)
                for k, v in payload.items():
                    runtime.caches[0][k].copy_(v)
                runtime.arena_layer = None
                runtime.arena_residents = set()
                torch.cuda.synchronize()
                directory = Path("/results/adaptive/profiles")
                directory.mkdir(exist_ok=True)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    original(
                        self, inputs, router, selected, None, execution_phase="prefill"
                    )
                    torch.cuda.synchronize()
                prof.export_chrome_trace(
                    str(directory / f"prefill-{mode}-{tokens}-{misses}.trace.json")
                )
            after = runtime.prefill_copy_stats[0]
            assert after["host_missing"] - before["host_missing"] == misses * (
                iterations
                + 3
                + int((tokens, misses) in ((3, 0), (3, 1), (3, 5), (2048, 5)))
            )
            selected_paths = {
                k: after.get(k, 0) - before.get(k, 0)
                for k in ("auto_arena_runs", "auto_cache_runs", "auto_split_runs")
            }
            result = dict(
                mode=mode,
                selected_paths=selected_paths,
                tokens=tokens,
                misses=misses,
                early=early,
                median_ms=statistics.median(times),
                wall_ms=statistics.median(wall),
                min_ms=min(times),
                max_ms=max(times),
                iterations=iterations,
                bytes_per_miss=sum(runtime.miss_mailbox.sizes[0]),
            )
            print(json.dumps(result), flush=True)
            results.append(result)
    Path("/results/adaptive/prefill-auto-1024.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    raise SystemExit(0)


Qwen38CachedNvFp4MoeMethod._run_experts = run
sys.argv = [
    "cached_w4a16_gpu_check.py",
    "--cache-overlap-only",
    "--iterations",
    "1",
    "--output",
    "/results/adaptive/unused.json",
]
runpy.run_path(
    "tools/qwen38_cache_tests/cached_w4a16_gpu_check.py", run_name="__main__"
)
