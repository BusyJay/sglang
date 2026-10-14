# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU analysis process; deliberately imports neither torch nor SGLang."""

import importlib.util
import json
import sys
import time
from pathlib import Path


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    load("sglang.srt.layers.moe.qwen38_expert_cache", "qwen38_expert_cache.py")
    adaptive = load("qwen38_analysis_allocator", "qwen38_cache_adaptive.py")
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        routes = request["routes"]
        warmup = min(128, len(routes[0]) // 4)
        # Every batch is contiguous, but batches may have an unobserved gap.
        # Reset all candidates together rather than pretending the gap was idle.
        allocator = adaptive.AdaptiveLayerAllocator(
            request["current"],
            request["num_experts"],
            min_slots=request["top_k"],
            window=len(routes[0]) - warmup,
            policy=request["policy"],
            protected_fraction=request["protected_fraction"],
            budget=request.get("budget"),
        )
        target = None
        for token in range(len(routes[0])):
            if token == warmup:
                # Keep warmed shadow residents, exclude cold-start misses.
                allocator.seen = [0] * allocator.num_layers
                allocator.misses = [
                    {c: 0 for c in allocator.candidates}
                    for _ in range(allocator.num_layers)
                ]
            for layer in range(len(routes)):
                if token < warmup:
                    active = sorted(set(routes[layer][token]))
                    for policy in allocator.policies[layer].values():
                        policy.ensure(active)
                    continue
                target = allocator.observe(layer, routes[layer][token]) or target
        print(
            json.dumps(
                {
                    "target": target,
                    "evaluation": allocator.last_evaluation,
                    "warmup_tokens": warmup,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
