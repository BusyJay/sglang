"""Profile the transport replay after its common warm prefix; never timing data."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    from sglang.srt.layers.moe.qwen38_cache_threaded import Qwen38ThreadedExpertRuntime
    from tools.qwen38_cache_tests.cache_transport_replay import worker

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", choices=["lfu", "lru"], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        record_shapes=False,
    )
    original = Qwen38ThreadedExpertRuntime.stats_snapshot
    original_run = Qwen38ThreadedExpertRuntime.run
    snapshots = 0

    def snapshot(runtime):
        nonlocal snapshots
        snapshots += 1
        if snapshots == 2:
            profile.stop()
            profile.export_chrome_trace(
                str(args.output / f"{args.policy}-decode.trace.json")
            )
        value = original(runtime)
        if snapshots == 1:
            profile.start()
        return value

    Qwen38ThreadedExpertRuntime.stats_snapshot = snapshot

    def scoped_run(runtime, layer, *values, **kwargs):
        with torch.profiler.record_function(f"transport/{args.policy}/layer_{layer}"):
            return original_run(runtime, layer, *values, **kwargs)

    Qwen38ThreadedExpertRuntime.run = scoped_run
    result = args.output / f"{args.policy}-profile-only.json"
    worker(
        SimpleNamespace(
            trace=args.trace,
            manifest=args.manifest,
            case=None,
            warmup=128,
            layout=None,
            budget=6390,
            policy=args.policy,
            protected_fraction=0.7,
            correctness=False,
            correctness_sweeps=16,
            output=result,
        )
    )
    if snapshots < 2:
        raise RuntimeError("Profiler did not capture the replay interval")
    data = json.loads(result.read_text())
    data["mode"] = "profile_only"
    data["limits"].append("Profiled wall time is excluded from all timing summaries")
    result.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
