#!/usr/bin/env python3
"""Capture per-token verify routes in an isolated, instrumented worker."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    trace = results / "verify-routes.jsonl"
    if trace.exists():
        parser.error("Refusing to overwrite trace")
    trace.touch()
    relative = Path("python/sglang/srt/layers/moe/qwen38_cache_threaded.py")
    source = (REPO / relative).read_text()
    needle = "            self._record_invocation(arena, topk_ids, layer_id)\n"
    assert source.count(needle) == 1
    instrumentation = """            if execution_phase == "verify" and 2 <= topk_ids.shape[0] <= 3:
                with open("/workspace/results/verify-routes.jsonl", "a") as verify_trace:
                    verify_trace.write(json.dumps(dict(
                        layer=layer_id, phase=execution_phase,
                        routes=topk_ids.detach().cpu().tolist()
                    )) + "\\n")
"""
    overlay = results / "overlay"
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.replace(needle, instrumentation + needle))
    spec = importlib.util.spec_from_file_location(
        "verify_ab", ROOT / "run_service_ab.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    original = bench.generate
    labels = iter(("warmup", "short", "raft", "long-0", "long-1"))

    def generate(prompt, tokens):
        case = next(labels)
        with trace.open("a") as output:
            output.write(json.dumps(dict(event="start", case=case)) + "\n")
        try:
            return original(prompt, tokens)
        finally:
            with trace.open("a") as output:
                output.write(json.dumps(dict(event="end", case=case)) + "\n")

    bench.generate = generate
    sys.argv = [
        str(ROOT / "run_service_ab.py"),
        "--results",
        str(results),
        "--variants",
        "defaults",
        "--tokens",
        str(args.tokens),
        "--runtime-overlay",
        str(overlay),
    ]
    bench.main()


if __name__ == "__main__":
    main()
