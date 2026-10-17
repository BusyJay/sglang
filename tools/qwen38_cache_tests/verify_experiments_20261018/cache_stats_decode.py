#!/usr/bin/env python3
"""Expand compact cache stats from JSON or service-log lines to original JSON."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "stats_format",
        repo / "python/sglang/srt/layers/moe/qwen38_cache_stats_format.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for line in args.input.read_text().splitlines():
        if "Qwen38ExpertCacheStats[" in line:
            line = line[line.index("{", line.index("Qwen38ExpertCacheStats[")) :]
        elif not line.startswith("{"):
            continue
        rows.append(module.expand_stats(json.loads(line)))
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
