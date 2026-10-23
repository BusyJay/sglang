"""Serial full-model CSF/native comparisons on GPU 0; run inside the test container."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/results")
REPO = Path("/workspace/sglang")
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
CACHE = Path("/root/.cache/sglang")
BASE = REPO / "tools/qwen38_cache_tests/strata_review_20261022/base-command.json"


def run(
    name,
    encoding,
    route,
    full=False,
    no_spec=False,
    smoke=False,
    no_graph=False,
    static_cache=False,
):
    out = ROOT / name
    out.mkdir(exist_ok=False)
    command = json.loads(BASE.read_text())["command"]
    index = command.index("--model-loader-extra-config") + 1
    cfg = json.loads(command[index])
    native = json.loads(
        (CACHE / "qwen38-fp4-local" / REV / "manifest.json").read_text()
    )
    # Six transferred planes; per-layer input scales stay outside the cache pool.
    names = [
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
        "w2_blockscale_swizzled",
        "g1_alphas",
        "g2_alphas",
    ]
    slot_bytes = sum(native["layers"]["0"][n]["nbytes"] for n in names) // 512
    cfg.pop("cache_slots_total")
    cfg["cache_budget_bytes"] = slot_bytes * 6388
    cfg["scale_compression"] = encoding
    cfg["manifest_path"] = str(
        CACHE
        / ("qwen38-fp4-local-csf-tiered" if encoding == "csf" else "qwen38-fp4-local")
        / REV
        / "manifest.json"
    )
    cfg["cpu_hybrid_mode"] = "off" if route == "off" else "layered_lfu"
    cfg["stats_log_interval"] = 5760
    if route != "off":
        cfg["layered_cpu_max_experts"] = int(route[1:])
    if static_cache:
        cfg.pop("kv_cache_stages", None)
        cfg["adaptive_cache"] = False
    command[index] = json.dumps(cfg)
    if no_spec:
        for flag in (
            "--speculative-algorithm",
            "--speculative-draft-load-format",
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
        ):
            j = command.index(flag)
            del command[j : j + 2]
    if no_graph:
        command[command.index("--cuda-graph-backend-decode=breakable")] = (
            "--cuda-graph-backend-decode=disabled"
        )
    reference = out / "command.json"
    reference.write_text(json.dumps({"command": command}, indent=2) + "\n")
    env = dict(
        os.environ,
        PYTHONPATH=str(REPO / "python"),
        SGLANG_ENABLE_QWEN38_CPU_HYBRID="0",
        SGLANG_QWEN38_CPU_CORES="2,3,4,5",
        SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH="1",
        SGLANG_QWEN38_FILL_CTA="adaptive",
        SGLANG_QWEN38_STARTUP_CACHE="auto",
        SGLANG_QWEN38_SOURCE_CACHE_DIR="/dev/shm/sglang-qwen38-startup",
        SGLANG_QWEN38_HOST_TOKEN_EMBEDDING="1",
        SGLANG_QWEN38_BORROW_PREFILL_ARENA="0" if static_cache else "1",
        SGLANG_QWEN38_PLE_PACKED_CACHE="1",
        SGLANG_QWEN38_CACHE_W4A16="1",
        SGLANG_QWEN38_CACHE_W4A16_BACKEND=os.getenv(
            "SGLANG_QWEN38_CACHE_W4A16_BACKEND", "b12x"
        ),
        SGLANG_QWEN38_CACHE_PREFILL_W4A16=os.getenv(
            "SGLANG_QWEN38_CACHE_PREFILL_W4A16", "0"
        ),
        SGLANG_QWEN38_FP32_NORM_FUSION="1",
        SGLANG_QWEN38_CACHE_W4A16_SPLIT_K="1",
    )
    (out / "environment.json").write_text(
        json.dumps({k: v for k, v in env.items() if k.startswith("SGLANG_")}, indent=2)
        + "\n"
    )
    args = [
        "python3",
        str(ROOT / "benchmark_checked.py"),
        "--reference",
        str(reference),
        "--output",
        str(out),
        "--modes",
        "default",
    ]
    if smoke:
        args.append("--warmup-only")
    if not full:
        args.append("--short-only")
    print(name, "starting", flush=True)
    subprocess.run(args, env=env, check=True)
    print(name, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--encoding", choices=["native", "csf"], required=True)
    parser.add_argument("--route", choices=["off", "n0", "n1"], required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--static-cache", action="store_true")
    a = parser.parse_args()
    run(
        a.name,
        a.encoding,
        a.route,
        a.full,
        a.no_spec,
        a.smoke,
        a.no_graph,
        a.static_cache,
    )
