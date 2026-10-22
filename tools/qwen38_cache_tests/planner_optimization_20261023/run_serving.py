"""Paired serving runs with fixed configuration and an isolated pre-change checkout."""

import argparse, json, os, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--before", action="store_true")
p.add_argument("--n", type=int, default=1)
p.add_argument("--gpu", action="store_true")
p.add_argument("--short", action="store_true")
p.add_argument("--no-spec", action="store_true")
a = p.parse_args()
r = Path("/results")
repo = Path("/before" if a.before else "/workspace/sglang")
out = r / a.name
if (out / "performance.json").exists():
    raise RuntimeError("refusing to replace measurements")
out.mkdir(exist_ok=True)
cmd = json.loads(
    Path(
        "/workspace/sglang/tools/qwen38_cache_tests/strata_review_20261022/base-command.json"
    ).read_text()
)["command"]
i = cmd.index("--model-loader-extra-config") + 1
cfg = json.loads(cmd[i])
cfg["cpu_hybrid_mode"] = "off" if a.gpu else "layered_lfu"
cfg["stats_log_interval"] = 19200
if not a.gpu:
    cfg["layered_cpu_max_experts"] = a.n
cmd[i] = json.dumps(cfg)
if a.no_spec:
    for flag in [
        "--speculative-algorithm",
        "--speculative-draft-load-format",
        "--speculative-num-steps",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
    ]:
        j = cmd.index(flag)
        del cmd[j : j + 2]
ref = out / "command.json"
ref.write_text(json.dumps({"command": cmd}, indent=2))
e = dict(
    os.environ,
    PYTHONPATH=str(repo / "python"),
    SGLANG_ENABLE_QWEN38_CPU_HYBRID="0",
    SGLANG_QWEN38_CPU_CORES="2,3,4,5",
    SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH="1",
    SGLANG_QWEN38_FILL_CTA="adaptive",
    SGLANG_QWEN38_STARTUP_CACHE="auto",
    SGLANG_QWEN38_SOURCE_CACHE_DIR="/dev/shm/sglang-qwen38-startup",
    SGLANG_QWEN38_HOST_TOKEN_EMBEDDING="1",
    SGLANG_QWEN38_BORROW_PREFILL_ARENA="1",
    SGLANG_QWEN38_PLE_PACKED_CACHE="1",
    SGLANG_QWEN38_CACHE_W4A16="1",
    SGLANG_QWEN38_CACHE_W4A16_BACKEND="b12x",
    SGLANG_QWEN38_CACHE_PREFILL_W4A16="0",
    SGLANG_QWEN38_FP32_NORM_FUSION="1",
    SGLANG_QWEN38_CACHE_W4A16_SPLIT_K="1",
)
args = [
    "python3",
    str(
        repo
        / "tools/qwen38_cache_tests/global_cpu_resize_v2_20261020/benchmark_performance.py"
    ),
    "--reference",
    str(ref),
    "--output",
    str(out),
    "--modes",
    "default",
]
if a.short:
    args.append("--short-only")
subprocess.run(args, env=e, cwd=repo, check=True)
