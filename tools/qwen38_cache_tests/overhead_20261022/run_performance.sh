#!/usr/bin/env bash
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
export QWEN38_REFERENCE=${QWEN38_REFERENCE:-/workspace/sglang/tools/qwen38_cache_tests/overhead_20261022/command-no-stats.json}
export QWEN38_CALIBRATION=${QWEN38_CALIBRATION:-/workspace/sglang/tools/qwen38_cache_tests/overhead_20261022/calibration-final.json}
export QWEN38_VERIFY_RESULTS=${QWEN38_VERIFY_RESULTS:-$repo_root/tools/qwen38_cache_tests/overhead_20261022/performance-replay}
exec bash "$repo_root/tools/qwen38_cache_tests/migration_20261021/run_hybrid.sh" "$@"
