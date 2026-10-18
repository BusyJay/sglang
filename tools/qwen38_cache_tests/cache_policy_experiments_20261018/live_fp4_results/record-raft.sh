#!/usr/bin/env bash
set -euo pipefail
rootdir=/prod/homelab-llm/home-jay/remote-mapping/llm
source "$rootdir/sglang-env.sh"
# Runtime tracing is opt-in. The service API can capture CPU/GPU traces at any time.
export SGLANG_QWEN38_OBSERVE=${SGLANG_QWEN38_OBSERVE:-0}
export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-/prod/homelab-llm/models/profiles}
if [[ "${1:-}" == "--profile-start" || "${1:-}" == "--profile-stop" ]]; then
  action=${1#--profile-}
  shift
  exec python3 "$rootdir/sglang-profile.py" "$action" --url "${PROFILE_URL:-http://127.0.0.1:8000}" "$@"
fi
decode_w4a16=1
while (( $# )); do
  case "$1" in
    --decode-a4) decode_w4a16=0; shift ;;
    --mtp)
      [[ $# -ge 2 ]] || { echo "--mtp requires off, 1, 2 or 3" >&2; exit 2; }
      export MTP_MODE="$2"; shift 2 ;;
    --profile-nvtx)
      export SGLANG_QWEN38_OBSERVE=1
      export SGLANG_QWEN38_OBSERVE_NVTX=1
      shift ;;
    --print-launch-config) break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
# Runtime settings live here; edit these literals, then restart homelab-llm.
# Explicit exports override stale environment values in an existing container.
export SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH="1"
export SGLANG_QWEN38_FILL_CTA="adaptive"
export SGLANG_QWEN38_STARTUP_CACHE="auto"
export SGLANG_QWEN38_SOURCE_CACHE_DIR="/dev/shm/sglang-qwen38-startup"
export SGLANG_QWEN38_HOST_TOKEN_EMBEDDING="1"
export SGLANG_QWEN38_BORROW_PREFILL_ARENA="1"
export SGLANG_QWEN38_PLE_PACKED_CACHE="1"

# All FP4-Local presets share the configuration below. Only decode precision differs.
# Explicit values override the old container environment on ordinary restart.
export MODEL=nvidia/Qwen3.8-Flash-Next-FP4-Local
export CHECKPOINT_DIR=nvidia/Qwen3.8-Flash-Next-FP4-Local
export CHECKPOINT_REVISION=b9ce18a6887e18c089fdea8509a33e63429a3300
export PACK_DIR=/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300
export MANIFEST_PATH=/root/.cache/sglang/qwen38-fp4-local/b9ce18a6887e18c089fdea8509a33e63429a3300/manifest.json
export PACK_SGLANG_COMMIT=acd9c47fed94afe9c753631e4f88df58ca3f676f+fp4-local-59db23b7bae46aa9
export PACK_AUTO_BUILD=0
export CACHE_SLOTS_TOTAL=6390
# Match the original NVFP4 KV capacity. Keep packed-weight savings available
# for 8192-token GDN prefill scratch instead of doubling the static KV pool.
#export MAX_TOTAL_TOKENS=221952
# Keep MTP enabled; verify transport is optimized in the runtime.
export MTP_MODE=${MTP_MODE:-2}
case "$MTP_MODE" in
  off|none|0|1|2|3) ;;
  *) echo "MTP_MODE must be off, 1, 2 or 3" >&2; exit 2 ;;
esac
# MTP retains target HC states during prefill; reserve dynamic scratch for
# 8192-token PLE/GDN and draft-extend instead of exhausting the static budget.
export MEM_FRACTION_STATIC=0.90
export PLE_OFFLOAD_BACKEND=pinned
export SGLANG_QWEN38_CACHE_W4A16="$decode_w4a16"
# SM12x native ModelOpt W4A16 for 1..3-token decode/verify.
if (( decode_w4a16 )); then
  export SGLANG_QWEN38_CACHE_W4A16_BACKEND=${SGLANG_QWEN38_CACHE_W4A16_BACKEND:-b12x}
else
  export SGLANG_QWEN38_CACHE_W4A16_BACKEND=triton
fi
# Prefill: CUTLASS W4A4; decode/verify follows CACHE_W4A16 above.
export SGLANG_QWEN38_CACHE_PREFILL_W4A16=0
export SGLANG_QWEN38_FP32_NORM_FUSION=1
# Both precision presets use the same launch settings and log destination.
export SGLANG_LOG_FILE="/results/raft-server.log"
# Measured SM120 small-verify winner; allow explicit service A/B overrides.
export SGLANG_QWEN38_CACHE_W4A16_SPLIT_K=${SGLANG_QWEN38_CACHE_W4A16_SPLIT_K:-1}

# Read-only metadata for llmctl preflight; never imports SGLang or starts a server.
if [[ "${1:-}" == "--print-launch-config" ]]; then
  python3 -c 'import json,os; keys=("MODEL","CHECKPOINT_DIR","CHECKPOINT_REVISION","PACK_DIR","MANIFEST_PATH","PACK_SGLANG_COMMIT","SGLANG_LOG_FILE","SGLANG_QWEN38_CACHE_W4A16","SGLANG_QWEN38_CACHE_W4A16_BACKEND","SGLANG_QWEN38_CACHE_PREFILL_W4A16","SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH","SGLANG_QWEN38_FILL_CTA","SGLANG_QWEN38_STARTUP_CACHE","SGLANG_QWEN38_SOURCE_CACHE_DIR","SGLANG_QWEN38_HOST_TOKEN_EMBEDDING","SGLANG_QWEN38_BORROW_PREFILL_ARENA","SGLANG_QWEN38_PLE_PACKED_CACHE","CACHE_SLOTS_TOTAL","MTP_MODE","PLE_OFFLOAD_BACKEND","MEM_FRACTION_STATIC","SGLANG_QWEN38_OBSERVE","SGLANG_QWEN38_OBSERVE_NVTX","SGLANG_TORCH_PROFILER_DIR"); print(json.dumps({k:os.environ[k] for k in keys if k in os.environ}))'
  exit 0
fi

export QWEN38_ROUTE_RECORD_PATH=/results/raft.jsonl
export QWEN38_ROUTE_RECORD_ARM=/results/raft.armed
export QWEN38_ROUTE_RECORD_SWEEPS=1024
export PYTHONPATH=/workspace/sglang/tools/qwen38_cache_tests/cache_policy_experiments_20261018/record_overlay:/workspace/sglang:/workspace/sglang/python
export QWEN38_WARMUP=none
export CUDA_VISIBLE_DEVICES=0
export PORT=8000
export HOST=0.0.0.0

exec bash "$rootdir/sglang-flash-next-common.sh" "$@"
