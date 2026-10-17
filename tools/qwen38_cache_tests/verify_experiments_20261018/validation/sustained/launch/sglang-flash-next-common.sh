#!/usr/bin/env bash
set -euo pipefail

binpath=$(readlink -f "$0")
rootdir=$(dirname "$binpath")

MODEL=${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-fc694b54fb0174e0913e6adf86691ef85a4ead47}
HF_CACHE_DIR=${HF_CACHE_DIR:-${HF_HUB_CACHE:-/prod/huggingface/hub}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$MODEL}
PACK_ROOT=${PACK_ROOT:-/root/.cache/sglang/qwen38-flash-next}
PACK_DIR=${PACK_DIR:-$PACK_ROOT/$CHECKPOINT_REVISION}
MANIFEST_PATH=${MANIFEST_PATH:-$PACK_DIR/manifest.json}
SGLANG_SRC=${SGLANG_SRC:-/workspace/sglang}
PACK_AUTO_BUILD=${PACK_AUTO_BUILD:-1}
PACK_RUNTIME_MODE=${PACK_RUNTIME_MODE:-threaded}
PACK_SGLANG_COMMIT=${PACK_SGLANG_COMMIT:-${SGLANG_BUILD_COMMIT:-unknown}}

# Total expert slots across all 48 layers (5040 = 105 per layer).
CACHE_SLOTS_TOTAL=${CACHE_SLOTS_TOTAL:-5140}
# MTP/NEXTN speculative decoding: OFF by default (Oct-15 rollback).
# Evidence: accept len med ~1.8-1.9 across every real sample ever logged; bench
# upside was +5..13% while decode sat at target parity, and turning any steps on
# forces mamba_ssm_dtype=bfloat16 (shared GDN state pool + verify stash math).
# Set MTP_MODE=1|2|3 to re-enable (draft tokens follow automatically).
MTP_MODE=${MTP_MODE:-off}
SPEC_DRAFT_KV_DTYPE=${SPEC_DRAFT_KV_DTYPE:-fp8_e4m3}
ADAPTIVE_CACHE=${ADAPTIVE_CACHE:-true}
PREALLOCATE_GDN_PROJ=${PREALLOCATE_GDN_PROJ:-true}
QWEN38_PACKED_DECODE=${QWEN38_PACKED_DECODE:-packed}
QWEN38_PACKED_PREFILL=${QWEN38_PACKED_PREFILL:-packed}
DECODE_LOG_INTERVAL=${DECODE_LOG_INTERVAL:-400}
RUNTIME_MODE=${RUNTIME_MODE:-threaded}
CACHE_STATS_LOG_INTERVAL=${CACHE_STATS_LOG_INTERVAL:-19200}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-220000}
# NOTE: chunk=1024 was the bench harness value, not a production choice.
# Historical caution: frac 0.91/0.92 + chunk 8192 hit silu() transient OOM twice
# (Oct-14 morning); raise MEM_FRACTION_STATIC cautiously only if headroom proves.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.910}
MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-0.22}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-10}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
PLE_OFFLOAD_BACKEND=${PLE_OFFLOAD_BACKEND:-lazy}
QWEN38_WARMUP=${QWEN38_WARMUP:-qwen38_first_prefill}
# HiCache：GPU KV 换出到主机内存层（旧 27B 栈验证过 ratio 6 ≈ host 23GB）。
# breakable decode graph 的 incompatible 规则表(cuda_graph_hook.py disable_breakable_*)
# 不含 hierarchical cache；该规则只砍 tc_piecewise/prefill，对本配置无图副作用。
# 不开 file 后端目录（sglang.sh 老注释：idle ongoing_backup busy-loop 坑）。
ENABLE_HIERARCHICAL_CACHE=${ENABLE_HIERARCHICAL_CACHE:-1}
HICACHE_RATIO=${HICACHE_RATIO:-6}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# The FP4 runtime supports remote ViT embeddings with same-device MTP.
if [[ -n "${CACHE_SLOTS:-}" ]]; then
  echo "CACHE_SLOTS was per-layer; use CACHE_SLOTS_TOTAL (default 5040)" >&2
  exit 2
fi
case "$MTP_MODE" in
  off|none|0) MTP_MODE=off ;;
  1|2|3) ;;
  *) echo "MTP_MODE must be off, 1, 2 or 3" >&2; exit 2 ;;
esac

# MTP forces a bf16 GDN state pool (single shared target+draft pool, verify
# stashes draft_token_num states per step). Pre-MTP baseline ran fp32.
if [[ "$MTP_MODE" == "off" ]]; then
  MAMBA_SSM_DTYPE=float32
else
  MAMBA_SSM_DTYPE=bfloat16
fi

if [[ ! -d "$SGLANG_SRC/python/sglang" ]]; then
  echo "Missing SGLang source mount: $SGLANG_SRC/python/sglang" >&2
  exit 2
fi
export PYTHONPATH="$SGLANG_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
echo "Using SGLang source: $(readlink -f -- "$SGLANG_SRC")" >&2

ensure_pack() {
  if [[ -f "$MANIFEST_PATH" ]]; then
    echo "Using existing expert pack: $MANIFEST_PATH" >&2
    return 0
  fi
  if [[ "$PACK_AUTO_BUILD" != "1" ]]; then
    echo "missing expert-pack manifest: $MANIFEST_PATH" >&2
    echo "set PACK_AUTO_BUILD=1 or prebuild the pack under $PACK_DIR" >&2
    return 1
  fi
  mkdir -p "$PACK_DIR"
  echo "Building expert pack into $PACK_DIR (first start only)..." >&2
  python3 -m sglang.srt.model_loader.qwen38_nvfp4_pack.convert \
    --checkpoint "$CHECKPOINT_DIR" \
    --cache-dir "$HF_CACHE_DIR" \
    --output "$PACK_DIR" \
    --checkpoint-revision "$CHECKPOINT_REVISION" \
    --sglang-commit "$PACK_SGLANG_COMMIT" \
    --assemble \
    --resume \
    --runtime-mode "$PACK_RUNTIME_MODE"
}

ensure_pack

MODEL_LOADER_EXTRA_CONFIG=$(cat <<JSON
{"manifest_path":"$MANIFEST_PATH","checkpoint_revision":"$CHECKPOINT_REVISION","sglang_commit":"$PACK_SGLANG_COMMIT","cache_slots_total":$CACHE_SLOTS_TOTAL,"prefill_arena":true,"runtime_mode":"$RUNTIME_MODE","stats_log_interval":$CACHE_STATS_LOG_INTERVAL,"adaptive_cache":$ADAPTIVE_CACHE,"preallocate_gdn_proj":$PREALLOCATE_GDN_PROJ,"async_pack_prefetch":true,"packed_decode":"$QWEN38_PACKED_DECODE","packed_prefill":"$QWEN38_PACKED_PREFILL"}
JSON
)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_CACHE_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0
export SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=${SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND:-python}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((256 * 1024 * 1024))
export SGLANG_QWEN4_PLE_LAZY_CACHE_MB=3072
#export SGLANG_QWEN38_STARTUP_MEMORY_DIR=$rootdir/startup-memory
#export SGLANG_QWEN38_ELASTIC_L2=1
#export SGLANG_QWEN38_FILL_CTA=adaptive
export CUDA_DEVICE_ORDER=PCI_BUS_ID

token_pool_flags=()
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  token_pool_flags+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

hicache_flags=()
if [[ "$ENABLE_HIERARCHICAL_CACHE" == "1" ]]; then
  hicache_flags+=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO"
    --hicache-mem-layout page_first_direct --hicache-io-backend direct
    --hicache-write-policy write_through)
fi

mtp_flags=()
if [[ "$MTP_MODE" != "off" ]]; then
  mtp_flags+=(--speculative-algorithm NEXTN
    --speculative-draft-load-format qwen38_mtp
    --speculative-draft-kv-cache-dtype "$SPEC_DRAFT_KV_DTYPE"
    --speculative-num-steps "$MTP_MODE"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$((MTP_MODE + 1))")
fi

warmup_flags=(--warmups "$QWEN38_WARMUP")
if [[ "$QWEN38_WARMUP" == "none" ]]; then
  warmup_flags=(--skip-server-warmup)
fi

sglang serve \
  --random-seed 103 \
  "${warmup_flags[@]}" \
  ${token_pool_flags[@]+"${token_pool_flags[@]}"} \
  ${hicache_flags[@]+"${hicache_flags[@]}"} \
  ${mtp_flags[@]+"${mtp_flags[@]}"} \
  --trust-remote-code \
  --model-path "$CHECKPOINT_DIR" \
  --served-model-name "$MODEL" \
  --revision "$CHECKPOINT_REVISION" \
  --download-dir "$HF_CACHE_DIR" \
  --load-format qwen38_nvfp4_cache \
  --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
  --language-only \
  --ple-offload-embedding \
  --ple-offload-backend "$PLE_OFFLOAD_BACKEND" \
  --disable-shared-experts-fusion \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --cuda-graph-backend-prefill=disabled \
  --cuda-graph-backend-decode=breakable \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-max-bs-decode 1 \
  --enable-cache-report \
  --decode-log-interval "$DECODE_LOG_INTERVAL" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sleep-on-idle \
  --encoder-urls http://$VIT_ENDPOINT \
  --host "$HOST" \
  --port "$PORT" \
  2>&1 | tee -a "${SGLANG_LOG_FILE:-$rootdir/sglang-flash-next.log}"
