#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-/Users/jay/workspace/sglang}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jay/workspace/prod/common/sglang-metal/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e}"
DRAFT_MODEL="${DRAFT_MODEL:-/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-4}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0
exec "$PYTHON_BIN" -u -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30007}" \
  --context-length 8192 --max-total-tokens 17024 \
  --chunked-prefill-size 256 --max-running-requests 2 \
  --mlx-enable-sampling --random-seed 1234 \
  --enable-deterministic-inference --disable-cuda-graph \
  --disable-overlap-schedule \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE" "$@"
