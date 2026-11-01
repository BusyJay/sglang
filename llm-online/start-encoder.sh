#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-/Users/jay/workspace/sglang}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jay/workspace/prod/common/sglang-metal/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0
exec "$PYTHON_BIN" -u -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --host "${HOST:-127.0.0.1}" \
  --port "${ENCODER_PORT:-30008}" --encoder-only \
  --encoder-transfer-backend zmq_to_tokenizer \
  --disable-cuda-graph --disable-overlap-schedule "$@"
