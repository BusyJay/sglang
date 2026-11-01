#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/start.sh" \
  --language-only \
  --encoder-urls "${ENCODER_URL:-http://127.0.0.1:30008}" \
  --encoder-transfer-backend zmq_to_tokenizer "$@"
