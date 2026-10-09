#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEV_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$DEV_ROOT/../.." && pwd)"

ENV_FILE="${QWEN38_DEV_ENV:-$DEV_ROOT/env.local}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  echo "Copy $DEV_ROOT/env.example to $DEV_ROOT/env.local" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${SGLANG_IMAGE:?SGLANG_IMAGE is required}"
: "${RESULTS_HOST:?RESULTS_HOST is required}"

mkdir -p "$RESULTS_HOST"

docker_base=(
  docker run --rm
  --ipc=host
  --ulimit memlock=-1
  -v "$REPO_ROOT:/workspace/sglang"
  -v "$RESULTS_HOST:/workspace/results"
  -w /workspace/sglang
  -e PYTHONPATH=/workspace/sglang/python
  --entrypoint ""
)

if [[ -n "${DOCKER_EXTRA_ARGS:-}" ]]; then
  # Local, user-controlled flags only. Avoid putting secrets here.
  read -r -a extra <<<"$DOCKER_EXTRA_ARGS"
  docker_base+=("${extra[@]}")
fi
