#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry; edit all settings in sglang-fp4-local.sh.
exec bash "$(dirname -- "$(readlink -f -- "$0")")/sglang-fp4-local.sh" --decode-a4 "$@"
