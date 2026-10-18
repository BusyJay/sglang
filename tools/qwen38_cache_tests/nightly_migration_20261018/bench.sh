#!/bin/bash
set -euo pipefail
out=$1
for i in $(seq 1 240); do
  if curl -fsS --max-time 2 http://127.0.0.1:8329/health >/dev/null 2>&1; then break; fi
  sleep 2
done
python3 tools/qwen38_cache_tests/bench_mtp_service.py --stream --tokens 192 --case 0 --output "$out-warmup.json"
for r in 1 2; do
  python3 tools/qwen38_cache_tests/bench_mtp_service.py --stream --tokens 1024 --output "$out-round$r.json"
done
