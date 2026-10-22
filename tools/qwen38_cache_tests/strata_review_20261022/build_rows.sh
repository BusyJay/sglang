#!/usr/bin/env bash
set -euo pipefail
# Run inside the diagnostic image with both repositories mounted read-only.
g++ -O3 -std=c++20 -march=native -fPIC -shared -pthread \
  -I/workspace/Strata/include -I/workspace/Strata/third_party/ggml \
  /results/row_bridge.cpp /workspace/Strata/src/kernels/cpu/nvfp4_avx512.cpp \
  -o /results/row_bridge.so
