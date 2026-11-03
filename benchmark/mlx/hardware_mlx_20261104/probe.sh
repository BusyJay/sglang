#!/bin/sh
set -eu
cd /Users/jay/workspace/sglang
out=benchmark/mlx/hardware_mlx_20261104
xcrun clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal "$out/device_capabilities.mm" -o "$out/device_capabilities"
"$out/device_capabilities" > "$out/device_capabilities.json"
xcrun xctrace list templates > "$out/xctrace_templates.txt" 2>&1
xcrun xctrace record --template 'Metal System Trace' --show-recording-options > "$out/metal_recording_options.json" 2> "$out/options.stderr"
# Metadata only: no allocation of model weights, GPU compute dispatch or timing.
