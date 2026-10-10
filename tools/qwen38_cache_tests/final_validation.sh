#!/usr/bin/env bash
# Staged SM120 validation for the Qwen3.8-Flash-Next NVFP4 cache runtime.
# Run under the gpu-switch supervisor; each stage owns its watchdog container.
set -uo pipefail

stages_dir="${1:?usage: final_validation.sh <stages-dir>}"
mkdir -p "$stages_dir"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
image="swallow-sglang:v0.5.21"
device=0
revision="fc694b54fb0174e0913e6adf86691ef85a4ead47"
commit="e00930c5489053f26d86b179cee0d087f846acbb"
checkpoint="/home/user/workspace/prod/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/$revision"
model="tools/qwen38_cache_tests/docker_model.sh"
summary="$stages_dir/final-validation-summary.json"

overall="passed"
entries=""

stage() {
  local name="$1"; shift
  echo "[stage] $name start $(date +%T)"
  local start=$(date +%s)
  "$@" > "$stages_dir/$name.log" 2>&1
  local rc=$?
  local dur=$(( $(date +%s) - start ))
  if [[ $rc -ne 0 ]]; then overall="failed_$name"; fi
  entries="$entries{\"stage\":\"$name\",\"rc\":$rc,\"seconds\":$dur,\"log\":\"$name.log\"},"
  echo "[stage] $name rc=$rc seconds=$dur (overall=$overall)"
}

# Stale unmarked per-layer exports are cleared inside convert.py itself (the
# container user owns whatever prior jobs left on these pack volumes); --resume
# still honors provenance markers for untouched layers.

stage layer0_sync_parity bash "$model" \
  --image "$image" --device "$device" --checkpoint "$checkpoint" \
  --pack "$root/results/packs/layer0-sync" --results "$stages_dir/layer0-sync" -- convert \
  --checkpoint-revision "$revision" --sglang-commit "$commit" \
  --layers 0 --parity --runtime-mode sync
[[ $overall == passed ]] || exit 1

stage layer0_threaded_parity bash "$model" \
  --image "$image" --device "$device" --checkpoint "$checkpoint" \
  --pack "$root/results/packs/layer0-threaded" --results "$stages_dir/layer0-threaded" -- convert \
  --checkpoint-revision "$revision" --sglang-commit "$commit" \
  --layers 0 --parity --runtime-mode threaded
[[ $overall == passed ]] || exit 1

stage assemble_all_layers bash "$model" \
  --image "$image" --device "$device" --checkpoint "$checkpoint" \
  --pack "$root/results/packs/assembled" --results "$stages_dir/assemble" -- convert \
  --checkpoint-revision "$revision" --sglang-commit "$commit" --assemble
[[ $overall == passed ]] || exit 1

stage canary_threaded_8k_32k bash "$model" \
  --image "$image" --device "$device" --checkpoint "$checkpoint" \
  --pack "$root/results/packs/assembled" --results "$stages_dir/canary-threaded" -- canary \
  --runtime-mode threaded --contexts 8192 32768 \
  --probe 'What is 1 plus 1? Answer with only the number.|(?<![0-9])2(?![0-9])' \
  --mem-fraction-static 0.90 --mamba-full-memory-ratio 0.22 --max-mamba-cache-size 15
[[ $overall == passed ]] || exit 1

printf '{"status":"%s","stages":[%s]}\n' "$overall" "${entries%,}" > "$summary"
echo "FINAL: $overall -> $summary"
[[ $overall == passed ]]
