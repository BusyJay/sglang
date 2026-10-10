# Offline Hugging Face model-ID support

The cache loader now accepts a checkpoint directory **or** a Hugging Face repo ID.
Repo IDs resolve with `snapshot_download(local_files_only=True)`, using the normal
HF cache location or `--download-dir`. No download, cache cleanup or online fallback
is attempted by this loader. Existing symlinked snapshots remain supported.

The user's `model_path` and `revision` are not rewritten. Config and tokenizer
continue through the standard SGLang/Transformers startup path. As in the deployment,
set both offline variables before starting the process:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/huggingface/hub
python3 -m sglang.launch_server \
  --model-path nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --reasoning-parser qwen3 \
  --load-format qwen38_nvfp4_cache \
  --model-loader-extra-config '{"manifest_path":"/packs/manifest.json","runtime_mode":"threaded"}' \
  --language-model-only --ple-offload-embedding \
  --disable-cuda-graph --disable-shared-experts-fusion \
  --moe-runner-backend flashinfer_cutlass
```

Keep the deployment's existing PLE/memory/context flags; this example only shows
identity/cache options and required offload switches. Omitting `--revision` uses
the usual cached `main` ref, not an arbitrarily chosen snapshot. Config's known
commit hash, resolved snapshot hash and pack revision must agree when verifiable.
If the pack corresponds to a different revision, specify that revision explicitly
for config/tokenizer/weights. Arbitrary local directories cannot always prove their
revision. Offline mode for config/tokenizer is controlled by the normal upstream
environment, not a global flag mutation inside this custom loader.

The loader validates indexed shard existence (including broken blob symlinks)
before loading. It does not inspect/download all tensor bytes at resolution time.

## Updated tests and entry points

- `full_model_canary.py --checkpoint REPO_ID --revision HASH --cache-dir HUB_DIR`:
  resolves locally for preflight/tokenizer, passes the original repo ID to the
  server, uses it as default served name, adds `--reasoning-parser qwen3`.
- `convert.py --checkpoint REPO_ID --checkpoint-revision HASH --cache-dir HUB_DIR`:
  resolves the existing snapshot locally before conversion.
- Docker wrapper supports `--checkpoint REPO_ID --hf-cache HOST_HUB_DIR` and mounts
  the entire hub cache read-only at `/workspace/hf-cache`. Networking remains off.
  Preserve normal repo `refs/snapshots/blobs` structure under that cache. Symlinks
  into a pool outside the mounted tree require an additional deployment mount.
- `final_validation.sh` uses a repo ID by default and passes the same revision to
  conversion and canary. Override `QWEN38_CHECKPOINT` / `HF_HUB_CACHE` for the host.

Example Docker canary:

```bash
bash tools/qwen38_cache_tests/docker_model.sh \
  --image YOUR_LOCAL_IMAGE --device 0 \
  --checkpoint nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --hf-cache /huggingface/hub --pack /packs/assembled --results /results/offline-id \
  -- canary --runtime-mode threaded \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --contexts 8192 32768
```

## Local validation

CPU suite: 93 tests, 80 passed, 13 skipped (PyTorch unavailable). Mocked tests
verify local-only HF calls, revision/cache arguments, missing snapshots, incomplete
shards, broken symlinks, revision mismatches, and actual Docker/canary argument
construction. Docker/GPU/network were not used. Shell syntax, Python AST and diff
checks passed. Actual cached repo-ID GPU startup remains to be tested on the target.
