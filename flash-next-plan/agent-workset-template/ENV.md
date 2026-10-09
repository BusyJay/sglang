# Environment

Last updated: 2026-10-09

## Source

- Repository: `<path or URL>`
- Branch: `<branch>`
- Commit: `<commit>`
- Docker image: `<v0.5.21-derived image tag or digest>`

## Runtime

- Python: `<version>`
- PyTorch: `<version>`
- CUDA: `<version>`
- FlashInfer: `<version>`
- NVIDIA driver: `<version>`

## Hardware

- Primary GPU: RTX 5090 32GB
- Secondary GPU: <optional GPU>
- Daily small-test GPU: <optional GPU>
- Daily GPU-test budget: 8GiB target, 10GiB maximum
- Host memory: `<actual MemTotal>`
- Storage: `<device/filesystem>`

## Models

- Development model: `<actual regular SGLang model>`
- Test model: `nvidia/Qwen3.8-Flash-Next-NVFP4`
- Revisions: `<pinned revisions>`

## Paths

- Model cache: `<path>`
- Expert pack: `<path>`
- Test output: `<path>`

## Commands

### Development backend

```bash
<command>
```

### Test backend

```bash
<command>
```

### Small GPU tests

```bash
bash tools/qwen38_cache_tests/docker_gpu.sh --image "$SGLANG_IMAGE" --device "$QWEN38_GPU_DEVICE" --results /absolute/results/gpu-small
```
