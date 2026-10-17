#!/usr/bin/env bash
# Shared runtime settings. Edit here, then docker compose restart llm vit.
export HF_HOME=/prod/huggingface
export HF_HUB_OFFLINE=True
export TZ=Asia/Shanghai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS=ignore::UserWarning:vllm.model_executor.layers.fla.ops
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONDONTWRITEBYTECODE=1
export VIT_ENDPOINT=vit:8000
export QWEN38_WARMUP=qwen38_first_prefill
