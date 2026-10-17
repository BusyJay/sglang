ARG BASE_IMAGE=lmsysorg/sglang:v0.5.21
FROM ${BASE_IMAGE}
COPY python/sglang /workspace/sglang/python/sglang
COPY tools/qwen38_cache_tests /workspace/sglang/tools/qwen38_cache_tests
COPY flash-next-plan /workspace/sglang/flash-next-plan
ENV PYTHONPATH=/workspace/sglang/python
WORKDIR /workspace/sglang
LABEL org.sglang.qwen38.base-commit="0d01fb2dc8e4c0d861fb16f5a77537936e598000"
