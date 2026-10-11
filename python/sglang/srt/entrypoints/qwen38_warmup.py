# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in warmup profile for single-rank Flash-Next offload."""

import logging
import time

logger = logging.getLogger(__name__)


async def warmup_qwen38_offload(disaggregation_mode, tokenizer_manager):
    """Exercise arena prefill and sustained decode before accepting traffic.

    Uses actual tokenizer IDs, not arbitrary IDs that may exceed the vocabulary.
    This pays representative cold-path costs, not a guarantee of cache hits or
    compilation coverage for all lengths, requests, or tool schemas.
    """
    await _warmup_qwen38_prefill(
        disaggregation_mode, tokenizer_manager, (64, 512, 2048), 16
    )


async def warmup_qwen38_first_prefill(disaggregation_mode, tokenizer_manager):
    """Warm the full first block and the QSA chunk-prefill continuation."""
    await _warmup_qwen38_prefill(
        disaggregation_mode, tokenizer_manager, (4224,), 1
    )


async def _warmup_qwen38_prefill(
    disaggregation_mode, tokenizer_manager, sizes, decode_tokens
):
    if disaggregation_mode != "null":
        raise ValueError("qwen38_offload warmup requires non-disaggregated serving")
    tokenizer = tokenizer_manager.tokenizer
    if tokenizer is None:
        raise ValueError("qwen38_offload warmup requires tokenizer initialization")
    from sglang.srt.managers.io_struct import GenerateReqInput

    phrase = tokenizer.encode(
        "Explain briefly how an expert cache handles a miss. ",
        add_special_tokens=False,
    )
    if not phrase:
        raise ValueError("qwen38_offload warmup encoded an empty prompt")
    context_length = tokenizer_manager.model_config.context_len
    decode_tokens = min(decode_tokens, context_length - 1)
    if decode_tokens < 1:
        raise ValueError("qwen38_offload warmup requires context length >= 2")
    # Small warmup remains bounded even for the 220K context configuration.
    # Deduplicate when a small configured context truncates the larger cases.
    sizes = sorted({min(size, context_length - decode_tokens) for size in sizes})
    for size in sizes:
        ids = (phrase * ((size + len(phrase) - 1) // len(phrase)))[:size]
        req = GenerateReqInput(
            input_ids=ids,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": decode_tokens,
                "ignore_eos": True,
            },
            stream=False,
        )
        started = time.perf_counter()
        result = None
        async for response in tokenizer_manager.generate_request(req, None):
            result = response
        if result is None:
            raise RuntimeError("qwen38_offload warmup returned no response")
        if isinstance(result, dict):
            meta = result.get("meta_info") or {}
            finish = meta.get("finish_reason") or {}
            if finish.get("type") == "abort":
                raise RuntimeError(f"qwen38_offload warmup aborted: {finish}")
        logger.info(
            "Qwen38 offload warmup: prefill=%d decode=%d elapsed=%.3fs",
            size, decode_tokens, time.perf_counter() - started,
        )
