# SPDX-License-Identifier: Apache-2.0
"""Same-weight AR/MTP experiment on a local original-layout GLM checkpoint.

Uses row-exact verification and private caches, independent of HTTP serving.
Run one process at a time; a full checkpoint needs about 200 GiB resident.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def check_verify(model, prompt):
    import mlx.core as mx

    serial, block = model.make_cache(), model.make_cache()
    mx.eval(model(prompt, serial), model(prompt, block))
    for step in range(16):
        root = mx.array([[31 + 2 * step]])
        draft = mx.array([[32 + 2 * step]])
        first, h1 = model.forward_with_hidden(root, serial)
        second, h2 = model.forward_with_hidden(draft, serial)
        logits, hidden, _ = model.verify_mtp(root, draft, block)
        for actual, expected in (
            (logits, mx.concatenate([first, second], 1)),
            (hidden, mx.concatenate([h1, h2], 1)),
        ):
            if not mx.array_equal(actual, expected).item():
                raise RuntimeError(f"Verification differs from AR at pair {step}")
        for a, b in zip(serial, block):
            if a.offset != b.offset:
                raise RuntimeError("Verification cache offset mismatch")
            for av, bv in zip(a.cache, b.cache):
                if av is None or bv is None:
                    same = av is bv
                else:
                    same = mx.array_equal(av, bv).item()
                if not same:
                    raise RuntimeError("Verification cache value mismatch")
    report = dict(positions=32, logits_bitwise_equal=True, caches_bitwise_equal=True)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wired-limit-gb", type=int)
    parser.add_argument("--text-prompts", action="store_true")
    parser.add_argument("--check-verify", action="store_true")
    args = parser.parse_args()
    if args.output_length < 1 or args.repeats < 1:
        parser.error("Require positive output length and repeats")

    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.mtp import (
        greedy_mtp_generate,
        load_mtp_model,
    )

    if args.wired_limit_gb:
        mx.set_wired_limit(args.wired_limit_gb * 1024**3)
    started = time.perf_counter()
    model = load_mtp_model(args.model_dir)
    print(f"Loaded target and MTP in {time.perf_counter() - started:.2f}s", flush=True)
    prompt = mx.array([[10, 11, 12, 13] * 4])
    prompts = {"synthetic": prompt}
    if args.text_prompts:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        prompts = {
            name: mx.array(
                [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                    )
                ]
            )
            for name, text in (
                (
                    "english",
                    "Explain how a hash table resolves collisions, "
                    "with a small example.",
                ),
                (
                    "chinese",
                    "请解释为什么推理时减少权重读取可以提升速度，并举一个例子。",
                ),
                (
                    "code",
                    "Write a Python function that merges overlapping intervals. "
                    "Explain its complexity.",
                ),
            )
        }
    parity = check_verify(model, prompt) if args.check_verify else None
    rows = []
    for name, prompt in prompts.items():
        baseline = None
        for repeat in range(args.repeats):
            modes = ("ar", "mtp") if repeat % 2 == 0 else ("mtp", "ar")
            for mode in modes:
                started = time.perf_counter()
                if mode == "ar":
                    cache = model.make_cache()
                    tokens = []
                    inputs = prompt
                    for _ in range(args.output_length):
                        inputs = mx.argmax(model.forward_last_logits(inputs, cache), -1)
                        tokens.append(inputs.item())
                        if len(tokens) == 1:
                            first_at = time.perf_counter()
                    decode_s = time.perf_counter() - first_at
                    stats = dict(
                        prefill_s=first_at - started,
                        decode_tokens_per_s=(len(tokens) - 1) / decode_s,
                    )
                else:
                    result = greedy_mtp_generate(model, prompt, args.output_length)
                    tokens = result.tokens
                    stats = dict(
                        num_correct_drafts=result.num_correct_drafts,
                        num_proposed_drafts=result.num_proposed_drafts,
                        verify_ct=result.verify_ct,
                        prefill_s=result.prefill_s,
                        decode_tokens_per_s=(len(tokens) - 1) / result.decode_s,
                    )
                elapsed = time.perf_counter() - started
                if baseline is None:
                    baseline = tokens
                row = dict(
                    prompt=name,
                    input_tokens=prompt.shape[1],
                    mode=mode,
                    repeat=repeat,
                    elapsed_s=elapsed,
                    completion_tokens_per_s=len(tokens) / elapsed,
                    matches_ar=tokens == baseline,
                    token_sha256=hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                    **stats,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        dict(
                            scope="model-level singleton greedy; private caches",
                            verify_parity=parity,
                            output_tokens=args.output_length,
                            active_memory_gb=mx.get_active_memory() / 1024**3,
                            peak_memory_gb=mx.get_peak_memory() / 1024**3,
                            rows=rows,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                if tokens != baseline:
                    raise RuntimeError("MTP output differs from same-weight AR")


if __name__ == "__main__":
    main()
