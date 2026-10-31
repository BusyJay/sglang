# SPDX-License-Identifier: Apache-2.0
"""Same-checkpoint MTP verification A/B with serving-equivalent process resources.

Reports numerical changes as measurements instead of assuming batched parity.
Use the HTTP benchmark afterwards for sampling/prefix/grammar contracts.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.output_length < 2
        or any(not 1 <= r <= 17 for r in args.rows)
    ):
        parser.error("Require positive repeats, output-length >= 2, and rows 1..17")
    import mlx.core as mx
    from transformers import AutoTokenizer

    from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
    from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing, clone_cache

    # Match the serving runner's process-local default; no OS limit is changed.
    recommended = mx.device_info().get("max_recommended_working_set_size", 0)
    if recommended:
        mx.set_wired_limit(recommended)
    model = load_mtp_model(args.model_dir)
    print("Loaded target and MTP", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt = mx.array(
        [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Explain how a hash table resolves collisions, with a small example.",
                    }
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        ]
    )
    candidates = [("serial", False), ("head", False), ("ffn", False), ("ffn", True)]
    report = dict(
        model_dir=str(args.model_dir),
        mlx=mx.__version__,
        device=mx.device_info(),
        candidates=candidates,
        forced=[],
        generation=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def configure(mode, latent):
        model.configure_mtp_verification(mode)
        model.configure_latent_projection_cache(latent)

    def difference(a, b):
        delta = a.astype(mx.float32) - b.astype(mx.float32)
        return dict(
            equal=mx.array_equal(a, b).item(),
            max_abs=mx.max(mx.abs(delta)).item(),
            rms=mx.sqrt(mx.mean(delta * delta)).item(),
            greedy_equal=mx.array_equal(mx.argmax(a, -1), mx.argmax(b, -1)).item(),
        )

    base = model.make_cache()
    mx.eval(model(prompt, base))
    for count in args.rows:
        print(f"Starting forced panel: {count} rows", flush=True)
        chain = mx.array([list(range(31, 31 + count))])
        configure("serial", False)
        reference = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
        mx.eval(reference[:2], [[c.cache for c in row] for row in reference[2]])
        records = {}
        for mode, latent in candidates:
            print(f"  candidate={mode}, latent={latent}", flush=True)
            configure(mode, latent)
            logits, hidden, snapshots = model.verify_mtp_rows(
                chain, [clone_cache(c) for c in base]
            )
            mx.eval(logits, hidden, [[c.cache for c in row] for row in snapshots])
            cache_error = 0.0
            for got, want in zip(snapshots, reference[2]):
                for a, b in zip(got, want):
                    if a.offset != b.offset:
                        raise RuntimeError("Cache position mismatch")
                    for av, bv in zip(a.cache, b.cache):
                        if av is None or bv is None:
                            if av is not bv:
                                raise RuntimeError("Cache structure mismatch")
                        else:
                            cache_error = max(
                                cache_error,
                                mx.max(
                                    mx.abs(
                                        av.astype(mx.float32) - bv.astype(mx.float32)
                                    )
                                ).item(),
                            )
            records[(mode, latent)] = dict(
                rows=count,
                mode=mode,
                cache_latent=latent,
                logits=difference(logits, reference[0]),
                hidden=difference(hidden, reference[1]),
                max_cache_abs=cache_error,
                samples_ms=[],
            )
            for _ in range(3):
                mx.eval(
                    model.verify_mtp_rows(chain, [clone_cache(c) for c in base])[:2]
                )
        for repeat in range(args.repeats):
            order = candidates if repeat % 2 == 0 else candidates[::-1]
            for mode, latent in order:
                configure(mode, latent)
                # One warm call keeps allocation/retained-weight initialization out of the sample.
                mx.eval(
                    model.verify_mtp_rows(chain, [clone_cache(c) for c in base])[:2]
                )
                start = time.perf_counter()
                out = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
                mx.eval(out[:2], [[c.cache for c in row] for row in out[2]])
                records[(mode, latent)]["samples_ms"].append(
                    1000 * (time.perf_counter() - start)
                )
        for record in records.values():
            record["median_ms"] = statistics.median(record["samples_ms"])
            report["forced"].append(record)
        save()
    baseline_tokens = None
    for repeat in range(2):
        for mode, latent in candidates if repeat == 0 else candidates[::-1]:
            configure(mode, latent)
            engine = MlxMTPServing(model, 1)
            cache = model.make_cache()
            hidden = engine.prefill("ab", prompt, cache)
            token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
            mx.eval(token)
            tokens = [token.item()]
            start = time.perf_counter()
            for _ in range(args.output_length - 1):
                engine.set_remaining("ab", args.output_length - len(tokens))
                token = mx.argmax(engine.decode("ab", token, cache), -1)[:, None]
                tokens.append(token.item())
            mx.eval(engine.states["ab"].head.cache)
            elapsed = time.perf_counter() - start
            if baseline_tokens is None:
                baseline_tokens = tokens
            engine.remove("ab")
            report["generation"].append(
                dict(
                    mode=mode,
                    cache_latent=latent,
                    repeat=repeat,
                    tokens_equal=tokens == baseline_tokens,
                    token_hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    decode_tokens_per_s=(len(tokens) - 1) / elapsed,
                    **engine.stats(),
                )
            )
            save()
    report["peak_bytes"] = mx.get_peak_memory()
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
