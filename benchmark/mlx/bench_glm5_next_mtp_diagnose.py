# SPDX-License-Identifier: Apache-2.0
"""Separate MTP acceptance, forced-chain verification cost and fast-path hits.

Loads one local checkpoint. The uninstrumented fixed-chain control removes
acceptance as a variable; synchronized attribution is reported separately.
"""

import argparse
import collections
import contextlib
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    args = parser.parse_args()
    import mlx.core as mx
    from transformers import AutoTokenizer

    from sglang.srt.hardware_backend.mlx.models import glm5_next as g
    from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
    from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing, clone_cache

    mx.set_wired_limit(240 * 1024**3)
    model = load_mtp_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt = mx.array(
        [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Explain how a hash table resolves collisions, "
                            "with a small example."
                        ),
                    }
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        ]
    )
    report = dict(acceptance=[], forced_chain=[], paths=[], attribution=[])
    baseline = None

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def generate(steps, count=128):
        engine = MlxMTPServing(model, steps)
        cache = model.make_cache()
        hidden = engine.prefill("r", prompt, cache)
        token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
        tokens = [token.item()]
        started = time.perf_counter()
        while len(tokens) < count:
            engine.set_remaining("r", count - len(tokens))
            logits = engine.decode("r", token, cache)
            token = mx.argmax(logits, -1)[:, None]
            tokens.append(token.item())
        mx.eval(engine.states["r"].head.cache)
        elapsed = time.perf_counter() - started
        engine.remove("r")
        return tokens, engine.stats(), elapsed

    for steps in args.steps:
        tokens, stats, elapsed = generate(steps)
        if baseline is None:
            baseline = tokens
        if tokens != baseline:
            raise RuntimeError(f"Token mismatch for step {steps}")
        row = dict(
            steps=steps,
            decode_tokens_per_s=127 / elapsed,
            tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            **stats,
        )
        report["acceptance"].append(row)
        print("acceptance", json.dumps(row), flush=True)
        save()

    base_cache = model.make_cache()
    mx.eval(model(prompt, base_cache))
    # The same true AR token chain is used at every width. No drafts can be
    # rejected in this control, and no MTP head runs inside its timed region.
    for steps in args.steps:
        width = steps + 1
        chain = mx.array([baseline[:width]])
        reference = None
        for mode in ("serial", "verify"):
            samples = []
            for repeat in range(4):
                cache = [clone_cache(c) for c in base_cache]
                started = time.perf_counter()
                if mode == "serial":
                    logits = mx.concatenate(
                        [
                            model.forward_with_hidden(chain[:, j : j + 1], cache)[0]
                            for j in range(width)
                        ],
                        axis=1,
                    )
                else:
                    logits, _, snapshots = model.verify_mtp_rows(chain, cache)
                mx.eval(logits, [c.cache for c in cache])
                elapsed = time.perf_counter() - started
                if reference is None:
                    reference = logits
                if not mx.array_equal(reference, logits).item():
                    raise RuntimeError(
                        f"Forced-chain logits mismatch for width {width}"
                    )
                if repeat:
                    samples.append(elapsed)
            row = dict(
                steps=steps,
                width=width,
                mode=mode,
                median_ms=1000 * statistics.median(samples),
                ms_per_row=1000 * statistics.median(samples) / width,
            )
            report["forced_chain"].append(row)
            print("forced_chain", json.dumps(row), flush=True)
            save()

    for steps in (1, 2, 4, 8):
        hits = collections.Counter()

        def wrap(name, original):
            def call(*values, **kwargs):
                out = original(*values, **kwargs)
                x = values[1] if name == "_compiled_quantized_ffn" else values[0]
                key = (name, str(x.shape), str(x.dtype), out is not None)
                hits[key] += 1
                return out

            return call

        with contextlib.ExitStack() as stack:
            for name in (
                "mlx_hc_collapse_norm",
                "mlx_kda_decode",
                "mlx_glm_router",
                "mlx_glm_down_combine",
                "_compiled_quantized_ffn",
            ):
                stack.enter_context(patch.object(g, name, wrap(name, getattr(g, name))))
            cache = [clone_cache(c) for c in base_cache]
            logits, _, _ = model.verify_mtp_rows(
                mx.array([baseline[: steps + 1]]), cache
            )
            mx.eval(logits)
        for (op, shape, dtype, fused), calls in hits.items():
            report["paths"].append(
                dict(
                    steps=steps,
                    op=op,
                    shape=shape,
                    dtype=dtype,
                    fused=fused,
                    calls=calls,
                )
            )
        save()

    # Synchronizing every head call distorts normal overlap. Keep this
    # attribution separate from uninstrumented timings and HTTP throughput.
    for steps in (1, 2, 4, 8):
        costs = collections.defaultdict(float)
        counts = collections.Counter()
        old_head = g.MTPHead.__call__
        old_verify = g.Model.verify_mtp_rows

        def head(self, hidden, embeddings, cache):
            mx.eval(hidden, embeddings, cache.cache)
            started = time.perf_counter()
            out = old_head(self, hidden, embeddings, cache)
            mx.eval(out, cache.cache)
            costs["head_s"] += time.perf_counter() - started
            counts["head_calls"] += 1
            return out

        def verify(self, inputs, cache):
            mx.eval(inputs, [c.cache for c in cache])
            started = time.perf_counter()
            result = old_verify(self, inputs, cache)
            mx.eval(result[0], result[1], [c.cache for c in cache])
            costs["verify_s"] += time.perf_counter() - started
            counts["verify_calls"] += 1
            return result

        with (
            patch.object(g.MTPHead, "__call__", head),
            patch.object(g.Model, "verify_mtp_rows", verify),
        ):
            _, stats, elapsed = generate(steps, count=32)
        # Head attribution includes one prefill head call, identified by its
        # multi-token shape; totals are intentionally not a speed benchmark.
        row = dict(
            steps=steps,
            decode_s=elapsed,
            costs=dict(costs),
            calls=dict(counts),
            stats=stats,
            scope="synchronized attribution; includes head priming",
        )
        report["attribution"].append(row)
        print("attribution", json.dumps(row), flush=True)
        save()


if __name__ == "__main__":
    main()
