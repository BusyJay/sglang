# SPDX-License-Identifier: Apache-2.0
"""Isolate dense head and KDA sharing with warm, alternating same-model controls.

All candidates retain the prior FFN/cache defaults. Reference attention uses
58a6c24e3f's row loop; selectors and method patches are diagnostic only.
"""

import argparse
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def reference_attention_rows(self, x, cache):
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.models.glm5_next import NativeCache

    attended, snapshots = [], []
    for j in range(x.shape[1]):
        normed, post, residual = self._branch_inputs(
            x[:, j : j + 1], "attn", self.input_layernorm
        )
        attended.append(
            self._branch_output(self.self_attn(normed, cache), post, residual)
        )
        snapshot = NativeCache(4)
        snapshot.cache = [
            v.view(v.dtype) if v is not None else None for v in cache.cache
        ]
        snapshot.offset = cache.offset
        snapshots.append(snapshot)
    normed, post, residual = self._branch_inputs(
        mx.concatenate(attended, axis=1), "ffn", self.post_attention_layernorm
    )
    return self._branch_output(
        self.mlp(normed, batch_fusion=True), post, residual
    ), snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 3, 5, 9])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.output_length < 2
        or any(not 1 <= n <= 17 for n in args.rows)
    ):
        parser.error("positive repeats, output-length >= 2 and rows 1..17 required")
    import mlx.core as mx
    from transformers import AutoTokenizer

    from sglang.srt.hardware_backend.mlx.models import glm5_next as g
    from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
    from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing, clone_cache

    recommended = mx.device_info().get("max_recommended_working_set_size", 0)
    if recommended:
        mx.set_wired_limit(recommended)
    model = load_mtp_model(args.model_dir, verify_mode="ffn", cache_latent=True)
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
    candidates = ["reference", "head", "kda", "combined"]
    kernel = g.mlx_dense_rows

    @contextmanager
    def candidate(name):
        def select(x, weight):
            head = weight is model.lm_head.weight
            enabled = (
                name == "combined"
                or (head and name == "head")
                or (not head and name == "kda")
            )
            return kernel(x, weight) if enabled else None

        with patch.object(g, "mlx_dense_rows", select):
            if name in ("reference", "head"):
                with patch.object(
                    g.DecoderLayer, "verify_ffn_rows", reference_attention_rows
                ):
                    yield
            else:
                yield

    report = dict(
        device=mx.device_info(),
        mlx=mx.__version__,
        model_dir=str(args.model_dir),
        candidates=candidates,
        baseline="58a6c24e3f FFN+cache; native dense GEMVs",
        forced=[],
        generation=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def evaluate(chain, base):
        out = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
        mx.eval(out[:2], [[c.cache for c in row] for row in out[2]])
        return out

    def difference(actual, expected):
        return dict(
            equal=mx.array_equal(actual, expected).item(),
            max_abs=mx.max(
                mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            ).item(),
        )

    base = model.make_cache()
    mx.eval(model(prompt, base))
    for count in args.rows:
        chain = mx.array([list(range(31, 31 + count))])
        with candidate("reference"):
            reference = evaluate(chain, base)
        records = {}
        for name in candidates:
            with candidate(name):
                out = evaluate(chain, base)
                error = 0.0
                for got, want in zip(out[2], reference[2]):
                    for a, b in zip(got, want):
                        if a.offset != b.offset:
                            raise RuntimeError("Cache position mismatch")
                        for av, bv in zip(a.cache, b.cache):
                            if av is None or bv is None:
                                if av is not bv:
                                    raise RuntimeError("Cache structure mismatch")
                            else:
                                error = max(error, difference(av, bv)["max_abs"])
                records[name] = dict(
                    candidate=name,
                    rows=count,
                    logits=difference(out[0], reference[0]),
                    hidden=difference(out[1], reference[1]),
                    max_cache_abs=error,
                    samples_ms=[],
                )
                evaluate(chain, base)
        for repeat in range(args.repeats):
            for name in candidates if repeat % 2 == 0 else candidates[::-1]:
                with candidate(name):
                    start = time.perf_counter()
                    evaluate(chain, base)
                    records[name]["samples_ms"].append(
                        1000 * (time.perf_counter() - start)
                    )
        for record in records.values():
            record["median_ms"] = statistics.median(record["samples_ms"])
            report["forced"].append(record)
            print(json.dumps(record), flush=True)
        save()
    reference_tokens = None
    for repeat in range(2):
        for name in candidates if repeat % 2 == 0 else candidates[::-1]:
            with candidate(name):
                engine = MlxMTPServing(model, 1)
                cache = model.make_cache()
                hidden = engine.prefill("ab", prompt, cache)
                token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
                tokens = [token.item()]
                start = time.perf_counter()
                for _ in range(args.output_length - 1):
                    engine.set_remaining("ab", args.output_length - len(tokens))
                    token = mx.argmax(engine.decode("ab", token, cache), -1)[:, None]
                    tokens.append(token.item())
                mx.eval(engine.states["ab"].head.cache)
                elapsed = time.perf_counter() - start
                if reference_tokens is None:
                    reference_tokens = tokens
                row = dict(
                    candidate=name,
                    repeat=repeat,
                    tokens_equal=tokens == reference_tokens,
                    token_hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    decode_tokens_per_s=(len(tokens) - 1) / elapsed,
                    **engine.stats(),
                )
                engine.remove("ab")
                report["generation"].append(row)
                print(json.dumps(row), flush=True)
                save()
    report["peak_bytes"] = mx.get_peak_memory()
    save()


if __name__ == "__main__":
    main()
