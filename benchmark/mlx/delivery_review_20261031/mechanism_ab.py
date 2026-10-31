"""One-load alternating ablation against the actual input commit's methods."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from transformers import AutoTokenizer

from sglang.srt.hardware_backend.mlx import mtp_serving as serving
from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model

p = argparse.ArgumentParser()
p.add_argument("--model-dir", type=Path, required=True)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load(
    "_delivery_glm_baseline",
    args.baseline / "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
)
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_mtp_model(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
ids = tokenizer.apply_chat_template(
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
prompt = mx.array([ids])


def serial_kda(self, x, cache):
    projected = old.KDA.project_verify_rows(self, x)
    if projected is None:
        return None
    out, snapshots = [], []
    for j in range(x.shape[1]):
        out.append(
            self(
                x[:, j : j + 1],
                cache,
                projected=tuple(v[:, j : j + 1] for v in projected),
            )
        )
        snapshots.append(serving.clone_cache(cache))
    return mx.concatenate(out, 1), snapshots


def serial_mla(self, x, cache, **kw):
    out, snapshots = [], []
    for j in range(x.shape[1]):
        out.append(old.MLA.__call__(self, x[:, j : j + 1], cache))
        snapshots.append(serving.clone_cache(cache))
    return mx.concatenate(out, 1), snapshots


@contextmanager
def candidate(name):
    with ExitStack() as stack:
        if name not in ("kda", "combined"):
            stack.enter_context(patch.object(g.KDA, "verify_rows", serial_kda))
        if name not in ("mla", "combined"):
            stack.enter_context(patch.object(g.MLA, "verify_rows", serial_mla))
        if name not in ("moe", "combined"):
            stack.enter_context(
                patch.object(g, "_row_exact_affine_ffn", old._row_exact_affine_ffn)
            )
            stack.enter_context(patch.object(g.MoE, "__call__", old.MoE.__call__))
        if name not in ("head", "combined"):
            stack.enter_context(
                patch.object(
                    g.Model,
                    "draft_token",
                    lambda self, h: mx.argmax(
                        self.lm_head(self.mtp.shared_head.norm(h)), axis=-1
                    ),
                )
            )
        yield serving.MlxMTPServing


names = ["baseline", "head", "kda", "mla", "moe", "combined"]
report = dict(
    device=mx.device_info(),
    mlx=mx.__version__,
    scope="Single-request mechanism isolation; all use current serving. Independent checkout is the end-to-end baseline.",
    forced=[],
    generation=[],
)


def save():
    report["peak_bytes"] = mx.get_peak_memory()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def evaluate(chain, base):
    out = model.verify_mtp_rows(chain, [serving.clone_cache(c) for c in base])
    mx.eval(out[:2], [[c.cache for c in row] for row in out[2]])
    return out


for prefix in (len(ids), 2050):
    base = model.make_cache()
    tokens = (ids * ((prefix + len(ids) - 1) // len(ids)))[:prefix]
    for start in range(0, prefix, 256):
        mx.eval(model(mx.array([tokens[start : start + 256]]), base))
    for count in (2, 3, 5, 9):
        chain = mx.array([list(range(31, 31 + count))])
        reference = None
        records = {}
        for name in names:
            with candidate(name):
                out = evaluate(chain, base)
                if reference is None:
                    reference = out
                cache_equal = all(
                    a.offset == b.offset
                    and all(
                        av is bv
                        if av is None or bv is None
                        else mx.array_equal(av, bv).item()
                        for av, bv in zip(a.cache, b.cache)
                    )
                    for actual, expected in zip(out[2], reference[2])
                    for a, b in zip(actual, expected)
                )
                records[name] = dict(
                    candidate=name,
                    prefix=prefix,
                    rows=count,
                    logits_equal=mx.array_equal(out[0], reference[0]).item(),
                    hidden_equal=mx.array_equal(out[1], reference[1]).item(),
                    cache_equal=cache_equal,
                    samples_ms=[],
                )
        for repeat in range(4):
            for name in names if repeat % 2 == 0 else names[::-1]:
                with candidate(name):
                    started = time.perf_counter()
                    evaluate(chain, base)
                    records[name]["samples_ms"].append(
                        (time.perf_counter() - started) * 1000
                    )
        for row in records.values():
            row["median_ms"] = statistics.median(row["samples_ms"])
            report["forced"].append(row)
            print(json.dumps(row), flush=True)
        del reference, out
        save()
    del base

for budget in (0, 2):
    reference_tokens = None
    for repeat in range(3):
        for name in (
            (["baseline", "moe", "combined"] if budget == 0 else names)
            if repeat % 2 == 0
            else (["combined", "moe", "baseline"] if budget == 0 else names[::-1])
        ):
            with candidate(name) as cls:
                engine = cls(model, budget)
                cache = model.make_cache()
                hidden = engine.prefill("ab", prompt, cache)
                token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
                tokens = [token.item()]
                started = time.perf_counter()
                for _ in range(127):
                    engine.set_remaining("ab", 128 - len(tokens))
                    token = mx.argmax(engine.decode("ab", token, cache), -1)[:, None]
                    tokens.append(token.item())
                mx.eval(engine.states["ab"].head.cache)
                elapsed = time.perf_counter() - started
                if reference_tokens is None:
                    reference_tokens = tokens
                row = dict(
                    candidate=name,
                    steps=budget,
                    repeat=repeat,
                    tokens_equal=tokens == reference_tokens,
                    token_hash=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    decode_s=elapsed,
                    tokens_per_s=127 / elapsed,
                    **engine.stats(),
                )
                engine.remove("ab")
                report["generation"].append(row)
                print(json.dumps(row), flush=True)
                save()
