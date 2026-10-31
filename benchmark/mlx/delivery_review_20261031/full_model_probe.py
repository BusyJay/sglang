"""Same-checkpoint probe, run with explicit baseline/candidate PYTHONPATH."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from sglang.srt.hardware_backend.mlx.models import glm5_next as g
from sglang.srt.hardware_backend.mlx.mtp import load_mtp_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing, clone_cache

p = argparse.ArgumentParser()
p.add_argument("--model-dir", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--steps", type=int, default=2)
p.add_argument("--length", type=int, default=128)
args = p.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_mtp_model(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
prompt_ids = tokenizer.apply_chat_template(
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
prompt = mx.array([prompt_ids])
report = dict(
    device=mx.device_info(),
    mlx=mx.__version__,
    source=g.__file__,
    source_sha256=hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),
    steps=args.steps,
    prompt_ids=prompt_ids,
    forced=[],
    generation=[],
)


def save():
    report["peak_bytes"] = mx.get_peak_memory()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def digest(array):
    return hashlib.sha256(np.asarray(array.astype(mx.float32)).tobytes()).hexdigest()


def descriptor(array):
    if array is None:
        return None
    return dict(sha256=digest(array), dtype=str(array.dtype), shape=array.shape)


for prefix in (1, 7, len(prompt_ids), 2050):
    base = model.make_cache()
    ids = (prompt_ids * ((prefix + len(prompt_ids) - 1) // len(prompt_ids)))[:prefix]
    for start in range(0, prefix, 256):
        mx.eval(model(mx.array([ids[start : start + 256]]), base))
    for rows in (2, 3, 5, 9):
        chain = mx.array([list(range(31, 31 + rows))])
        routes = []
        original = g.mlx_glm_router

        def route(*a, _routes=routes, _original=original, **kw):
            out = _original(*a, **kw)
            if out is not None:
                _routes.append(out)
            return out

        with patch.object(g, "mlx_glm_router", route):
            out = model.verify_mtp_rows(chain, [clone_cache(c) for c in base])
            mx.eval(out[:2], [[c.cache for c in row] for row in out[2]], routes)
        np.savez_compressed(
            args.output.with_name(args.output.stem + f"_p{prefix}_r{rows}.npz"),
            logits=np.asarray(out[0].astype(mx.float32)),
            hidden=np.asarray(out[1].astype(mx.float32)),
        )
        row = dict(
            prefix=prefix,
            rows=rows,
            logits=digest(out[0]),
            hidden=digest(out[1]),
            caches=[
                [
                    dict(
                        offset=c.offset,
                        values=[descriptor(v) for v in c.cache],
                    )
                    for c in s
                ]
                for s in out[2]
            ],
            routes=[[digest(v) for v in route] for route in routes],
            samples_ms=[],
        )
        del out, routes
        for trial in range(5):
            cache = [clone_cache(c) for c in base]
            started = time.perf_counter()
            out = model.verify_mtp_rows(chain, cache)
            mx.eval(out[:2], [[c.cache for c in s] for s in out[2]])
            if trial:
                row["samples_ms"].append((time.perf_counter() - started) * 1000)
            del out, cache
        row["median_ms"] = statistics.median(row["samples_ms"])
        report["forced"].append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ("caches", "routes")}),
            flush=True,
        )
        save()
    del base

for prefix in (len(prompt_ids), 2050):
    for repeat in range(2):
        engine = MlxMTPServing(model, args.steps)
        cache = model.make_cache()
        sequence = (prompt_ids * ((prefix + len(prompt_ids) - 1) // len(prompt_ids)))[
            :prefix
        ]
        for start in range(0, prefix, 256):
            hidden = engine.prefill(
                "probe", mx.array([sequence[start : start + 256]]), cache
            )
        token = mx.argmax(model.lm_head(hidden[:, -1:]), -1)
        tokens = [token.item()]
        started = time.perf_counter()
        for _ in range(args.length - 1):
            engine.set_remaining("probe", args.length - len(tokens))
            token = mx.argmax(engine.decode("probe", token, cache), -1)[:, None]
            tokens.append(token.item())
        mx.eval(engine.states["probe"].head.cache)
        elapsed = time.perf_counter() - started
        row = dict(
            repeat=repeat,
            prefix=prefix,
            tokens=tokens,
            decode_s=elapsed,
            tokens_per_s=(len(tokens) - 1) / elapsed,
            **engine.stats(),
        )
        report["generation"].append(row)
        print(json.dumps(row), flush=True)
        engine.remove("probe")
        save()
