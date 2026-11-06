# SPDX-License-Identifier: Apache-2.0
"""Full-model alternating MLA projection A/B, one resident checkpoint.

Run from the workspace with the service stopped; no packages are installed.
Baseline methods/operators come from 46b98b4030, candidate from this checkout.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer
from sglang.srt.hardware_backend.mlx.dflash_serving import MlxDFlashServing
from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--tokens", type=int, default=192)
parser.add_argument("--rounds", type=int, default=1)
parser.add_argument("--natural-limit", type=int, default=0)
parser.add_argument(
    "--history", choices=("short", "dense1k", "selected2k"), default="short"
)
args = parser.parse_args()
OUT = Path(args.output).resolve()
OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE = Path(__file__).resolve().parent
import urllib.request

try:
    response = urllib.request.urlopen("http://127.0.0.1:39007/health", timeout=1)
except (OSError, TimeoutError):
    pass
else:
    response.close()
    raise RuntimeError(
        "Stop the local 39007 service before this resident-model benchmark."
    )
TARGET = Path(
    "/Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
)
DRAFT = Path(
    "/Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"
)
data = {
    "device": mx.device_info(),
    "mlx": mx.__version__,
    "target": str(TARGET),
    "draft": str(DRAFT),
    "samples": [],
    "swap_before": subprocess.check_output(["sysctl", "vm.swapusage"], text=True),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "SGLANG_"))},
}


def save():
    (OUT / "fullmodel.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


budget = int(mx.device_info()["max_recommended_working_set_size"] * 0.78)
mx.set_memory_limit(budget)
mx.set_wired_limit(budget)
mx.set_cache_limit(2**30)
start = time.perf_counter()
model, config = load_fp4_model(TARGET, enable_mtp=False)
model.configure_optimizations("affine_fused")
model.configure_mtp_verification("ffn")
model.configure_latent_projection_cache(True)
model.configure_eval_groups(decode=2, prefill=4, verification=2)
model.prepare_for_inference()
model.configure_precision(kv_cache_dtype="bf16", prefill_activation_dtype="bf16")
mx.eval(model.parameters())
engine = MlxDFlashServing(model, DRAFT, 3)
tokenizer = load_tokenizer(TARGET)
texts = [
    "Explain why the sky is blue.",
    "Write a Python function that computes the first n Fibonacci numbers, with an explanation.",
    "请介绍机器学习中的过拟合，并给出避免过拟合的方法。",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": s}], tokenize=True, add_generation_prompt=True
    )
    for s in texts
]
data.update(
    load_seconds=time.perf_counter() - start,
    resident_bytes=mx.get_active_memory(),
    prompts=prompts,
)
save()
print("READY", data["load_seconds"], data["resident_bytes"] / 2**30, flush=True)


def drain():
    state = engine.states.get("p")
    if state is not None and state.prefetch is not None:
        result = state.prefetch.result
        mx.eval(
            result[0],
            result[3],
            [[c.cache for c in boundary] for boundary in result[2]],
        )
    engine.clear()


def run(arm, pi=0, count=192, *, width=3, natural=False):
    drain()
    engine.num_drafts = width
    engine.counters.clear()
    engine.correct_by_depth.clear()
    engine.eligible_by_depth.clear()
    engine.timings.clear()
    cache = model.make_cache()
    start = time.perf_counter()
    for begin in range(0, len(prompts[pi]), 256):
        hidden = engine.prefill(
            "p", mx.array([prompts[pi][begin : begin + 256]], mx.int32), cache
        )
        mx.eval(hidden, [c.cache for c in cache])
    token = int(model.target_tokens(hidden[:, -1:]).item())
    prefill_seconds = time.perf_counter() - start
    tokens = [token]
    raw = []
    mx.reset_peak_memory()
    eos = set(tokenizer.eos_token_ids)
    for step in range(count - 1):
        if natural and token in eos:
            break
        engine.set_remaining("p", count - 1 - step)
        start = time.perf_counter()
        before = engine.counters["verify_ct"]
        inputs = mx.array([[token]], mx.int32)
        output = (
            engine.decode_batch(
                ["p"], inputs, [cache], [token], token_only=True, prefetch_allowed=True
            )
            if width
            else model.forward_last_tokens(inputs, cache)
        )
        token = int(output.item())
        raw.append(
            {
                "seconds": time.perf_counter() - start,
                "verify": engine.counters["verify_ct"] - before,
                "accepted": engine.counters["num_correct_drafts"],
                "prefetch": engine.counters["prefetch_ct"],
            }
        )
        tokens.append(token)
    expected = len(prompts[pi]) + len(tokens) - 1
    assert all(c.offset == expected for c in cache), [c.offset for c in cache]
    row = {
        "arm": arm,
        "prompt": pi,
        "width": width,
        "natural": natural,
        "tokens": tokens,
        "raw": raw,
        "tok_s": len(raw) / sum(v["seconds"] for v in raw),
        "prefill_seconds": prefill_seconds,
        "stats": copy.deepcopy(engine.stats()),
        "offsets": [c.offset for c in cache],
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "finish": "eos" if token in eos else "length",
    }
    row["mla_cache_layout"] = [
        {
            "layer": i,
            "kind": type(c).__name__,
            "offset": c.offset,
            "arrays": [
                None if v is None else {"shape": v.shape, "dtype": str(v.dtype)}
                for v in c.cache
            ],
        }
        for i, c in enumerate(cache)
        if type(c).__name__ in ("MLACache", "FP8MLACache")
    ]
    drain()
    data["samples"].append(row)
    save()
    print("RUN", arm, pi, len(tokens), round(row["tok_s"], 3), row["stats"], flush=True)
    return row


import ast
import types

from sglang.srt.hardware_backend.mlx.models import glm5_next as glm

BASE = "46b98b4030"


def baseline_module(name, path):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], text=True)
    module = types.ModuleType(name)
    exec(compile(source, f"{BASE}:{path}", "exec"), module.__dict__)  # noqa: S102
    return module


dense = baseline_module(
    "baseline_dense", "python/sglang/kernels/ops/gemm/mlx_dense_rows.py"
)
ragged = baseline_module(
    "baseline_ragged", "python/sglang/kernels/ops/attention/mlx_ragged.py"
)
model_path = "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py"
source = subprocess.check_output(["git", "show", f"{BASE}:{model_path}"], text=True)
cls = next(
    n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MLA"
)
method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "attend_rows"
)
scope = dict(glm.__dict__)
scope["mlx_headed_dense_rows"] = dense.mlx_headed_dense_rows
scope["mlx_ragged_latent_attention"] = ragged.mlx_ragged_latent_attention
exec(  # noqa: S102 -- extract a method from the pinned repository baseline
    compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        f"{BASE}:{model_path}",
        "exec",
    ),
    scope,
)
baseline = scope["attend_rows"]
candidate = glm.MLA.attend_rows
data["baseline"] = BASE
paths = [
    model_path,
    "python/sglang/kernels/ops/gemm/mlx_dense_rows.py",
    "python/sglang/kernels/ops/attention/mlx_ragged.py",
]
data["candidate_source_sha256"] = {
    p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
}
if args.history != "short":
    histories = json.loads((EVIDENCE / "fixed_histories.json").read_text())
    prompts[0] = histories[args.history]
    prompts = prompts[:1]
data["history"] = args.history
data["prompts"] = prompts
try:
    for repetition in range(args.rounds):
        for arm, function in [
            ("A", baseline),
            ("B", candidate),
            ("B", candidate),
            ("A", baseline),
        ]:
            glm.MLA.attend_rows = function
            model.configure_fixed_groups(True)
            for prompt in range(len(prompts)):
                run(f"{repetition}_{arm}_warm", prompt, count=args.tokens)
                row = run(f"{repetition}_{arm}_measured", prompt, count=args.tokens)
                reference = next(
                    r
                    for r in data["samples"]
                    if r["arm"].endswith("A_measured") and r["prompt"] == prompt
                )
                assert row["tokens"] == reference["tokens"], (arm, prompt)
                assert (
                    row["stats"]["num_correct_drafts"]
                    == reference["stats"]["num_correct_drafts"]
                ), (arm, prompt)
finally:
    glm.MLA.attend_rows = candidate
    drain()
    data["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    save()
