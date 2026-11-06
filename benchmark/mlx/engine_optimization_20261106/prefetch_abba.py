# SPDX-License-Identifier: Apache-2.0
"""Full-model alternating MLA projection A/B, one resident checkpoint.

Run from the workspace with the service stopped; no packages are installed.
Baseline DFlash scheduling comes from 00f5b6c890; candidate from this checkout.
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
parser.add_argument("--validation-only", action="store_true")
parser.add_argument("--kv-cache-dtype", choices=("bf16", "fp8"), default="bf16")
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
model.configure_precision(
    kv_cache_dtype=args.kv_cache_dtype, prefill_activation_dtype="bf16"
)
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

import sys
import types

source_path = "python/sglang/srt/hardware_backend/mlx/dflash_serving.py"
source = subprocess.check_output(
    ["git", "show", f"00f5b6c890:{source_path}"], text=True
)
baseline_module = types.ModuleType("baseline_dflash_serving")
sys.modules[baseline_module.__name__] = baseline_module
exec(compile(source, "00f5b6c890:" + source_path, "exec"), baseline_module.__dict__)  # noqa: S102
baseline_engine = baseline_module.MlxDFlashServing(
    model, None, 3, drafter=engine.drafter
)
candidate_engine = engine
cache_references = {}
data["validation_only"] = args.validation_only
data["kv_cache_dtype"] = args.kv_cache_dtype
data["baseline_engine"] = "00f5b6c890"
data["candidate_source_sha256"] = hashlib.sha256(
    Path(source_path).read_bytes()
).hexdigest()


def drain(serving):
    for state in serving.states.values():
        if state.prefetch is not None:
            result = state.prefetch.result
            mx.eval(result[0], result[3], [[c.cache for c in b] for b in result[2]])
    serving.clear()


def cohort(serving, arm, case, indices):
    drain(serving)
    serving.counters.clear()
    serving.timings.clear()
    serving.correct_by_depth.clear()
    serving.eligible_by_depth.clear()
    caches = [model.make_cache() for _ in indices]
    tokens, outputs = [], []
    started = time.perf_counter()
    for rid, (index, cache) in enumerate(zip(indices, caches)):
        for begin in range(0, len(prompts[index]), 256):
            hidden = serving.prefill(
                str(rid),
                mx.array([prompts[index][begin : begin + 256]], mx.int32),
                cache,
            )
            mx.eval(hidden, [c.cache for c in cache])
        token = int(model.target_tokens(hidden[:, -1:]).item())
        tokens.append(token)
        outputs.append([token])
    prefill_seconds = time.perf_counter() - started
    raw = []
    mx.reset_peak_memory()
    for step in range(args.tokens - 1):
        for rid in range(len(indices)):
            serving.set_remaining(str(rid), args.tokens - 1 - step)
        started = time.perf_counter()
        values = serving.decode_batch(
            [str(i) for i in range(len(indices))],
            mx.array(tokens, mx.int32)[:, None],
            caches,
            tokens,
            token_only=True,
            prefetch_allowed=True,
        )
        tokens = values.reshape(-1).tolist()
        raw.append(time.perf_counter() - started)
        for output, token in zip(outputs, tokens):
            output.append(token)
    offsets = [[c.offset for c in cache] for cache in caches]
    for index, offset in zip(indices, offsets):
        assert all(v == len(prompts[index]) + args.tokens - 1 for v in offset)
    row = {
        "arm": arm,
        "case": case,
        "indices": indices,
        "tokens": outputs,
        "decode_seconds": raw,
        "prefill_seconds": prefill_seconds,
        "aggregate_tok_s": len(indices) * len(raw) / sum(raw),
        "stats": copy.deepcopy(serving.stats()),
        "offsets": offsets,
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
    if args.validation_only:
        arrays = [
            [None if v is None else c[i] for c in cache for i, v in enumerate(c.cache)]
            for cache in caches
        ]
        if arm == "A_measured":
            cache_references[case] = arrays
        else:
            for owner, (got, expected) in enumerate(
                zip(arrays, cache_references[case])
            ):
                assert len(got) == len(expected)
                for slot, (a, b) in enumerate(zip(got, expected)):
                    if a is None or b is None:
                        assert a is b, (case, owner, slot)
                    else:
                        assert mx.array_equal(a, b).item(), (
                            case,
                            owner,
                            slot,
                            a.shape,
                            mx.max(
                                mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
                            ).item(),
                        )
            row["target_cache_bitwise_equal"] = True
    drain(serving)
    data["samples"].append(row)
    save()
    print(
        "COHORT", case, arm, round(row["aggregate_tok_s"], 3), row["stats"], flush=True
    )
    return row


try:
    for case, indices in [
        ("natural_pair", [0, 1]),
        ("aligned_pair", [0, 0]),
        ("single", [0]),
    ]:
        order = [
            ("A", baseline_engine),
            ("B", candidate_engine),
            ("B", candidate_engine),
            ("A", baseline_engine),
        ]
        if args.validation_only:
            order = order[:2]
        for arm, serving in order:
            model.configure_fixed_groups(True)
            if not args.validation_only:
                cohort(serving, arm + "_warm", case, indices)
            row = cohort(serving, arm + "_measured", case, indices)
            reference = next(
                r
                for r in data["samples"]
                if r["case"] == case and r["arm"] == "A_measured"
            )
            assert row["tokens"] == reference["tokens"], (arm, case, "tokens")
            assert (
                row["stats"]["num_correct_drafts"]
                == reference["stats"]["num_correct_drafts"]
            ), (arm, case, "accept")
finally:
    drain(baseline_engine)
    drain(candidate_engine)
    data["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    save()
