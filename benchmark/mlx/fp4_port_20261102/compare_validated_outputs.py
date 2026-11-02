"""Compare a candidate with frozen outputs without generating an AR reference."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx

from sglang.srt.hardware_backend.mlx.models.glm5_fp4 import load_fp4_model
from sglang.srt.hardware_backend.mlx.mtp_serving import MlxMTPServing

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--compare-result", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
refpath = args.reference or Path(
    json.loads((root / "validated_outputs/CURRENT.json").read_text())["reference"]
)
if not refpath.is_absolute():
    refpath = root / "validated_outputs" / refpath
if not refpath.is_absolute():
    refpath = root / "validated_outputs" / refpath
if not refpath.is_absolute():
    refpath = root / "validated_outputs" / refpath
ref = json.loads(refpath.read_text())
identity = ref["qualification"]["checkpoint"]
for name, key in (
    ("config.json", "config_sha256"),
    ("model.safetensors.index.json", "index_sha256"),
):
    if (
        hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
        != identity[key]
    ):
        raise ValueError(
            "Checkpoint identity changed; this baseline cannot qualify its semantics"
        )
if args.checkpoint.name != identity["revision"]:
    raise ValueError("Checkpoint revision differs from frozen reference")
for case in ref["cases"]:
    digest = hashlib.sha256(json.dumps(case["output_ids"]).encode()).hexdigest()
    if digest != case["output_sha256"]:
        raise ValueError("Reference output checksum mismatch")
if args.compare_result:
    candidate = json.loads(args.compare_result.read_text())
    samples = candidate.get("samples", candidate.get("results", []))
    comparisons = []
    for case in ref["cases"]:
        rows = [
            v
            for v in samples
            if v.get("id") == case["id"]
            or (
                f"{v.get('context')}-{v.get('text_id')}" == case["id"]
                and v.get("arm") == "mtp2"
            )
        ]
        if len(rows) != 1:
            raise ValueError("Exactly one candidate result required per frozen case")
        ids = rows[0].get("output_ids", rows[0].get("tokens"))
        exact = ids == case["output_ids"]
        first = next(
            (i for i, (a, b) in enumerate(zip(ids, case["output_ids"])) if a != b), None
        )
        comparisons.append(dict(id=case["id"], exact=exact, first_difference=first))
    args.output.write_text(
        json.dumps(dict(reference=str(refpath), comparisons=comparisons), indent=2)
        + "\n"
    )
    print("STORED_OUTPUT_COMPARISON", comparisons)
    raise SystemExit(0 if all(v["exact"] for v in comparisons) else 1)
if args.validate_only:
    print("REFERENCE_VALID", str(refpath), len(ref["cases"]))
    raise SystemExit(0)
mx.set_memory_limit(188 * 2**30)
mx.set_wired_limit(188 * 2**30)
mx.set_cache_limit(64 * 2**20)
model, _ = load_fp4_model(args.checkpoint, enable_mtp=True)
model.configure_optimizations("affine_fused")
model.configure_mtp_verification("ffn")
model.configure_latent_projection_cache(True)
model.prepare_for_inference()
engine = MlxMTPServing(model, 2)
results = []
for case in ref["cases"]:
    engine.clear()
    cache = model.make_cache()
    prompt = case["input_ids"]
    for i in range(0, len(prompt), 256):
        h = engine.prefill(case["id"], mx.array([prompt[i : i + 256]], mx.int32), cache)
        mx.eval(h, [c.cache for c in cache])
    token = int(model.target_tokens(h[:, -1:]).item())
    ids = [token]
    timings = []
    for i in range(len(case["output_ids"]) - 1):
        engine.set_remaining(case["id"], len(case["output_ids"]) - i - 1)
        start = time.perf_counter()
        v = engine.decode_batch(
            [case["id"]],
            mx.array([[token]], mx.int32),
            [cache],
            [token],
            token_only=True,
        )
        mx.eval(v, [c.cache for c in cache])
        token = int(v.item())
        timings.append(time.perf_counter() - start)
        ids.append(token)
    first = next(
        (i for i, (a, b) in enumerate(zip(ids, case["output_ids"])) if a != b), None
    )
    row = dict(
        id=case["id"],
        exact=first is None,
        first_difference=first,
        output_ids=ids,
        raw_seconds=timings,
        stats=engine.stats(),
    )
    results.append(row)
    args.output.write_text(
        json.dumps(dict(reference=str(refpath), results=results), indent=2) + "\n"
    )
    print("CANDIDATE", case["id"], row["exact"], first, flush=True)
    engine.remove(case["id"])
    del cache, h, v
    mx.clear_cache()
if not all(row["exact"] for row in results):
    raise SystemExit(1)
