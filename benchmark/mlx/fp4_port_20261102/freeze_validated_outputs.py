"""Freeze qualified reasoning outputs; subsequent candidates need no AR rerun."""

import hashlib
import json
import platform
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / "reasoning_aligned_compare.json").read_text())
assert len(result["reasoning_comparisons"]) == 6
assert all(v["exact"] for v in result["reasoning_comparisons"])
checkpoint = Path(result["checkpoint"])
tokenizer = load_tokenizer(checkpoint)
texts = [
    "Explain why the sky is blue.",
    "Write a Python function that computes the first n Fibonacci numbers, with an explanation.",
    "请介绍机器学习中的过拟合，并给出避免过拟合的方法。",
]
cases = []
for row in result["samples"]:
    if row["arm"] != "ar":
        continue
    text = texts[row["text_id"]]
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=text)], tokenize=True, add_generation_prompt=True
    )
    if row["context"] > len(prompt):
        pad = tokenizer.encode(
            "Background notes: "
            + "Machine learning uses data and algorithms. " * row["context"]
        )
        prompt = pad[: row["context"] - len(prompt)] + prompt
    assert len(prompt) == row["context"]
    cases.append(
        dict(
            id=f"{row['context']}-{row['text_id']}",
            messages=[dict(role="user", content=text)],
            input_ids=prompt,
            output_ids=row["tokens"],
            output_text=row["text"],
            output_sha256=row["hash"],
        )
    )
files = [
    "python/sglang/srt/hardware_backend/mlx/models/glm5_fp4.py",
    "python/sglang/srt/hardware_backend/mlx/models/mlx_fp4_cache.py",
    "python/sglang/srt/hardware_backend/mlx/models/glm5_next.py",
    "python/sglang/srt/hardware_backend/mlx/mtp_serving.py",
    "python/sglang/kernels/ops/attention/linear/mlx_kda.py",
    "python/sglang/kernels/ops/gemm/mlx_dense_rows.py",
    "python/sglang/kernels/ops/gemm/mlx_fp8_rows.py",
]
repo = ROOT.parents[2]
source = {
    name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files
}
identity = {
    "revision": checkpoint.name,
    "config_sha256": hashlib.sha256(
        (checkpoint / "config.json").read_bytes()
    ).hexdigest(),
    "index_sha256": hashlib.sha256(
        (checkpoint / "model.safetensors.index.json").read_bytes()
    ).hexdigest(),
}
qualification = dict(
    scope="six deterministic reasoning prefixes, 128 generated tokens, native MTP2 equals AR; projection independent numerical checks and decoded-weight cache roundtrip. Not a complete-answer/HTTP/general accuracy certification.",
    invalid_evidence="KV-B cache restore bug affected earlier warm runs; exclude all pre-fix warm outputs.",
    numerical=json.loads((ROOT / "mxfp8_independent_reference.json").read_text()),
    configuration=dict(
        mtp_steps=2,
        optimizations="affine_fused",
        verification="ffn",
        tf32=False,
        prefill_chunk=256,
        latent_projection_cache=True,
    ),
    source=source,
    checkpoint=identity,
    device=result["device"],
    platform=platform.platform(),
    mlx=mx.__version__,
)
reference = dict(schema=1, qualification=qualification, cases=cases)
digest = hashlib.sha256(
    json.dumps(reference, sort_keys=True, ensure_ascii=False).encode()
).hexdigest()
folder = ROOT / "validated_outputs" / checkpoint.name / digest[:24]
folder.mkdir(parents=True, exist_ok=True)
(folder / "reference.json").write_text(
    json.dumps(reference, indent=2, ensure_ascii=False) + "\n"
)
shutil.copy2(
    ROOT / "reasoning_aligned_compare.json", folder / "raw_ar_mtp_compare.json"
)
for name in files:
    target = folder / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / name, target)
(ROOT / "validated_outputs" / "CURRENT.json").write_text(
    json.dumps(dict(reference=str((folder / "reference.json").relative_to(ROOT / "validated_outputs")), sha256=digest), indent=2)
    + "\n"
)
print("VALIDATED_REFERENCE", folder / "reference.json", flush=True)
