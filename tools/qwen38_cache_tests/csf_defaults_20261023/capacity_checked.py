"""Retain paired workloads, add near-limit context and post-resize exact answers."""

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
reference = Path(sys.argv[sys.argv.index("--reference") + 1])
output.mkdir(parents=True, exist_ok=True)
repo = Path("/workspace/sglang")
source_files = [
    "python/sglang/kernels/ops/attention/fla/" + name + ".py"
    for name in ("chunk", "chunk_delta_h", "chunk_fwd", "chunk_o", "l2norm", "wy_fast")
] + [
    "python/sglang/srt/layers/attention/linear/gdn_backend.py",
    "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py",
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
    "python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu",
    "python/sglang/srt/layers/moe/qwen38_csf_banks.py",
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
    "python/sglang/srt/layers/attention/qsa/sparse_attn.py",
    "python/sglang/srt/models/qwen4_exp.py",
]
(output / "source-manifest.json").write_text(
    json.dumps(
        {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in source_files
        },
        indent=2,
    )
    + "\n"
)
environment = json.loads(
    Path("/results/capacity-native-mamba5/environment.json").read_text()
)
os.environ.update(environment)
os.environ.update(
    ROOT_CSF_CHAT_AFTER="8",
    ROOT_CSF_PROBE="production",
    ROOT_CSF_OUTPUT=str(output / "logits.json"),
    PYTHONPATH="/workspace/sglang/python",
)
(output / "command.json").write_text(reference.read_text())
(output / "environment.json").write_text(
    json.dumps(
        {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("SGLANG_", "ROOT_", "PYTHONPATH", "PYTORCH_"))
        },
        indent=2,
    )
    + "\n"
)
spec = importlib.util.spec_from_file_location(
    "capacity_checked_base", "/results/benchmark_checked.py"
)
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)
benchmark = checked.benchmark
tree = ast.parse(inspect.getsource(benchmark.main))
body = tree.body[0].body
for index, node in enumerate(body):
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "report"
        for target in node.targets
    ):
        body[index:index] = ast.parse(
            'workloads.insert(-1, ("near_limit_219k", fixture(219000, 7), 32))'
        ).body
        break
else:
    raise RuntimeError("paired benchmark workload boundary changed")
exec(
    compile(ast.fix_missing_locations(tree), str(benchmark.__file__), "exec"),
    benchmark.__dict__,
)
try:
    benchmark.main()
except Exception as error:
    destination = output / "performance.json"
    report = json.loads(destination.read_text()) if destination.exists() else {}
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise
