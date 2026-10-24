"""Run the established paired serving workload with explicit candidate switches.

Inside qwen38-csf-defaults-20261023, run with baseline, hc_cta, or all.
Outputs go to the writable /results mount; each invocation shuts its server down.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

variant = sys.argv[1]
if variant not in ("baseline", "baseline_repeat", "hc_cta", "all"):
    raise ValueError(variant)
root = Path("/results")
source = Path(__file__).resolve().parent
output = root / "all_optimizations_20261024" / variant
output.mkdir(parents=True, exist_ok=True)
environment = json.loads((source / "serving-environment.json").read_text())
environment.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    SGLANG_OPT_HC_MIX_SPLIT_K="1" if variant in ("hc_cta", "all") else "0",
    SGLANG_OPT_QWEN38_FILL_CTA_TABLE="1" if variant in ("hc_cta", "all") else "0",
    SGLANG_OPT_QWEN38_MXFP8_SMALL_M="1" if variant == "all" else "0",
    ROOT_CSF_CHAT_AFTER="5",
)
os.environ.update(environment)
(output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
reference = source / "serving-command.json"
(output / "command.json").write_text(reference.read_text())
sys.argv = [
    __file__,
    "--reference",
    str(reference),
    "--output",
    str(output),
    "--modes",
    "default",
]
spec = importlib.util.spec_from_file_location(
    "checked_ab", root / "benchmark_checked.py"
)
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)
try:
    checked.benchmark.main()
except Exception as error:
    destination = output / "performance.json"
    report = json.loads(destination.read_text()) if destination.exists() else {}
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise
